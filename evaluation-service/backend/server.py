import argparse
import sys
from pathlib import Path

from airo_dataset_tools.data_parsers.pose import Pose
from cloth_tools.dataset.bookkeeping import datetime_for_filename, find_latest_sample_dir_with_observation_start
from flask import Flask, jsonify, render_template, request, send_file
from flask_cors import CORS
from loguru import logger
from werkzeug.utils import secure_filename

sys.path.append("..")
import datetime
import glob
import json
import os

import cv2
import numpy as np
import torch
from segment_anything import SamPredictor, sam_model_registry

app = Flask(__name__)
CORS(app)  #

from multiprocessing import Process, Queue


def euler_to_mat(roll, pitch, yaw):
    # Create rotation matrix from Euler angles
    R_x = np.array([[1, 0, 0], [0, np.cos(roll), -np.sin(roll)], [0, np.sin(roll), np.cos(roll)]])

    R_y = np.array([[np.cos(pitch), 0, np.sin(pitch)], [0, 1, 0], [-np.sin(pitch), 0, np.cos(pitch)]])

    R_z = np.array([[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]])

    R = np.dot(R_z, np.dot(R_y, R_x))

    return R


def sam_worker(q, ack, args):

    sam_checkpoint = "../weights/sam_vit_b_01ec64.pth"
    model_type = "vit_b"
    device = "cpu"

    cuda = args.cuda
    model_size = args.model
    scenes_directory = args.scenes_directory

    if model_size == "h":
        sam_checkpoint = "../weights/sam_vit_h_4b8939.pth"
        model_type = "vit_h"
    elif model_size == "l":
        sam_checkpoint = "../weights/sam_vit_l_0b3195.pth"
        model_type = "vit_l"

    if cuda and torch.cuda.is_available():
        logger.success("Segment-Anything worker running on GPU")
        device = "cuda"
    else:
        logger.warning("Segment-Anything worker running on CPU")

    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)

    predictor = SamPredictor(sam)

    while True:
        print("Waiting for data")
        data = q.get()
        print("Data received", data)

        manual = data["manual"]

        bbox = data["bbox"]
        positive_keypoints = data["positiveKeypoints"]
        negative_keypoints = data["negativeKeypoints"]

        x1 = bbox["x1"]
        y1 = bbox["y1"]
        x2 = bbox["x2"]
        y2 = bbox["y2"]

        if "outlierThreshold" in data and data["outlierThreshold"] is not None:
            outlier_threshold = float(data["outlierThreshold"])
        else:
            outlier_threshold = 0.5

        scene_name = data["sceneName"]

        image = cv2.imread(f"{scenes_directory}/{scene_name}/observation_result/image_left.png")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        predictor.set_image(image)

        input_box = np.array([[x1, y1, x2, y2]])

        input_point = []
        input_label = []

        for point in positive_keypoints:
            input_point.append(point)
            input_label.append(1)

        for point in negative_keypoints:
            input_point.append(point)
            input_label.append(0)

        logger.info(
            f"Predicting for scene {scene_name} with box {input_box} and input points {input_point} and labels {input_label}"
        )

        masks, _, _ = predictor.predict(
            point_coords=np.array(input_point) if len(input_point) > 0 else None,
            point_labels=np.array(input_label) if len(input_label) > 0 else None,
            box=input_box[None, :],
            multimask_output=False,
        )

        mask = masks[0]

        depth_map_path = f"{scenes_directory}/{scene_name}/observation_result/depth_map.tiff"
        depth_map = cv2.imread(depth_map_path, cv2.IMREAD_UNCHANGED)

        depth_map = np.nan_to_num(depth_map, nan=0, posinf=0, neginf=0)

        masked_depth_map = np.where(mask > 0, depth_map, 0)
        masked_values = masked_depth_map[mask > 0]

        mean = np.mean(masked_values)

        lower_bound = mean - outlier_threshold
        upper_bound = mean + outlier_threshold

        masked_depth_map = np.where(
            (masked_depth_map > lower_bound) & (masked_depth_map < upper_bound), masked_depth_map, 0
        )

        x, y = np.meshgrid(np.arange(masked_depth_map.shape[1]), np.arange(masked_depth_map.shape[0]))

        intrinsics = json.load(open(f"{scenes_directory}/{scene_name}/observation_result/camera_intrinsics.json"))

        cx = intrinsics["principal_point_in_pixels"]["cx"]
        cy = intrinsics["principal_point_in_pixels"]["cy"]

        fx = intrinsics["focal_lengths_in_pixels"]["fx"]
        fy = intrinsics["focal_lengths_in_pixels"]["fy"]

        X1 = (x - 0.5 - cx) * masked_depth_map / fx
        Y1 = (y - 0.5 - cy) * masked_depth_map / fy
        X2 = (x + 0.5 - cx) * masked_depth_map / fx
        Y2 = (y + 0.5 - cy) * masked_depth_map / fy

        pixel_areas = np.abs((X2 - X1) * (Y2 - Y1))

        mask_image_path = f"{scenes_directory}/{scene_name}/observation_result/mask.png"

        cv2.imwrite(mask_image_path, ((pixel_areas > 0) * 255).astype(np.uint8))

        coverage = np.nan_to_num(np.sum(pixel_areas), nan=0, posinf=0, neginf=0)

        json.dump(
            {
                "bbox": {
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2,
                },
                "positiveKeypoints": positive_keypoints,
                "negativeKeypoints": negative_keypoints,
                "outlierThreshold": outlier_threshold,
                "coverage": coverage,
            },
            open(f"{scenes_directory}/{scene_name}/observation_result/result.json", "w"),
        )

        print("Data processed")
        if manual:
            ack.put(scene_name)


def get_last_modified_time(directory):
    files = glob.glob(directory + "/**", recursive=True)
    if not files:  # No files in the directory
        return None
    latest_file = max(files, key=os.path.getmtime)
    timestamp = os.path.getmtime(latest_file)
    dt_object = datetime.datetime.fromtimestamp(timestamp)
    return dt_object.isoformat()


def create_app(scenes_directory, q, ack, queued_scenes):  # noqa C901
    @app.route("/")
    def index():
        root_dir = "./static"  # Directory you want to explore
        if os.path.isdir(root_dir):
            files = sorted(os.listdir(root_dir))
            return render_template("explorer.html", root=root_dir, files=files)

    @app.route("/explore")
    def explore():
        root_dir = request.args.get("dir", ".")
        if os.path.isdir(root_dir):
            files = sorted(os.listdir(root_dir))
            return render_template("explorer.html", root=root_dir, files=files)
        else:
            return "Invalid directory"

    @app.route("/latest_observation_start_dir")
    def latest_observation_start_dir():
        # TODO make the team a command line argument, or search over all teams
        sample_dir = find_latest_sample_dir_with_observation_start("static/data/dry_run_2024-05-13")
        observation_start_dir = Path(sample_dir) / "observation_start"
        return str(observation_start_dir)

    @app.route("/dev_latest_observation_start_dir")
    def dev_latest_observation_start_dir():
        # TODO make the team a command line argument, or search over all teams
        sample_dir = find_latest_sample_dir_with_observation_start("static/data/dry_run_2024-05-13/dev_team")
        observation_start_dir = Path(sample_dir) / "observation_start"
        return str(observation_start_dir)

    ALLOWED_EXTENSIONS = {"json"}

    def allowed_file(filename):
        return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

    @app.route("/upload_grasp", methods=["POST"])
    def upload_grasp():
        # TODO clean up these checks
        if "file" not in request.files:
            return jsonify({"error": "No file provided"}), 400

        file = request.files["file"]

        if not file:
            return jsonify({"error": "No file selected"}), 400

        if file.filename == "":
            return jsonify({"error": "No file selected"}), 400

        if not file or not allowed_file(file.filename):
            return jsonify({"error": "Invalid file type"}), 400

        # Try to parse the file with pydantic
        try:
            # grasp_pose_model = Pose.model_validate_json(json.load(file))
            file_contents = file.read().decode("utf-8")  # Read file contents as text
            grasp_pose_model = Pose.parse_raw(file_contents)  # Use Pydantic's parse_raw
            grasp_pose_model.as_homogeneous_matrix()
        except Exception as e:
            return jsonify({"error": f"Invalid file: {e}"}), 400

        current_upload_dir = "./static/data/dry_run_2024-05-13"

        # sample_id = "2024-04-26_00-00-00-000000"
        sample_id = request.form.get("sample_id")
        team_name = request.form.get("team_name")

        sample_id = secure_filename(sample_id)
        team_name = secure_filename(team_name)

        grasps_dirname = f"grasps_{sample_id}"
        grasps_dir = os.path.join(current_upload_dir, team_name, grasps_dirname)
        os.makedirs(grasps_dir, exist_ok=True)
        filename = f"grasp_pose_{datetime_for_filename()}.json"
        filepath = os.path.join(grasps_dir, filename)
        file = request.files["file"]  #
        file.seek(0)
        file.save(filepath)
        return jsonify({"message": "File uploaded successfully"}), 201

    # Define a route to serve images
    @app.route("/scenes/<scene_name>/image")
    def get_image(scene_name):
        # Construct the image path using the provided directory
        image_path = f"{scenes_directory}/{scene_name}/observation_result/image_left.png"
        return send_file(image_path, mimetype="image/png")  # Adjust mimetype as per your image type

    @app.route("/scenes/<scene_name>/depth")
    def get_depth(scene_name):
        # Construct the image path using the provided directory
        image_path = f"{scenes_directory}/{scene_name}/observation_result/depth_image.jpg"
        return send_file(image_path, mimetype="image/jpeg")  # Adjust mimetype as per your image type

    # Define a route to serve images
    @app.route("/scenes/<scene_name>/mask")
    def get_mask(scene_name):
        # Construct the image path using the provided directory
        try:
            image_path = f"{scenes_directory}/{scene_name}/observation_result/mask.png"
            return send_file(image_path, mimetype="image/png")  # Adjust mimetype as per your image type
        except FileNotFoundError:
            return jsonify({"error": "Mask not found"})

    # Define a route to serve images
    @app.route("/scenes/<scene_name>/coverage")
    def get_coverage(scene_name):
        # Construct the image path using the provided directory

        coverage = json.load(open(f"{scenes_directory}/{scene_name}/observation_result/coverage.json"))

        return jsonify(coverage)

    @app.route("/api/scenes", methods=["GET"])
    def get_scenes():
        scene_names = sorted(os.listdir(scenes_directory))
        scenes_info = []

        for scene_name in scene_names:
            dataset_path = os.path.join(scenes_directory, scene_name)
            last_modified_time = get_last_modified_time(dataset_path)

            result_exists = os.path.exists(f"{scenes_directory}/{scene_name}/observation_result/result.json")
            result = None

            if result_exists:
                result = json.load(open(f"{scenes_directory}/{scene_name}/observation_result/result.json"))
            elif scene_name not in queued_scenes:
                intrinsics = json.load(
                    open(f"{scenes_directory}/{scene_name}/observation_result/camera_intrinsics.json")
                )
                extrinsics = json.load(
                    open(f"{scenes_directory}/{scene_name}/observation_result/camera_pose_in_world.json")
                )
                tcp_left = json.load(
                    open(f"{scenes_directory}/{scene_name}/observation_result/arm_left_tcp_pose_in_world.json")
                )
                tcp_right = json.load(
                    open(f"{scenes_directory}/{scene_name}/observation_result/arm_right_tcp_pose_in_world.json")
                )

                cx = intrinsics["principal_point_in_pixels"]["cx"]
                cy = intrinsics["principal_point_in_pixels"]["cy"]

                fx = intrinsics["focal_lengths_in_pixels"]["fx"]
                fy = intrinsics["focal_lengths_in_pixels"]["fy"]

                position = extrinsics["position_in_meters"]
                rotation = extrinsics["rotation_euler_xyz_in_radians"]

                # Convert rotation from Euler angles to rotation matrix
                R = euler_to_mat(rotation["roll"], rotation["pitch"], rotation["yaw"])

                # Convert position to numpy array
                T = np.array([position["x"], position["y"], position["z"]])

                tcp_left_position = tcp_left["position_in_meters"]
                tcp_right_position = tcp_right["position_in_meters"]

                x_left = tcp_left_position["x"]
                y_left = tcp_left_position["y"]
                z_left = tcp_left_position["z"]

                x_right = tcp_right_position["x"]
                y_right = tcp_right_position["y"]
                z_right = tcp_right_position["z"]

                # Create the 3D rectangle for the bounding box
                y_padding = 0.1
                c1 = np.array([x_left, y_left + y_padding, z_left])
                c2 = np.array([x_right, y_right - y_padding, z_right])
                c3 = np.array([x_left, y_left + y_padding, 0.05])
                c4 = np.array([x_right, y_right - y_padding, 0.05])

                # Generate all corners
                corners = [c1, c2, c3, c4]

                projected_corners = []
                for corner in corners:
                    # Convert world coordinates to camera coordinates
                    X_cam = R.T @ (np.array(corner) - T)

                    # Project to the image plane
                    u = fx * (X_cam[0] / X_cam[2]) + cx
                    v = fy * (X_cam[1] / X_cam[2]) + cy

                    print("Projected corner", u, v, "for corner", corner)

                    projected_corners.append((u, v))

                # Get the 2D bounding box
                u_min = min(u for u, _ in projected_corners)
                v_min = min(v for _, v in projected_corners)
                u_max = max(u for u, _ in projected_corners)
                v_max = max(v for _, v in projected_corners)

                print("Queueing unannotated scene", scene_name, "with bounding box", u_min, v_min, u_max, v_max)

                q.put(
                    {
                        "sceneName": scene_name,
                        "bbox": {"x1": u_min, "y1": v_min, "x2": u_max, "y2": v_max},
                        "positiveKeypoints": [],
                        "negativeKeypoints": [],
                        "outlierThreshold": 0.5,
                        "manual": False,
                    }
                )
                queued_scenes.add(scene_name)

            scenes_info.append(
                {
                    "sceneName": scene_name,
                    "lastModifiedTime": last_modified_time,
                    "result": result,
                }
            )

        return jsonify({"scenes": scenes_info})

    @app.route("/api/annotate", methods=["POST"])
    def annotate():
        data = request.get_json()
        data["manual"] = True
        q.put(data)

        logger.info(f"Manually queueing scene {data['sceneName']} for annotation")

        while True:
            logger.info(f"Waiting for scene {data['sceneName']} to be processed")
            processed_scene = ack.get()
            logger.info(f"Processed scene {processed_scene} asked for {data['sceneName']}")

            if processed_scene == data["sceneName"]:
                logger.info(f"Scene {data['sceneName']} processed")
                return jsonify({})

    return app


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Flask server to serve the competition scenes")
    parser.add_argument("scenes_directory", type=str, help="Path to the directory containing the scenes")
    parser.add_argument("--model", default="b", type=str, choices=["h", "l", "b"], help="Model size (h, l, or b)")
    parser.add_argument("--cuda", action="store_true", help="Use CUDA if available")

    args = parser.parse_args()

    q = Queue()
    ack = Queue()

    p = Process(
        target=sam_worker,
        args=(
            q,
            ack,
            args,
        ),
    )
    p.start()

    queued_scenes = set()

    app = create_app(args.scenes_directory, q, ack, queued_scenes)
    app.run(debug=True, host="10.42.0.1", use_reloader=False)

    p.join()
