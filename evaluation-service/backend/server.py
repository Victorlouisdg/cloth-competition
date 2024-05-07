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
sys.path.append(".")

import datetime
import glob
import json
import os

import numpy as np

app = Flask(__name__)
CORS(app)  #

from multiprocessing import Queue

from airo_camera_toolkit.pinhole_operations.projection import project_points_to_image_plane
from airo_dataset_tools.data_parsers.camera_intrinsics import CameraIntrinsics
from airo_spatial_algebra import transform_points


def get_heuristic_cloth_bounding_box(sample_dir: str) -> tuple[float, float, float, float]:
    """Calculates an approximate 2D bounding box for the cloth region held by the robot arms.

    This function assume the case where the cloth is held both robots arms and stretched in front of the camera.

    Args:
        sample_dir: The path to the sample directory containing the necessary data files.

    Returns:
        A tuple of (u_min, v_min, u_max, v_max) representing the coordinates of the
        estimated bounding box within the image.
    """
    # Load data
    tcp_left = json.load(open(f"{sample_dir}/observation_result/arm_left_tcp_pose_in_world.json"))
    tcp_right = json.load(open(f"{sample_dir}/observation_result/arm_right_tcp_pose_in_world.json"))

    intrinsics_path = f"{sample_dir}/observation_result/camera_intrinsics.json"
    with open(intrinsics_path, "r") as f:
        intrinsics_model = CameraIntrinsics.model_validate_json(f.read())
        intrinsics = intrinsics_model.as_matrix()

    extrinsics_path = f"{sample_dir}/observation_result/camera_pose_in_world.json"
    with open(extrinsics_path) as f:
        extrinsics = Pose.model_validate_json(f.read()).as_homogeneous_matrix()

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
    corners_3d = np.array([c1, c2, c3, c4])

    X_C_W = np.linalg.inv(extrinsics)
    projected_corners = project_points_to_image_plane(transform_points(X_C_W, corners_3d), intrinsics).squeeze()

    # Get the 2D bounding box
    u_min = min(u for u, _ in projected_corners)
    v_min = min(v for _, v in projected_corners)
    u_max = max(u for u, _ in projected_corners)
    v_max = max(v for _, v in projected_corners)

    return u_min, v_min, u_max, v_max


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

        static_dir = os.path.abspath("./static")
        # check if root_dir is within static_dir
        if not os.path.normpath(os.path.abspath(root_dir)).startswith(static_dir):
            return "Invalid directory"

        if os.path.isdir(root_dir):
            files = sorted(os.listdir(root_dir))
            return render_template("explorer.html", root=root_dir, files=files)
        else:
            return "Invalid directory"

    @app.route("/latest_observation_start_dir")
    def latest_observation_start_dir():
        # TODO make the team a command line argument, or search over all teams
        sample_dir = find_latest_sample_dir_with_observation_start("static/data/remote_dry_run_2024-04-26")
        observation_start_dir = Path(sample_dir) / "observation_start"
        return str(observation_start_dir)

    @app.route("/dev_latest_observation_start_dir")
    def dev_latest_observation_start_dir():
        # TODO make the team a command line argument, or search over all teams
        sample_dir = find_latest_sample_dir_with_observation_start("static/data/remote_dry_run_2024-04-26/dev_team")
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

        current_upload_dir = "./static/data/remote_dry_run_2024-04-26"

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

    @app.route("/api/teams", methods=["GET"])
    def get_teams():
        dataset_dir = scenes_directory
        teams = sorted(os.listdir(dataset_dir))
        return jsonify({"teams": teams})

    @app.route("/api/scenes", methods=["GET"])
    def get_scenes():
        scene_names = sorted(os.listdir(scenes_directory))
        scenes_info = []

        for scene_name in scene_names:
            sample_dir = os.path.join(scenes_directory, scene_name)
            last_modified_time = get_last_modified_time(sample_dir)

            result_path = f"{scenes_directory}/{scene_name}/observation_result/result.json"
            result = json.load(open(result_path)) if os.path.exists(result_path) else None

            if result is None:
                # Calculate initial bounding box
                u_min, v_min, u_max, v_max = get_heuristic_cloth_bounding_box(sample_dir)

                # Queue the scene if it has not been segmented yet
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

    # p = Process(
    #     target=sam_worker,
    #     args=(
    #         q,
    #         ack,
    #         args,
    #     ),
    # )
    # p.start()

    queued_scenes = set()

    app = create_app(args.scenes_directory, q, ack, queued_scenes)
    app.run(debug=True, host="0.0.0.0", use_reloader=False)

    # p.join()
