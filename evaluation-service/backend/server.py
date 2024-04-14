import argparse
import sys

from flask import Flask, jsonify, request, send_file
from flask_cors import CORS

sys.path.append("..")
import os
import time
import cv2
import numpy as np
import torch
from segment_anything import SamPredictor, sam_model_registry
import json
import glob
import datetime
from scipy.spatial.transform import Rotation as R
import itertools

app = Flask(__name__)
CORS(app)  #

from multiprocessing import Process, Queue

import numpy as np

def euler_to_mat(roll, pitch, yaw):
    # Create rotation matrix from Euler angles
    R_x = np.array([[1, 0, 0],
                    [0, np.cos(roll), -np.sin(roll)],
                    [0, np.sin(roll), np.cos(roll)]])

    R_y = np.array([[np.cos(pitch), 0, np.sin(pitch)],
                    [0, 1, 0],
                    [-np.sin(pitch), 0, np.cos(pitch)]])

    R_z = np.array([[np.cos(yaw), -np.sin(yaw), 0],
                    [np.sin(yaw), np.cos(yaw), 0],
                    [0, 0, 1]])

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
        device = "cuda"

    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)

    predictor = SamPredictor(sam)

    while True:
        print("Waiting for data")
        data = q.get()
        print("Data received", data)

        x1 = data["x1"]
        y1 = data["y1"]
        x2 = data["x2"]
        y2 = data["y2"]
        
        if "outlierThreshold" in data and data["outlierThreshold"] is not None:
            outlier_threshold = float(data["outlierThreshold"])
        else:
            outlier_threshold = 0.5

        scene_name = data["sceneName"]

        image = cv2.imread(f"{scenes_directory}/{scene_name}/observation_result/image_left.png")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        predictor.set_image(image)

        input_box = np.array([[x1, y1, x2, y2]])


        masks, _, _ = predictor.predict(
            point_coords=None,
            point_labels=None,
            box=input_box[None, :],
            multimask_output=False,
        )

        mask = masks[0]

        depth_map_path = f"{scenes_directory}/{scene_name}/observation_result/depth_map.tiff"
        depth_map = cv2.imread(depth_map_path, cv2.IMREAD_UNCHANGED)

        masked_depth_map = np.where(mask > 0, depth_map, 0)
        masked_values = masked_depth_map[mask > 0]

        mean = np.mean(masked_values)

        lower_bound = mean - outlier_threshold
        upper_bound = mean + outlier_threshold

        masked_depth_map = np.where((masked_depth_map > lower_bound) & (masked_depth_map < upper_bound), masked_depth_map, 0)

        x, y = np.meshgrid(np.arange(masked_depth_map.shape[1]), np.arange(masked_depth_map.shape[0]))

        intrinsics = json.load(open(f"{scenes_directory}/{scene_name}/observation_result/camera_intrinsics.json"))

        cx = intrinsics["principal_point_in_pixels"]["cx"]
        cy = intrinsics["principal_point_in_pixels"]["cy"]

        fx = intrinsics["focal_lengths_in_pixels"]["fx"]
        fy = intrinsics["focal_lengths_in_pixels"]["fy"]

        X1 = (x - 0.5 - cx) * masked_depth_map/ fx
        Y1 = (y - 0.5 - cy) * masked_depth_map / fy
        X2 = (x + 0.5 - cx) * masked_depth_map / fx
        Y2 = (y + 0.5 - cy) * masked_depth_map / fy

        pixel_areas = np.abs((X2 - X1) * (Y2 - Y1))

        mask_image_path = f"{scenes_directory}/{scene_name}/observation_result/mask.png"


        cv2.imwrite(mask_image_path, ((pixel_areas > 0) * 255).astype(np.uint8))

        json.dump({
            "x1": x1,
            "y1": y1,
            "x2": x2,
            "y2": y2,
            "outlierThreshold": outlier_threshold,
            "coverage": np.sum(pixel_areas),
            },
            open(f"{scenes_directory}/{scene_name}/observation_result/result.json", "w"))
        
        print("Data processed")
        ack.put(scene_name)
        

def get_last_modified_time(directory):
    files = glob.glob(directory + "/**", recursive=True)
    if not files:  # No files in the directory
        return None
    latest_file = max(files, key=os.path.getmtime)
    timestamp = os.path.getmtime(latest_file)
    dt_object = datetime.datetime.fromtimestamp(timestamp)
    return dt_object.isoformat()


def create_app(scenes_directory, q, ack, queued_scenes):
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
        scene_names = os.listdir(scenes_directory)
        scenes_info = []

        for scene_name in scene_names:
            dataset_path = os.path.join(scenes_directory, scene_name)
            last_modified_time = get_last_modified_time(dataset_path)

            result_exists = os.path.exists(f"{scenes_directory}/{scene_name}/observation_result/result.json")
            result = None

            if result_exists:
                result = json.load(open(f"{scenes_directory}/{scene_name}/observation_result/result.json"))
            elif scene_name not in queued_scenes:
                intrinsics = json.load(open(f"{scenes_directory}/{scene_name}/observation_result/camera_intrinsics.json"))
                extrinsics = json.load(open(f"{scenes_directory}/{scene_name}/observation_result/camera_pose_in_world.json"))
                tcp_left = json.load(open(f"{scenes_directory}/{scene_name}/observation_result/arm_left_tcp_pose_in_world.json"))
                tcp_right = json.load(open(f"{scenes_directory}/{scene_name}/observation_result/arm_right_tcp_pose_in_world.json"))

                cx = intrinsics["principal_point_in_pixels"]["cx"]
                cy = intrinsics["principal_point_in_pixels"]["cy"]

                fx = intrinsics["focal_lengths_in_pixels"]["fx"]
                fy = intrinsics["focal_lengths_in_pixels"]["fy"]

                position = extrinsics["position_in_meters"]
                position = extrinsics["position_in_meters"]
                rotation = extrinsics["rotation_euler_xyz_in_radians"]

                # Convert rotation from Euler angles to rotation matrix
                R = euler_to_mat(rotation["roll"], rotation["pitch"], rotation["yaw"])

                # Convert position to numpy array
                T = np.array([position["x"], position["y"], position["z"]])

                tcp_left_position = tcp_left["position_in_meters"]
                tcp_right_position = tcp_right["position_in_meters"]
                
                #TODO: bit hacky, the tcp positions seem funky
                c1 = np.array([-tcp_left_position["x"], tcp_left_position["y"] + 0.1, tcp_left_position["z"]])
                c2 = np.array([tcp_right_position["x"], -tcp_right_position["y"] - 0.1, tcp_right_position["z"]])
                c3 = np.array([-tcp_left_position["x"], tcp_left_position["y"] + 0.1, 0.15])
                c4 = np.array([tcp_right_position["x"], -tcp_right_position["y"] - 0.1, 0.15])

                # Generate all corners
                corners = [c1, c2, c3, c4]

                projected_corners = []
                for corner in corners:
                    # Convert world coordinates to camera coordinates
                    X_cam = R.T @ (np.array(corner) - T)

                    # Project to the image plane
                    u = fx * (X_cam[0]/X_cam[2]) + cx
                    v = fy * (X_cam[1]/X_cam[2]) + cy

                    print("Projected corner", u, v, "for corner", corner)

                    projected_corners.append((u, v))

                # Get the 2D bounding box
                u_min = min(u for u, v in projected_corners)
                v_min = min(v for u, v in projected_corners)
                u_max = max(u for u, v in projected_corners)
                v_max = max(v for u, v in projected_corners)

                print("Queueing unannotated scene", scene_name, "with bounding box", u_min, v_min, u_max, v_max)

                q.put({
                    "sceneName": scene_name,
                    "x1": u_min,
                    "y1": v_min,
                    "x2": u_max,
                    "y2": v_max,
                    "outlierThreshold": 0.5,
                })
                queued_scenes.add(scene_name)

            scenes_info.append({
                'sceneName': scene_name,
                'lastModifiedTime': last_modified_time,
                'result': result,
            })

        return jsonify({"scenes": scenes_info})
    
    @app.route("/api/annotate", methods=["POST"])
    def annotate():
        data = request.get_json()
        q.put(data)

        while True:
            processed_scene = ack.get()
            print("Processed scene", processed_scene, "asked for", data["sceneName"])

            if processed_scene == data["sceneName"]:
                return jsonify({})

        

    return app


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Flask server to serve the competition scenes")
    parser.add_argument("scenes_directory", type=str, help="Path to the directory containing the scenes")
    parser.add_argument("--model", default="b", type=str, choices=['h', 'l', 'b'], help="Model size (h, l, or b)")
    parser.add_argument('--cuda', action='store_true', help='Use CUDA if available')

    args = parser.parse_args()

    q = Queue()
    ack = Queue()

    p = Process(target=sam_worker, args=(q,ack,args,))
    p.start()

    
    queued_scenes = set()

    app = create_app(args.scenes_directory, q, ack, queued_scenes)
    app.run(debug=True)

    p.join()
