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

app = Flask(__name__)
CORS(app)  #


def create_app(scenes_directory, predictor):
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
        image_path = f"{scenes_directory}/{scene_name}/observation_result/mask.png"
        return send_file(image_path, mimetype="image/png")  # Adjust mimetype as per your image type
    
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
            last_modified_time = time.ctime(os.path.getmtime(dataset_path))
            mask_exists = os.path.exists(f"{scenes_directory}/{scene_name}/observation_result/mask.png")

            scenes_info.append({
                'sceneName': scene_name,
                'lastModifiedTime': last_modified_time,
                'maskExists': mask_exists
            })

        return jsonify({"scenes": scenes_info})
    
    @app.route("/api/annotate", methods=["POST"])
    def annotate():
        data = request.get_json()
        x = data["x"]
        y = data["y"]
        scene_name = data["sceneName"]

        image = cv2.imread(f"{scenes_directory}/{scene_name}/observation_result/image_left.png")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        predictor.set_image(image)
        input_point = np.array([[x, y]])
        input_label = np.array([1])

        masks, scores, logits = predictor.predict(
            point_coords=input_point,
            point_labels=input_label,
            multimask_output=False,
        )

        mask = masks[0]
        mask_image_path = f"{scenes_directory}/{scene_name}/observation_result/mask.png"


        cv2.imwrite(mask_image_path, (mask * 255).astype(np.uint8))

        depth_map_path = f"{scenes_directory}/{scene_name}/observation_result/depth_map.tiff"
        depth_map = cv2.imread(depth_map_path, cv2.IMREAD_UNCHANGED)

        masked_depth_map = np.where(mask > 0, depth_map, 0)
        masked_values = masked_depth_map[mask > 0]

        mean = np.mean(masked_values)

        lower_bound = mean - 0.5
        upper_bound = mean + 0.5

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

        json.dump({"coverage": np.sum(pixel_areas)}, open(f"{scenes_directory}/{scene_name}/observation_result/coverage.json", "w"))

        # Return the new coordinates as a response
        return jsonify({})

    return app


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Flask server to serve the competition scenes")
    parser.add_argument("scenes_directory", type=str, help="Path to the directory containing the scenes")
    parser.add_argument("--model", default="b", type=str, choices=['h', 'l', 'b'], help="Model size (h, l, or b)")
    parser.add_argument('--cuda', action='store_true', help='Use CUDA if available')

    

    sam_checkpoint = "../weights/sam_vit_b_01ec64.pth"
    model_type = "vit_b"
    device = "cpu"

    args = parser.parse_args()

    cuda = args.cuda
    model_size = args.model

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

    app = create_app(args.scenes_directory, predictor)
    app.run(debug=True)
