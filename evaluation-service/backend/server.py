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

app = Flask(__name__)
CORS(app)  #


def create_app(datasets_directory, predictor):
    # Define a route to serve images
    @app.route("/datasets/<scene_name>")
    def get_image(scene_name):
        # Construct the image path using the provided directory
        image_path = f"{datasets_directory}/{scene_name}/observation_result/image_left.png"
        return send_file(image_path, mimetype="image/png")  # Adjust mimetype as per your image type

    @app.route("/api/datasets", methods=["GET"])
    def get_images():
        scene_names = os.listdir(datasets_directory)
        datasets_info = []

        for scene_name in scene_names:
            dataset_path = os.path.join(datasets_directory, scene_name)
            last_modified_time = time.ctime(os.path.getmtime(dataset_path))
            mask_exists = os.path.exists(f"{datasets_directory}/{scene_name}/observation_result/mask.png")

            datasets_info.append({
                'sceneName': scene_name,
                'lastModifiedTime': last_modified_time,
                'maskExists': mask_exists
            })

        return jsonify({"datasets": datasets_info})
    
    @app.route("/api/coordinates/<scene_name>", methods=["GET"])
    def get_mask_coordinates(scene_name):

        mask = cv2.imread(f"{datasets_directory}/{scene_name}/observation_result/mask.png")

        coords = np.where(mask > 0)
        combined_coords = np.vstack((coords[1], coords[0])).T.reshape(-1)

        return jsonify({"coordinates": combined_coords.tolist()})

    @app.route("/api/annotate", methods=["POST"])
    def annotate():
        data = request.get_json()
        x = data["x"]
        y = data["y"]
        scene_name = data["sceneName"]

        image = cv2.imread(f"{datasets_directory}/{scene_name}/observation_result/image_left.png")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        predictor.set_image(image)
        input_point = np.array([[x, y]])
        input_label = np.array([1])

        masks, scores, logits = predictor.predict(
            point_coords=input_point,
            point_labels=input_label,
            multimask_output=False,
        )


        coords = np.where(masks[0])
        combined_coords = np.vstack((coords[1], coords[0])).T.reshape(-1)

        cv2.imwrite(f"{datasets_directory}/{scene_name}/observation_result/mask.png", (masks[0] * 255).astype(np.uint8))

        # Return the new coordinates as a response
        return jsonify({"coordinates": combined_coords.tolist()})

    return app


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Flask server to serve the competition datasets")
    parser.add_argument("datasets_directory", type=str, help="Path to the directory containing the datasets")
    args = parser.parse_args()

    # TODO: use best model
    sam_checkpoint = "../weights/sam_vit_b_01ec64.pth"
    model_type = "vit_b"

    device = "cpu"

    if torch.cuda.is_available():
        device = "cuda"

    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)

    predictor = SamPredictor(sam)

    app = create_app(args.datasets_directory, predictor)
    app.run(debug=True)
