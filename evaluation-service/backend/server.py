import argparse
import sys

from flask import Flask, jsonify, request, send_file
from flask_cors import CORS

sys.path.append("..")
import os

import cv2
import numpy as np
import torch
from segment_anything import SamPredictor, sam_model_registry

app = Flask(__name__)
CORS(app)  #


def create_app(datasets_directory, predictor):
    # Define a route to serve images
    @app.route("/datasets/<dataset_name>")
    def get_image(dataset_name):
        # Construct the image path using the provided directory
        image_path = f"{datasets_directory}/{dataset_name}/observation_result/image_left.png"
        return send_file(image_path, mimetype="image/png")  # Adjust mimetype as per your image type

    @app.route("/api/datasets", methods=["GET"])
    def get_images():
        dataset_names = os.listdir(datasets_directory)

        return jsonify({"datasets": dataset_names})

    @app.route("/api/coordinates", methods=["POST"])
    def get_coordinates():
        print("get_coordinates called")
        data = request.get_json()
        x = data["x"]
        y = data["y"]
        dataset_name = data["datasetName"]

        image = cv2.imread(f"{datasets_directory}/{dataset_name}/observation_result/image_left.png")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        predictor.set_image(image)
        input_point = np.array([[x, y]])
        input_label = np.array([1])

        masks, scores, logits = predictor.predict(
            point_coords=input_point,
            point_labels=input_label,
            multimask_output=True,
        )

        print(masks.shape, scores.shape, logits.shape)

        # TODO: allow using any of the masks
        coords = np.where(masks[2])
        combined_coords = np.vstack((coords[1], coords[0])).T.reshape(-1)

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
