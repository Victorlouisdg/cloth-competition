import argparse
from flask import Flask, send_file, request, jsonify
from flask_cors import CORS
import sys
sys.path.append("..")
from segment_anything import sam_model_registry, SamPredictor
import cv2
import numpy as np

app = Flask(__name__)
CORS(app)  #

def generate_coordinates(x, y):
    # Just a placeholder logic, you should replace this with your actual logic
    return [x, y, x + 50, y + 50, x + 100, y, x + 50, y - 50]



def create_app(image_directory, predictor):
    # Define a route to serve images
    @app.route('/images/<image_name>')
    def get_image(image_name):
        # Construct the image path using the provided directory
        image_path = f'{image_directory}/{image_name}'
        return send_file(image_path, mimetype='image/jpeg')  # Adjust mimetype as per your image type
    
    @app.route('/api/coordinates', methods=['POST'])
    def get_coordinates():
        print("get_coordinates called")
        data = request.get_json()
        x = data['x']
        y = data['y']
        image_name = data['imageName']

        print(x, y)

        image = cv2.imread(f'images/{image_name}')
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        print(image.shape)

        print("image", x, y)

        predictor.set_image(image)

        print("set_image called", x, y)
        input_point = np.array([[x, y]])
        input_label = np.array([1])

        print(input_point)

        masks, scores, logits = predictor.predict(
            point_coords=input_point,
            point_labels=input_label,
            multimask_output=True,
        )

        print(masks.shape, scores.shape, logits.shape)

        # coords = np.where(np.any(masks, axis=0))
        coords = np.where(masks[0])
        combined_coords = np.vstack((coords[1], coords[0])).T.reshape(-1)

        print(combined_coords)

        # Return the new coordinates as a response
        return jsonify({'coordinates': combined_coords.tolist()})

    return app

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run Flask server to serve images')
    parser.add_argument('image_directory', type=str, help='Path to the directory containing images')
    args = parser.parse_args()

    sam_checkpoint = "sam_vit_h_4b8939.pth"
    model_type = "vit_h"

    device = "cpu"

    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)

    predictor = SamPredictor(sam)

    app = create_app(args.image_directory, predictor)
    app.run(debug=True)