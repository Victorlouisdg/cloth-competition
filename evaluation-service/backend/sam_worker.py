import json
import sys

import cv2
import numpy as np
import torch
from loguru import logger

sys.path.append("..")
from segment_anything import SamPredictor, sam_model_registry


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
