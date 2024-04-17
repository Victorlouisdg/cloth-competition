import cv2
import numpy as np


def calculate_pixel_areas_for_image(depth_image_path, mask_image_path, fx, fy, cx, cy):

    depth_map = cv2.imread(depth_image_path, cv2.IMREAD_UNCHANGED)
    mask = cv2.imread(mask_image_path, cv2.IMREAD_UNCHANGED)

    masked_depth_map = np.where(mask > 0, depth_map, 0)
    masked_values = masked_depth_map[mask > 0]

    mean = np.mean(masked_values)
    median = np.median(masked_values)
    min = np.min(masked_values)
    max = np.max(masked_values)
    std = np.std(masked_values)

    print("mean", mean, "median", median, "min", min, "max", max, "std", std)

    lower_bound = mean - 0.5
    upper_bound = mean + 0.5

    masked_depth_map = np.where(
        (masked_depth_map > lower_bound) & (masked_depth_map < upper_bound), masked_depth_map, 0
    )

    x, y = np.meshgrid(np.arange(masked_depth_map.shape[1]), np.arange(masked_depth_map.shape[0]))

    X1 = (x - 0.5 - cx) * masked_depth_map / fx
    Y1 = (y - 0.5 - cy) * masked_depth_map / fy
    X2 = (x + 0.5 - cx) * masked_depth_map / fx
    Y2 = (y + 0.5 - cy) * masked_depth_map / fy

    pixel_areas = np.abs((X2 - X1) * (Y2 - Y1))
    pixel_areas_masked = np.where(mask > 0, pixel_areas, 0)

    print("masked areas", np.sum(pixel_areas_masked))

    # Normalize the pixel areas to the range 0-255
    pixel_areas_normalized = cv2.normalize(
        pixel_areas_masked, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U
    )

    # # # Display the normalized pixel areas
    cv2.imshow("Pixel Areas", pixel_areas_normalized)

    # Wait for a key press and close the window
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":

    # "image_resolution": {
    #     "width": 2208,
    #     "height": 1242
    # },
    # "focal_lengths_in_pixels": {
    #     "fx": 1048.748291015625,
    #     "fy": 1048.748291015625
    # },
    # "principal_point_in_pixels": {
    #     "cx": 1104.63525390625,
    #     "cy": 621.6848754882812
    # }

    fx = 1048.748291015625
    fy = 1048.748291015625
    cx = 1104.63525390625
    cy = 621.6848754882812

    calculate_pixel_areas_for_image(
        "/home/jh/dev/cloth-competition/datasets/sample_000001/observation_result/depth_map.tiff",
        "/home/jh/dev/cloth-competition/datasets/sample_000001/observation_result/mask.png",
        fx,
        fy,
        cx,
        cy,
    )
