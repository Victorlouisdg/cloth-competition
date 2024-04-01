import numpy as np
import cv2


def calculate_pixel_areas_for_image(image_path, fx, fy, cx, cy):
    # Load the depth map from the image
    depth_map = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Get the pixel coordinates
    x, y = np.meshgrid(np.arange(depth_map.shape[1]), np.arange(depth_map.shape[0]))

    # Calculate the world coordinates of the four corners of each pixel
    X1 = (x - 0.5 - cx) * depth_map / fx
    Y1 = (y - 0.5 - cy) * depth_map / fy
    X2 = (x + 0.5 - cx) * depth_map / fx
    Y2 = (y + 0.5 - cy) * depth_map / fy

    # Calculate the area of each pixel
    pixel_areas = np.abs((X2 - X1) * (Y2 - Y1))

    # Normalize the pixel areas to the range 0-255
    pixel_areas_normalized = cv2.normalize(pixel_areas, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    # Display the normalized pixel areas
    cv2.imshow('Pixel Areas', pixel_areas_normalized)

    # Wait for a key press and close the window
    cv2.waitKey(0)
    cv2.destroyAllWindows()




if __name__ == "__main__":
    image_path = "/home/jh/dev/cloth-competition/datasets/sample_000001/observation_result/depth_image.jpg"


    
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

    calculate_pixel_areas_for_image(image_path, fx, fy, cx, cy)