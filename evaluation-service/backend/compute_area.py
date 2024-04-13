import numpy as np
import cv2


def calculate_pixel_areas_for_image(depth_image_path, mask_image_path, fx, fy, cx, cy):
    # Load the depth map from the image
    depth_map = cv2.imread(depth_image_path, cv2.IMREAD_UNCHANGED)
    # Check for NaN values in the depth map
    nan_mask = np.isnan(depth_map)

    # Print the number of NaN values
    print("Number of NaN values in depth map:", np.sum(nan_mask))
    mask = cv2.imread(mask_image_path, cv2.IMREAD_UNCHANGED)

    # masked_depth = np.where(mask > 0, depth_map, 0)

    # print(masked_depth)

    # mean_masked_depth = np.mean(masked_depth[mask > 0])

    # print(mean_masked_depth)

    # Get the pixel coordinates
    x, y = np.meshgrid(np.arange(depth_map.shape[1]), np.arange(depth_map.shape[0]))

    # Calculate the world coordinates of the four corners of each pixel
    X1 = (x - 0.5 - cx) * depth_map / fx
    Y1 = (y - 0.5 - cy) * depth_map / fy
    X2 = (x + 0.5 - cx) * depth_map / fx
    Y2 = (y + 0.5 - cy) * depth_map / fy

    print("X1", X1)
    print("X2", X2)
    print("dff1", X1-X2)
    print("dff2", Y1-Y2)
    print("mul", (X1 - X2) * (Y1 - Y2))



    # Calculate the area of each pixel
    pixel_areas = np.abs((X2 - X1) * (Y2 - Y1))

    print("areas", pixel_areas)

    pixel_areas_masked = np.where(mask > 0, pixel_areas, 0)

    print("masked areas", np.sum(pixel_areas_masked), np.mean(pixel_areas_masked))

    

    # Normalize the pixel areas to the range 0-255
    pixel_areas_normalized = cv2.normalize(pixel_areas_masked, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    # # # # Display the normalized pixel areas
    # cv2.imshow('Pixel Areas', pixel_areas_normalized)

    # # Wait for a key press and close the window
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()




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

    calculate_pixel_areas_for_image(depth_image_path, mask_image_path, fx, fy, cx, cy)