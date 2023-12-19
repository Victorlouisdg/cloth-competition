"""Collect samples of the input RGB-D data without moving the robots.
This also means that no grasp labels are collected."""
import cv2
from airo_camera_toolkit.cameras.zed2i import Zed2i
from airo_camera_toolkit.utils import ImageConverter

if __name__ == "__main__":
    print("test")

    camera = Zed2i()

    window_name = "RGBD sample collection"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    while True:
        image_rgb = camera.get_rgb_image_as_int()
        depth_map = camera._retrieve_depth_map()

        image_bgr = ImageConverter.from_numpy_int_image(image_rgb).image_in_opencv_format

        cv2.imshow(window_name, image_bgr)
        key = cv2.waitKey(1)
        if key == ord("q"):
            break
        elif key == ord("s"):
            cv2.imwrite("image.png", image_bgr)
            # cv2.imwrite("depth.png", depth_map)
            print("Saved image.png to current directory.")
            break
