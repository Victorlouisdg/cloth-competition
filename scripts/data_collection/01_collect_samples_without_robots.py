"""Collect samples of the input RGB-D data without moving the robots.
This also means that no grasp labels are collected."""
import cv2
from airo_camera_toolkit.cameras.zed2i import Zed2i
from airo_camera_toolkit.utils import ImageConverter
from cloth_tools.config import load_camera_pose_in_left_and_right

if __name__ == "__main__":
    camera = Zed2i()
    camera_pose, _ = load_camera_pose_in_left_and_right()

    window_name = "RGBD sample collection"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    while True:
        image_rgb = camera.get_rgb_image_as_int()
        depth_map = camera._retrieve_depth_map()

        image_bgr = ImageConverter.from_numpy_int_format(image_rgb).image_in_opencv_format

        cv2.imshow(window_name, image_bgr)
        key = cv2.waitKey(1)
        if key == ord("q"):
            break
        elif key == ord("s"):

            # sample = CompetitionInputSample()

            cv2.imwrite("image.png", image_bgr)
            # cv2.imwrite("depth.png", depth_map)
            print("Saved image.png to current directory.")
            break
