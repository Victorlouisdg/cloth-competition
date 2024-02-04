import cv2
import numpy as np
from airo_camera_toolkit.cameras.zed.zed2i import Zed2i
from airo_camera_toolkit.utils.image_converter import ImageConverter
from cloth_tools.config import load_camera_pose_in_left_and_right
from cloth_tools.visualization.opencv import draw_pose
from pydrake.math import RigidTransform, RollPitchYaw

# LCB stands for "What the left control box considers to be the coordinate frame"
y_distance = 0.45
X_W_L = RigidTransform(rpy=RollPitchYaw([0, 0, np.pi / 2]), p=[0, y_distance, 0]).GetAsMatrix4()
X_W_R = RigidTransform(rpy=RollPitchYaw([0, 0, np.pi / 2]), p=[0, -y_distance, 0]).GetAsMatrix4()
X_CB_B = RigidTransform(rpy=RollPitchYaw([0, 0, np.pi]), p=[0, 0, 0]).GetAsMatrix4()
X_LCB_W = X_CB_B @ np.linalg.inv(X_W_L)
X_RCB_W = X_CB_B @ np.linalg.inv(X_W_R)
X_LCB_C, X_RCB_C = load_camera_pose_in_left_and_right()
X_C_W = np.linalg.inv(X_LCB_C) @ X_LCB_W
X_C_LCB = np.linalg.inv(X_LCB_C)
X_C_RCB = np.linalg.inv(X_RCB_C)

camera = Zed2i(resolution=Zed2i.RESOLUTION_2K, depth_mode=Zed2i.NEURAL_DEPTH_MODE, fps=15)

window_name = "Check world frame"
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)


while True:
    image_rgb = camera.get_rgb_image_as_int()
    image_bgr = ImageConverter.from_numpy_int_format(image_rgb).image_in_opencv_format

    # visualize X_C_W, X_C_LCB. X_C_RCB
    draw_pose(image_bgr, X_C_W, camera.intrinsics_matrix(), np.identity(4), 0.25)
    draw_pose(image_bgr, X_C_LCB, camera.intrinsics_matrix(), np.identity(4))
    draw_pose(image_bgr, X_C_RCB, camera.intrinsics_matrix(), np.identity(4))

    cv2.imshow(window_name, image_bgr)
    key = cv2.waitKey(1)
    if key == ord("q"):
        break
