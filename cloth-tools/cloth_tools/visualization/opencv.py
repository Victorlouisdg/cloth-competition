from typing import Tuple

import cv2
import numpy as np
from airo_camera_toolkit.pinhole_operations.projection import project_points_to_image_plane
from airo_spatial_algebra import SE3Container, transform_points
from airo_typing import CameraIntrinsicsMatrixType, HomogeneousMatrixType, OpenCVIntImageType, Vector3DType


def draw_point_3d(
    image: OpenCVIntImageType,
    point_3d: Vector3DType,
    intrinsics: CameraIntrinsicsMatrixType,
    camera_pose: HomogeneousMatrixType,
    color: Tuple[int, int, int],
) -> None:
    """Draws a 3D point on an image. Image and color may be in either RGB or BGR.

    Args:
        image: The image to draw on.
        point_3d: The 3D point to draw.
        color: The color of the point.
    """
    X_C_W = np.linalg.inv(camera_pose)
    point_2d = project_points_to_image_plane(transform_points(X_C_W, point_3d), intrinsics).squeeze()
    point_2d_int = np.rint(point_2d).astype(int)
    cv2.circle(image, tuple(point_2d_int), 10, color, thickness=2)


def draw_pose(
    image: OpenCVIntImageType,
    pose: HomogeneousMatrixType,
    intrinsics: CameraIntrinsicsMatrixType,
    camera_pose: HomogeneousMatrixType,
    length: float = 0.1,
) -> None:
    """Draws a pose on an image. Image must be BGR to have correct axes colors.

    Args:
        image: The image to draw on.
        pose: The pose to draw.
        color: The color of the pose.
    """
    X_W_C = camera_pose
    X_W_P = pose
    X_C_P = np.linalg.inv(X_W_C) @ X_W_P
    frame_pose_in_camera = X_C_P

    # copied from airo_camera_toolkit/calibration/fiducial_markers.py
    charuco_se3 = SE3Container.from_homogeneous_matrix(frame_pose_in_camera)
    rvec = charuco_se3.orientation_as_rotation_vector
    tvec = charuco_se3.translation
    cv2.drawFrameAxes(image, intrinsics, None, rvec, tvec, length)
