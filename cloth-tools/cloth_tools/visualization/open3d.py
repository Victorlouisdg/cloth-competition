from typing import Optional, Tuple

import numpy as np
import open3d as o3d
from airo_typing import CameraIntrinsicsMatrixType, HomogeneousMatrixType


def open3d_camera(
    camera_pose: HomogeneousMatrixType,
    intrinsics: CameraIntrinsicsMatrixType,
    resolution: Tuple[int, int],
    color: Optional[Tuple[float, float, float]] is None,
    scale: float = 0.1,
) -> o3d.geometry.LineSet:
    """Creates a camera visualization in open3d.

    Args:
        camera_pose: camera pose
        intrinsics: camera intrinsics
        resolution: width, height of the camera's images

    Returns:
        camera_lines: a line set
    """

    camera_lines = o3d.geometry.LineSet.create_camera_visualization(
        *resolution, intrinsics, np.linalg.inv(camera_pose), scale=scale
    )
    if color is not None:
        camera_lines.paint_uniform_color(color)

    return camera_lines
