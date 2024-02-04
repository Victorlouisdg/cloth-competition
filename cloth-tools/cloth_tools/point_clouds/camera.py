from typing import Tuple

from airo_camera_toolkit.cameras.zed.zed2i import Zed2i
from airo_camera_toolkit.point_clouds.conversions import open3d_to_point_cloud, point_cloud_to_open3d
from airo_camera_toolkit.point_clouds.operations import filter_point_cloud
from airo_typing import HomogeneousMatrixType, NumpyIntImageType, PointCloud


def get_image_and_filtered_point_cloud(
    camera: Zed2i, camera_pose: HomogeneousMatrixType
) -> Tuple[NumpyIntImageType, PointCloud]:
    """Get an RGB image and a point cloud from the camera.
    Low confidence points are filtered out and the point cloud is transformed to the world frame.

    Args:
        camera: The camera.
        camera_pose: The pose of the camera in the world frame.

    Returns:
        image_rgb: The RGB image from the camera.
        point_cloud_filtered: The point cloud in the world frame.
    """
    image_rgb = camera.get_rgb_image_as_int()
    point_cloud_in_camera = camera._retrieve_colored_point_cloud()
    confidence_map = camera._retrieve_confidence_map()

    # Transform the point cloud to the world frame
    pcd_in_camera = point_cloud_to_open3d(point_cloud_in_camera)
    pcd = pcd_in_camera.transform(camera_pose)  # transform to world frame (= base frame of left robot)

    # Filter the low confidence points
    confidence_mask = (confidence_map <= 1.0).reshape(-1)  # Threshold and flatten

    point_cloud = open3d_to_point_cloud(pcd)
    point_cloud_filtered = filter_point_cloud(point_cloud, confidence_mask.nonzero())
    return image_rgb, point_cloud_filtered
