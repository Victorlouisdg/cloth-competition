from airo_camera_toolkit.point_clouds.operations import crop_point_cloud_mask, filter_point_cloud
from airo_typing import BoundingBox3DType, NumpyDepthMapType, PointCloud


def filter_and_crop_point_cloud(
    point_cloud: PointCloud,
    confidence_map: NumpyDepthMapType,
    bounding_box: BoundingBox3DType,
    confidence_threshold: float = 1.0,
) -> PointCloud:
    """Filters and crops the point cloud to the region of interest.
    This function should be faster than doing both operations sequentially.

    Args:
        point_cloud: The point cloud to filter and crop.
        confidence_map: The confidence map (where high values are more uncertain).
        bounding_box: The bounding box.
        confidence_threshold: The confidence threshold.

    Returns:
        The filtered and cropped point cloud.
    """
    # Filter the low confidence points
    confidence_mask = (confidence_map <= confidence_threshold).reshape(-1)  # Threshold and flatten
    crop_mask = crop_point_cloud_mask(point_cloud, bounding_box)

    # Combine the masks
    mask = confidence_mask & crop_mask

    point_cloud_filtered_and_cropped = filter_point_cloud(point_cloud, mask.nonzero())
    return point_cloud_filtered_and_cropped
