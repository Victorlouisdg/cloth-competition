"""Collect samples of the input RGB-D data without moving the robots.
This also means that no grasp labels are collected."""
import cv2
from airo_camera_toolkit.cameras.zed2i import Zed2i
from airo_camera_toolkit.interfaces import StereoRGBDCamera
from airo_camera_toolkit.point_clouds.conversions import open3d_to_point_cloud, point_cloud_to_open3d
from airo_camera_toolkit.point_clouds.operations import crop_point_cloud_mask, filter_point_cloud
from airo_camera_toolkit.utils import ImageConverter
from airo_typing import BoundingBox3DType, NumpyDepthMapType, PointCloud
from cloth_tools.bounding_boxes import BBOX_CLOTH_IN_THE_AIR
from cloth_tools.config import load_camera_pose_in_left_and_right
from cloth_tools.dataset.format import CompetitionInputSample, save_competition_input_sample
from loguru import logger


def filter_and_crop_point_cloud(
    point_cloud: PointCloud,
    confidence_map: NumpyDepthMapType,
    bounding_box=BoundingBox3DType,
):
    """Filters and crops the point cloud to the region of interest.

    Args:
        point_cloud: The point cloud to filter and crop.

    Returns:
        The filtered and cropped point cloud.
    """
    # Filter the low confidence points
    confidence_mask = (confidence_map <= 1.0).reshape(-1)  # Threshold and flatten
    crop_mask = crop_point_cloud_mask(point_cloud, bounding_box)

    # Combine the masks
    mask = confidence_mask & crop_mask

    point_cloud_filtered_and_cropped = filter_point_cloud(point_cloud, mask.nonzero())
    return point_cloud_filtered_and_cropped


if __name__ == "__main__":
    camera = Zed2i(resolution=Zed2i.RESOLUTION_2K, depth_mode=Zed2i.NEURAL_DEPTH_MODE, fps=15)
    camera_pose, _ = load_camera_pose_in_left_and_right()

    window_name = "RGBD sample collection"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    dataset_dir = "dataset"
    sample_index = 0

    while True:
        image_left = camera.get_rgb_image_as_int()
        image_right = camera._retrieve_rgb_image_as_int(StereoRGBDCamera.RIGHT_RGB)
        depth_map = camera._retrieve_depth_map()
        depth_image = camera._retrieve_depth_image()
        confidence_map = camera._retrieve_confidence_map()

        point_cloud_in_camera = camera._retrieve_colored_point_cloud()
        pcd_in_camera = point_cloud_to_open3d(point_cloud_in_camera)
        pcd = pcd_in_camera.transform(camera_pose)  # transform to world frame (= base frame of left robot)
        point_cloud = open3d_to_point_cloud(pcd)

        point_cloud_cropped = filter_and_crop_point_cloud(point_cloud, confidence_map, BBOX_CLOTH_IN_THE_AIR)

        image_left_bgr = ImageConverter.from_numpy_int_format(image_left).image_in_opencv_format

        cv2.imshow(window_name, image_left_bgr)
        key = cv2.waitKey(1)
        if key == ord("q"):
            break
        elif key == ord("s"):
            sample = CompetitionInputSample(
                image_left,
                image_right,
                depth_map,
                point_cloud,
                depth_image,
                confidence_map,
                camera_pose,
                camera.intrinsics_matrix(),
                camera.resolution,
            )

            save_competition_input_sample(sample, dataset_dir, sample_index)
            logger.info(f"Saved sample_{sample_index:06d} to {dataset_dir}")
            sample_index += 1
