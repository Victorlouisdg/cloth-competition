import click
import cv2
from airo_camera_toolkit.cameras.zed.zed2i import Zed2i
from airo_camera_toolkit.interfaces import StereoRGBDCamera
from airo_camera_toolkit.point_clouds.conversions import open3d_to_point_cloud, point_cloud_to_open3d
from airo_camera_toolkit.utils.image_converter import ImageConverter
from airo_typing import CameraExtrinsicMatrixType
from cloth_tools.bounding_boxes import BBOX_CLOTH_IN_THE_AIR
from cloth_tools.config import load_camera_pose_in_left_and_right
from cloth_tools.dataset.bookkeeping import ensure_dataset_dir, find_highest_suffix
from cloth_tools.dataset.format import CompetitionInputSample, save_competition_input_sample
from cloth_tools.point_clouds.operations import filter_and_crop_point_cloud
from loguru import logger


def collect_competition_input_samples(
    camera: Zed2i, camera_pose: CameraExtrinsicMatrixType, dataset_dir: str | None = None
) -> str:
    """Collects samples of the input RGB-D data without moving the robots.
    This also means that no grasp labels are collected.

    Args:
        camera: The camera. (Currently must be a ZED2i camera for the confidence map.)
        camera_pose: The camera pose in the world frame.
        dataset_dir: The directory the samples will be added to.

    Returns:
        The path to the dataset directory.
    """

    window_name = "RGBD sample collection"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    dataset_dir = ensure_dataset_dir(dataset_dir)
    sample_index = find_highest_suffix(dataset_dir, "sample") + 1

    while True:
        image_left = camera.get_rgb_image_as_int()
        image_right = camera._retrieve_rgb_image_as_int(StereoRGBDCamera.RIGHT_RGB)
        depth_map = camera._retrieve_depth_map()
        depth_image = camera._retrieve_depth_image()
        confidence_map = camera._retrieve_confidence_map()

        # Retrieve, transform, filter and crop the point cloud
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
                point_cloud_cropped,
                depth_image,
                confidence_map,
                camera_pose,
                camera.intrinsics_matrix(),
                camera.resolution,
            )

            save_competition_input_sample(sample, dataset_dir, sample_index)
            logger.info(f"Saved sample_{sample_index:06d} to {dataset_dir}")
            sample_index += 1
    return dataset_dir


@click.command()
@click.option("--dataset_dir", type=str)
def collect_competition_input_samples_with_zed2i(dataset_dir: str | None = None) -> str:
    """Collects samples of the competition input data with a ZED2i camera.

    Args:
        dataset_dir: The directory the samples will be added to.

    Returns:
        The path to the dataset directory.
    """
    camera = Zed2i(resolution=Zed2i.RESOLUTION_2K, depth_mode=Zed2i.NEURAL_DEPTH_MODE, fps=15)
    camera_pose, _ = load_camera_pose_in_left_and_right()

    dataset_dir = collect_competition_input_samples(camera, camera_pose, dataset_dir)
    return dataset_dir


if __name__ == "__main__":
    collect_competition_input_samples_with_zed2i()
