import click
import cv2
from airo_camera_toolkit.cameras.zed.zed2i import Zed2i
from airo_camera_toolkit.interfaces import StereoRGBDCamera
from airo_camera_toolkit.utils.image_converter import ImageConverter
from airo_typing import CameraExtrinsicMatrixType
from cloth_tools.config import load_camera_pose_in_left_and_right
from cloth_tools.dataset.bookkeeping import ensure_dataset_dir, find_highest_suffix
from cloth_tools.dataset.format import CompetitionInputSample, save_competition_input_sample
from cloth_tools.stations.coordinate_frames import create_egocentric_world_frame
from loguru import logger


def collect_competition_input_samples(
    camera: Zed2i,
    camera_pose_in_world: CameraExtrinsicMatrixType,
    camera_pose_in_left: CameraExtrinsicMatrixType,
    camera_pose_in_right: CameraExtrinsicMatrixType,
    dataset_dir: str | None = None,
) -> str:
    """Collects samples of the input RGB-D data without moving the robots.
    This also means that no grasp labels are collected.

    Args:
        camera: The camera. (Currently must be a ZED2i camera for the confidence map.)
        camera_pose: The camera pose in the world frame, X_W_C.
        left_arm_pose: The pose of the left robot arm X_W_LCB.
        right_arm_pose: The pose of the right robot arm X_W_RCB.
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

        point_cloud_in_camera = camera._retrieve_colored_point_cloud()
        # pcd_in_camera = point_cloud_to_open3d(point_cloud_in_camera)
        # pcd = pcd_in_camera.transform(camera_pose)  # transform to world frame (= base frame of left robot)
        # point_cloud = open3d_to_point_cloud(pcd)
        # point_cloud_cropped = filter_and_crop_point_cloud(point_cloud, confidence_map, BBOX_CLOTH_IN_THE_AIR)

        # TODO also save right intrinsics as they might be slightly different
        # TODO save pose of right_camera_frame_in_left_camera_frame?
        # TODO decided whether to save the point cloud in the camera frame (or in the world frame)
        # TODO save bboxes?

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
                point_cloud_in_camera,
                depth_image,
                confidence_map,
                camera_pose_in_world,
                camera_pose_in_left,
                camera_pose_in_right,
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
    camera_pose_in_left, camera_pose_in_right = load_camera_pose_in_left_and_right()

    X_W_C, X_W_LCB, X_W_RCB = create_egocentric_world_frame(camera_pose_in_left, camera_pose_in_right)

    dataset_dir = collect_competition_input_samples(camera, X_W_C, X_W_LCB, X_W_RCB, dataset_dir)
    return dataset_dir


if __name__ == "__main__":
    collect_competition_input_samples_with_zed2i()
