from pathlib import Path

import click
import cv2
from airo_camera_toolkit.interfaces import StereoRGBDCamera
from airo_camera_toolkit.point_clouds.conversions import open3d_to_point_cloud, point_cloud_to_open3d
from airo_camera_toolkit.utils.image_converter import ImageConverter
from cloth_tools.annotation.grasp_annotation import get_manual_grasp_annotation, save_grasp_info
from cloth_tools.dataset.bookkeeping import ensure_dataset_dir, find_highest_suffix
from cloth_tools.dataset.format import CompetitionObservation, save_competition_observation
from cloth_tools.stations.competition_station import CompetitionStation
from loguru import logger


def collect_competition_samples(
    station: CompetitionStation,
    dataset_dir: str | None = None,
) -> str:
    """Collects RGB-D data without moving the robots.
    This also means that you can't collect both start and result observation with this script.

    Args:
        station: The competition station with the camera and the robots.
        dataset_dir: The directory the samples will be added to.

    Returns:
        The path to the dataset directory.
    """

    camera = station.camera

    camera_pose_in_world = station.camera_pose
    arm_left_pose_in_world = station.left_arm_pose
    arm_right_pose_in_world = station.right_arm_pose
    right_camera_pose_in_left_camera = camera.pose_of_right_view_in_left_view

    camera_intrinsics = camera.intrinsics_matrix()
    camera_resolution = camera.resolution

    window_name = "Competition observation collection"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    dataset_dir = Path(ensure_dataset_dir(dataset_dir))

    while True:
        image_left = camera.get_rgb_image_as_int()
        image_right = camera._retrieve_rgb_image_as_int(StereoRGBDCamera.RIGHT_RGB)
        depth_map = camera._retrieve_depth_map()
        depth_image = camera._retrieve_depth_image()
        confidence_map = camera._retrieve_confidence_map()

        point_cloud_in_camera = camera._retrieve_colored_point_cloud()
        pcd_in_camera = point_cloud_to_open3d(point_cloud_in_camera)
        pcd = pcd_in_camera.transform(camera_pose_in_world)  # transform to world frame (= base frame of left robot)
        point_cloud = open3d_to_point_cloud(pcd)

        arm_left_tcp_pose_in_world = station.dual_arm.left_manipulator.get_tcp_pose()
        arm_left_joints = station.dual_arm.left_manipulator.get_joint_configuration()
        arm_right_tcp_pose_in_world = station.dual_arm.right_manipulator.get_tcp_pose()
        arm_right_joints = station.dual_arm.right_manipulator.get_joint_configuration()

        image_left_bgr = ImageConverter.from_numpy_int_format(image_left).image_in_opencv_format

        observation = CompetitionObservation(
            image_left=image_left,
            image_right=image_right,
            depth_map=depth_map,
            point_cloud=point_cloud,
            depth_image=depth_image,
            confidence_map=confidence_map,
            camera_pose_in_world=camera_pose_in_world,
            arm_left_pose_in_world=arm_left_pose_in_world,
            arm_right_pose_in_world=arm_right_pose_in_world,
            arm_left_joints=arm_left_joints,
            arm_right_joints=arm_right_joints,
            arm_left_tcp_pose_in_world=arm_left_tcp_pose_in_world,
            arm_right_tcp_pose_in_world=arm_right_tcp_pose_in_world,
            right_camera_pose_in_left_camera=right_camera_pose_in_left_camera,
            camera_intrinsics=camera_intrinsics,
            camera_resolution=camera_resolution,
        )

        cv2.imshow(window_name, image_left_bgr)
        key = cv2.waitKey(1)
        if key == ord("q"):
            break
        elif key == ord("s"):
            sample_index = find_highest_suffix(dataset_dir, "sample") + 1

            sample_dir = dataset_dir / f"sample_{sample_index:06d}"
            observation_dir = str(sample_dir / "observation")
            save_competition_observation(observation, observation_dir)
            logger.info(f"Saved observation to {observation_dir}")

            grasp_info = get_manual_grasp_annotation(
                image_left, depth_map, point_cloud, camera_pose_in_world, camera_intrinsics, log_to_rerun=True
            )

            grasp_dir = Path(sample_dir) / "grasp"
            save_grasp_info(str(grasp_dir), grasp_info)

    return dataset_dir


@click.command()
@click.option("--dataset_dir", type=str)
def collect_competition_observations(dataset_dir: str | None = None) -> str:
    """Collects observations as used in the ICRA 2024 Cloth Competition.

    Args:
        dataset_dir: The directory the observations will be added to.

    Returns:
        The path to the dataset directory.
    """
    station = CompetitionStation()

    dataset_dir = collect_competition_samples(station, dataset_dir)
    return dataset_dir


if __name__ == "__main__":
    collect_competition_observations()
