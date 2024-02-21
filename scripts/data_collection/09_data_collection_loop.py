import time
from pathlib import Path

from airo_camera_toolkit.cameras.multiprocess.multiprocess_video_recorder import MultiprocessVideoRecorder
from airo_camera_toolkit.interfaces import StereoRGBDCamera
from airo_camera_toolkit.point_clouds.conversions import open3d_to_point_cloud, point_cloud_to_open3d
from cloth_tools.bounding_boxes import BBOX_CLOTH_IN_THE_AIR, BBOX_CLOTH_ON_TABLE
from cloth_tools.controllers.grasp_hanging_controller import GraspHangingController
from cloth_tools.controllers.grasp_highest_controller import GraspHighestController
from cloth_tools.controllers.grasp_lowest_controller import GraspLowestController
from cloth_tools.controllers.home_controller import HomeController
from cloth_tools.controllers.stretch_controller import StretchController
from cloth_tools.dataset.bookkeeping import ensure_dataset_dir, find_highest_suffix
from cloth_tools.dataset.format import CompetitionObservation, save_competition_observation
from cloth_tools.motion_blur_detector import MotionBlurDetector
from cloth_tools.stations.competition_station import CompetitionStation
from loguru import logger


def collect_observation(station: CompetitionStation) -> str:
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
    return observation


if __name__ == "__main__":
    # import os

    # os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    # os.environ["CUDA_VISIBLE_DEVICES"] = "3"

    station = CompetitionStation()

    dual_arm = station.dual_arm
    camera = station.camera

    camera_pose_in_world = station.camera_pose
    arm_left_pose_in_world = station.left_arm_pose
    arm_right_pose_in_world = station.right_arm_pose
    right_camera_pose_in_left_camera = camera.pose_of_right_view_in_left_view

    camera_intrinsics = camera.intrinsics_matrix()
    camera_resolution = camera.resolution

    dataset_dir = Path(ensure_dataset_dir("notebooks/data/cloth_competition_dataset_0000"))

    while True:
        start_time = time.time()

        sample_index = find_highest_suffix(dataset_dir, "sample") + 1
        sample_dir = dataset_dir / f"sample_{sample_index:06d}"
        sample_dir.mkdir(parents=True, exist_ok=True)

        video_path = str(sample_dir / "episode.mp4")
        video_recorder = MultiprocessVideoRecorder("camera", video_path)
        video_recorder.start()

        # Move the arms to their home positions
        home_controller = HomeController(station)
        home_controller.execute(interactive=False)

        # Start of new episode
        grasp_highest_controller = GraspHighestController(station, BBOX_CLOTH_ON_TABLE)
        grasp_highest_controller.execute(interactive=True)

        motion_blur_detector = MotionBlurDetector(station.camera, station.hanging_cloth_crop)
        motion_blur_detector.wait_for_blur_to_stabilize()

        grasp_lowest_controller = GraspLowestController(station, BBOX_CLOTH_IN_THE_AIR)
        grasp_lowest_controller.execute(interactive=True)

        motion_blur_detector = MotionBlurDetector(station.camera, station.hanging_cloth_crop)
        motion_blur_detector.wait_for_blur_to_stabilize(timeout=20)

        # Save competition input data here
        observation_start_dir = str(sample_dir / "observation_start")
        observation_start = collect_observation(station)
        save_competition_observation(observation_start, observation_start_dir)

        grasp_hanging_controller = GraspHangingController(station, BBOX_CLOTH_IN_THE_AIR, sample_dir)
        grasp_hanging_controller.execute(interactive=True)

        stretch_controller = StretchController(station)
        stretch_controller.execute(interactive=True)

        motion_blur_detector = MotionBlurDetector(station.camera, station.hanging_cloth_crop)
        motion_blur_detector.wait_for_blur_to_stabilize(timeout=15)

        # Save results here
        observation_result_dir = str(sample_dir / "observation_result")
        observation_result = collect_observation(station)
        save_competition_observation(observation_result, observation_result_dir)

        video_recorder.stop()

        logger.info(f"Time taken: {time.time() - start_time:.2f} seconds")
