import time
from pathlib import Path

from airo_camera_toolkit.cameras.multiprocess.multiprocess_video_recorder import MultiprocessVideoRecorder
from cloth_tools.bounding_boxes import BBOX_CLOTH_IN_THE_AIR, BBOX_CLOTH_ON_TABLE
from cloth_tools.controllers.grasp_hanging_controller import GraspHangingController
from cloth_tools.controllers.grasp_highest_controller import GraspHighestController
from cloth_tools.controllers.grasp_lowest_controller import GraspLowestController
from cloth_tools.controllers.home_controller import HomeController
from cloth_tools.controllers.stretch_controller import StretchController
from cloth_tools.dataset.bookkeeping import ensure_dataset_dir, find_highest_suffix
from cloth_tools.dataset.collection import collect_observation
from cloth_tools.dataset.format import save_competition_observation
from cloth_tools.motion_blur_detector import MotionBlurDetector
from cloth_tools.point_clouds.cloth_detection import sufficient_points_in_bbox
from cloth_tools.stations.competition_station import CompetitionStation
from loguru import logger

if __name__ == "__main__":
    station = CompetitionStation()

    dual_arm = station.dual_arm
    camera = station.camera

    camera_pose_in_world = station.camera_pose
    arm_left_pose_in_world = station.left_arm_pose
    arm_right_pose_in_world = station.right_arm_pose
    right_camera_pose_in_left_camera = camera.pose_of_right_view_in_left_view

    camera_intrinsics = camera.intrinsics_matrix()
    camera_resolution = camera.resolution

    dataset_dir = Path(ensure_dataset_dir("notebooks/data/cloth_competition_dataset_0001"))

    while True:
        start_time = time.time()

        sample_index = find_highest_suffix(dataset_dir, "sample") + 1
        sample_dir = dataset_dir / f"sample_{sample_index:06d}"
        sample_dir.mkdir(parents=True, exist_ok=True)

        video_path = str(sample_dir / "episode.mp4")
        video_recorder = MultiprocessVideoRecorder("camera", video_path)
        video_recorder.start()

        # Move the arms to their home positions
        while True:
            home_controller = HomeController(station)
            home_controller.execute(interactive=False)

            dual_arm.left_manipulator.rtde_control.zeroFtSensor()
            dual_arm.right_manipulator.rtde_control.zeroFtSensor()

            # Start of new episode
            grasp_highest_controller = GraspHighestController(station, BBOX_CLOTH_ON_TABLE)
            grasp_highest_controller.execute(interactive=True)

            motion_blur_detector = MotionBlurDetector(station.camera, station.hanging_cloth_crop)
            motion_blur_detector.wait_for_blur_to_stabilize()

            if not sufficient_points_in_bbox(station, BBOX_CLOTH_IN_THE_AIR):
                logger.warning("No cloth in the air, going back home to restart")
                home_controller.execute(interactive=False)
                continue

            grasp_lowest_controller = GraspLowestController(station, BBOX_CLOTH_IN_THE_AIR)
            grasp_lowest_controller.execute(interactive=False)

            motion_blur_detector = MotionBlurDetector(station.camera, station.hanging_cloth_crop)
            motion_blur_detector.wait_for_blur_to_stabilize(timeout=20)

            if sufficient_points_in_bbox(station, BBOX_CLOTH_IN_THE_AIR):
                break
            else:
                logger.warning("No cloth in the air, going back home to restart")
                home_controller.execute(interactive=False)
                continue

        # Save competition input data here
        observation_start_dir = str(sample_dir / "observation_start")
        observation_start = collect_observation(station)
        save_competition_observation(observation_start, observation_start_dir)

        # TODO attempt GraspHaningController to accept observation_start as input
        # Currently, it will retrieve the next camera frame, which is very slightly different
        grasp_hanging_controller = GraspHangingController(station, BBOX_CLOTH_IN_THE_AIR, sample_dir)
        grasp_hanging_controller.execute(interactive=False)

        stretch_controller = StretchController(station)
        stretch_controller.execute(interactive=False)

        motion_blur_detector = MotionBlurDetector(station.camera, station.hanging_cloth_crop)
        motion_blur_detector.wait_for_blur_to_stabilize(timeout=15)

        # Save results here
        observation_result_dir = str(sample_dir / "observation_result")
        observation_result = collect_observation(station)
        save_competition_observation(observation_result, observation_result_dir)

        video_recorder.stop()

        logger.info(f"Time taken: {time.time() - start_time:.2f} seconds")
