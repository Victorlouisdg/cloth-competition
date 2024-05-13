import time
from pathlib import Path

from airo_camera_toolkit.cameras.multiprocess.multiprocess_video_recorder import MultiprocessVideoRecorder
from cloth_tools.bounding_boxes import BBOX_CLOTH_IN_THE_AIR
from cloth_tools.controllers.grasp_hanging_controller import GraspHangingController
from cloth_tools.controllers.hang_controller import HangController
from cloth_tools.controllers.stretch_controller import StretchController
from cloth_tools.dataset.bookkeeping import datetime_for_filename
from cloth_tools.dataset.collection import collect_observation
from cloth_tools.dataset.format import save_competition_observation
from cloth_tools.motion_blur_detector import MotionBlurDetector
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

    dataset_dir = Path("notebooks/data/cloth_competition_dataset_0003_dev")

    while True:
        start_time = time.time()

        sample_id = datetime_for_filename()
        sample_dir = dataset_dir / f"sample_{sample_id}"
        sample_dir.mkdir(parents=True, exist_ok=True)

        video_path = str(sample_dir / "episode.mp4")
        video_recorder = MultiprocessVideoRecorder("camera", video_path)
        video_recorder.start()

        controller = HangController(station)
        controller.execute()

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
