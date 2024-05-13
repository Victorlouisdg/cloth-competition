import time
from pathlib import Path

import rerun as rr
from airo_camera_toolkit.cameras.multiprocess.multiprocess_video_recorder import MultiprocessVideoRecorder
from cloth_tools.controllers.dry_run_grasp_controller import DryRunGraspController
from cloth_tools.controllers.hang_controller import HangController
from cloth_tools.controllers.stretch_controller import StretchController
from cloth_tools.dataset.bookkeeping import datetime_for_filename
from cloth_tools.dataset.collection import collect_observation
from cloth_tools.dataset.format import save_competition_observation
from cloth_tools.motion_blur_detector import MotionBlurDetector
from cloth_tools.multiprocess_viewer import MultiprocessViewer
from cloth_tools.stations.competition_station import CompetitionStation
from loguru import logger

if __name__ == "__main__":
    rr.init("Remote dry run")
    rr.spawn(memory_limit="25%")

    current_team = "dev_team"
    dataset_dir = Path(f"notebooks/data/evaluation_icra_2024/{current_team}")

    station = CompetitionStation()
    dual_arm = station.dual_arm
    camera = station.camera
    camera_pose_in_world = station.camera_pose
    arm_left_pose_in_world = station.left_arm_pose
    arm_right_pose_in_world = station.right_arm_pose
    right_camera_pose_in_left_camera = camera.pose_of_right_view_in_left_view
    camera_intrinsics = camera.intrinsics_matrix()
    camera_resolution = camera.resolution

    start_time = time.time()

    sample_id = datetime_for_filename()

    sample_dir = dataset_dir / f"sample_{sample_id}"
    sample_dir.mkdir(parents=True, exist_ok=False)

    grasps_dir = dataset_dir / f"grasps_{sample_id}"
    grasps_dir.mkdir(parents=True, exist_ok=False)

    viewer = MultiprocessViewer("camera")
    viewer.start()

    video_path = str(sample_dir / "episode.mp4")
    video_recorder = MultiprocessVideoRecorder("camera", video_path)
    video_recorder.start()

    # run hangcontroller
    hang_controller = HangController(station)
    hang_controller.execute()

    # Save competition input data here
    observation_start_dir = str(sample_dir / "observation_start")
    observation_start = collect_observation(station)
    save_competition_observation(observation_start, observation_start_dir)

    dry_run_grasp_controller = DryRunGraspController(station, sample_dir, grasps_dir)
    dry_run_grasp_controller.execute()

    # TODO Do I need to handle case where no grasp sent or grasp not plannable?
    # Maybe just Keyboard interrupt?

    stretch_controller = StretchController(station)
    stretch_controller.execute(interactive=False)

    motion_blur_detector = MotionBlurDetector(station.camera, station.hanging_cloth_crop)
    motion_blur_detector.wait_for_blur_to_stabilize(timeout=15)

    # Save results here
    observation_result_dir = str(sample_dir / "observation_result")
    observation_result = collect_observation(station)
    save_competition_observation(observation_result, observation_result_dir)

    video_recorder.stop()
    viewer.stop()

    logger.info(f"Time taken: {time.time() - start_time:.2f} seconds")
