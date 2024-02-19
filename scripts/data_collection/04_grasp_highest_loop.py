import time

from airo_camera_toolkit.cameras.multiprocess.multiprocess_video_recorder import MultiprocessVideoRecorder
from cloth_tools.bounding_boxes import BBOX_CLOTH_ON_TABLE
from cloth_tools.controllers.grasp_highest_controller import GraspHighestController
from cloth_tools.controllers.home_controller import HomeController
from cloth_tools.stations.competition_station import CompetitionStation
from loguru import logger

if __name__ == "__main__":
    import os

    os.environ["CUDA_VISIBLE_DEVICES"] = "1"

    station = CompetitionStation()

    dual_arm = station.dual_arm

    while True:
        start_time = time.time()

        video_recorder = MultiprocessVideoRecorder("camera")
        video_recorder.start()

        # Move the arms to their home positions
        home_controller = HomeController(station)
        home_controller.execute(interactive=False)

        grasp_highest_controller = GraspHighestController(station, BBOX_CLOTH_ON_TABLE)
        grasp_highest_controller.execute(interactive=True)

        time.sleep(5)

        video_recorder.stop()

        logger.info(f"Time taken: {time.time() - start_time:.2f} seconds")
        break
