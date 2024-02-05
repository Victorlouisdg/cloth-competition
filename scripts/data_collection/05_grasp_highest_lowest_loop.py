import time

from cloth_tools.bounding_boxes import BBOX_CLOTH_IN_THE_AIR, BBOX_CLOTH_ON_TABLE
from cloth_tools.controllers.grasp_highest_controller import GraspHighestController
from cloth_tools.controllers.grasp_lowest_controller import GraspLowestController
from cloth_tools.controllers.home_controller import HomeController
from cloth_tools.stations.competition_station import CompetitionStation
from loguru import logger

station = CompetitionStation()

dual_arm = station.dual_arm

while True:
    # Move the arms to their home positions
    home_controller = HomeController(station)
    home_controller.execute(interactive=False)

    grasp_highest_controller = GraspHighestController(station, BBOX_CLOTH_ON_TABLE)
    grasp_highest_controller.execute(interactive=True)

    time_to_stop_swinging = 5
    logger.info(f"Waiting {time_to_stop_swinging} seconds for the cloth to stop swinging")
    time.sleep(time_to_stop_swinging)

    grasp_lowest_controller = GraspLowestController(station, BBOX_CLOTH_IN_THE_AIR)
    grasp_lowest_controller.execute(interactive=True)
