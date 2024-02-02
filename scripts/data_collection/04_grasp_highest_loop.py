from cloth_tools.bounding_boxes import BBOX_CLOTH_ON_TABLE
from cloth_tools.controllers.grasp_highest_controller import GraspHighestController
from cloth_tools.controllers.home_controller import HomeController
from cloth_tools.stations.competition_station import CompetitionStation

station = CompetitionStation()

dual_arm = station.dual_arm

while True:
    # Move the arms to their home positions
    home_controller = HomeController(station)
    home_controller.execute(interactive=False)

    grasp_highest_controller = GraspHighestController(station, BBOX_CLOTH_ON_TABLE)
    grasp_highest_controller.execute(interactive=True)
