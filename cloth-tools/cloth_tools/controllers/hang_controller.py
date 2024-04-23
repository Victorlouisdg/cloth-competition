from cloth_tools.bounding_boxes import BBOX_CLOTH_IN_THE_AIR, BBOX_CLOTH_ON_TABLE
from cloth_tools.controllers.controller import Controller
from cloth_tools.controllers.grasp_highest_controller import GraspHighestController
from cloth_tools.controllers.grasp_lowest_controller import GraspLowestController
from cloth_tools.controllers.home_controller import HomeController
from cloth_tools.motion_blur_detector import MotionBlurDetector
from cloth_tools.point_clouds.cloth_detection import sufficient_points_in_bbox
from cloth_tools.stations.competition_station import CompetitionStation
from loguru import logger


class HangController(Controller):
    """Controller that handles the initialization sequence for the ICRA 2024 Cloth Competition."""

    def __init__(self, station: CompetitionStation):
        self.station = station

    def loop_until_cloth_hanging(self):
        station = self.station
        dual_arm = self.station.dual_arm
        # Move the arms to their home positions
        while True:
            home_controller = HomeController(station)
            home_controller.execute(interactive=False)

            # This is done here because we know the robot should not be holding anything.
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

    def execute(self) -> None:
        logger.info(f"{self.__class__.__name__} started.")
        self.loop_until_cloth_hanging()
        logger.info(f"{self.__class__.__name__} finished.")


if __name__ == "__main__":
    from cloth_tools.stations.competition_station import CompetitionStation

    station = CompetitionStation()
    controller = HangController(station)
    controller.execute()
