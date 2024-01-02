from cloth_tools.controllers.controller import Controller
from cloth_tools.stations.dual_arm_station import DualArmStation
from loguru import logger


class HomeController(Controller):
    """
    Opens the grippers and move the arms to their home positions.

    TODO: use motion planning for this.
    """

    def __init__(self, station: DualArmStation):
        self.station = station

    def execute(self) -> None:
        logger.info(f"{self.__class__.__name__} started.")

        dual_arm = self.station.dual_arm

        home_joints_left = self.station.home_joints_left
        home_joints_right = self.station.home_joints_right

        left_opened = dual_arm.left_manipulator.gripper.open()
        right_opened = dual_arm.right_manipulator.gripper.open()
        left_opened.wait()
        right_opened.wait()

        dual_arm.left_manipulator.move_to_joint_configuration(home_joints_left).wait()
        dual_arm.right_manipulator.move_to_joint_configuration(home_joints_right).wait()


if __name__ == "__main__":
    # Check whether all hardware is connected
    from cloth_tools.stations.competition_station import CompetitionStation

    station = CompetitionStation()
    controller = HomeController(station)
    controller.execute()
