from cloth_tools.controllers.controller import Controller
from cloth_tools.stations.dual_arm_station import DualArmStation
from loguru import logger


class BasicHomeController(Controller):
    """
    Opens the grippers and move the arms to their home positions.

    TODO: use motion planning for this.
    """

    def __init__(
        self,
        station: DualArmStation,
    ):
        self.station = station

    def execute(self) -> None:
        logger.info(f"{self.__class__.__name__} started.")

        dual_arm = self.station.dual_arm

        assert dual_arm.left_manipulator.gripper is not None
        assert dual_arm.right_manipulator.gripper is not None

        left_opened = dual_arm.left_manipulator.gripper.open()
        right_opened = dual_arm.right_manipulator.gripper.open()
        left_opened.wait()
        right_opened.wait()

        dual_arm.left_manipulator.move_to_joint_configuration(self.station.home_joints_left).wait()
        dual_arm.right_manipulator.move_to_joint_configuration(self.station.home_joints_right).wait()

        logger.info(f"{self.__class__.__name__} finished.")


if __name__ == "__main__":
    # Check whether all hardware is connected
    from cloth_tools.stations.competition_station import CompetitionStation

    station = CompetitionStation()
    controller = BasicHomeController(station)
    controller.execute()
