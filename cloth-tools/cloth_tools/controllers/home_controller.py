import sys

from cloth_tools.controllers.controller import Controller
from cloth_tools.drake.visualization import publish_dual_arm_joint_path
from cloth_tools.path.execution import calculate_dual_path_duration, execute_dual_arm_joint_path
from cloth_tools.stations.competition_station import CompetitionStation
from loguru import logger


class HomeController(Controller):
    """
    Opens the grippers and move the arms to their home positions.
    """

    def __init__(self, station: CompetitionStation):
        self.station = station

    def plan(self) -> None:
        dual_arm = self.station.dual_arm
        start_joints_left = dual_arm.left_manipulator.get_joint_configuration()
        start_joints_right = dual_arm.right_manipulator.get_joint_configuration()
        goal_joints_left = self.station.home_joints_left
        goal_joints_right = self.station.home_joints_right

        planner = self.station.planner
        path = planner.plan_to_joint_configuration(
            start_joints_left, start_joints_right, goal_joints_left, goal_joints_right
        )
        self._path = path

    def visualize_plan(self) -> None:
        path = self._path
        station = self.station
        duration = calculate_dual_path_duration(path)
        publish_dual_arm_joint_path(
            path, duration, station._meshcat, station._diagram, station._context, *station._arm_indices
        )

    def execute_plan(self) -> None:
        if self._path is None:
            logger.info("Home not executed because no path was found.")
            return

        dual_arm = self.station.dual_arm
        path = self._path

        assert dual_arm.left_manipulator.gripper is not None
        assert dual_arm.right_manipulator.gripper is not None

        left_opened = dual_arm.left_manipulator.gripper.open()
        right_opened = dual_arm.right_manipulator.gripper.open()
        left_opened.wait()
        right_opened.wait()

        execute_dual_arm_joint_path(dual_arm, path)

    def execute_interactive(self) -> None:
        while True:
            self.plan()
            self.visualize_plan()
            answer = input(f"{self.__class__.__name__}: Execute? (y/n)")
            if answer == "y":
                self.execute_plan()
                return
            elif answer == "n":
                continue
            elif answer == "q":
                sys.exit(0)

    def execute(self, interactive: bool = True) -> None:
        logger.info(f"{self.__class__.__name__} started.")

        if interactive:
            self.execute_interactive()
        else:
            # Autonomous execution
            self.plan()
            self.visualize_plan()
            self.execute_plan()

        logger.info(f"{self.__class__.__name__} finished.")


if __name__ == "__main__":
    # Check whether all hardware is connected
    from cloth_tools.stations.competition_station import CompetitionStation

    station = CompetitionStation()
    controller = HomeController(station)
    controller.execute(interactive=True)
