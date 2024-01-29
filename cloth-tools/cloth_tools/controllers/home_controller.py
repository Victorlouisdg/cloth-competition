import sys

import numpy as np
from cloth_tools.controllers.controller import Controller
from cloth_tools.drake.visualization import publish_dual_arm_joint_path
from cloth_tools.stations.competition_station import CompetitionStation
from loguru import logger


def execute_joint_path(dual_arm, path, duration):
    period = duration / len(path)
    for joints_left, joints_right in path:
        left_servo = dual_arm.left_manipulator.servo_to_joint_configuration(joints_left, period)
        right_servo = dual_arm.right_manipulator.servo_to_joint_configuration(joints_right, period)
        left_servo.wait()
        right_servo.wait()


class HomeController(Controller):
    """
    Opens the grippers and move the arms to their home positions.
    """

    def __init__(
        self,
        station: CompetitionStation,
    ):
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
        self._duration = 2.0 * planner._path_length_dual

        # make path take at least 2 seconds, this is a quick fix for short paths with too many waypoints causing servo problems
        self._duration = max(2.0, self._duration)

    def visualize_plan(self) -> None:
        path = self._path
        duration = self._duration
        publish_dual_arm_joint_path(
            path, duration, station._meshcat, station._diagram, station._context, *station._arm_indices
        )

    def execute_plan(self) -> None:
        if self._path is None:
            logger.info("Home not executed because no path was found.")
            return

        if self._duration is None:
            logger.info("Home not executed because path duration was not set.")
            return

        dual_arm = self.station.dual_arm
        path = self._path
        duration = self._duration

        assert dual_arm.left_manipulator.gripper is not None
        assert dual_arm.right_manipulator.gripper is not None

        left_opened = dual_arm.left_manipulator.gripper.open()
        right_opened = dual_arm.right_manipulator.gripper.open()
        left_opened.wait()
        right_opened.wait()

        # This is an incomplete fix to the problem that all paths currently have a fixed amount of waypoints, no matter how short they are.
        if np.isclose(duration, 0.0):
            return

        execute_joint_path(self.station.dual_arm, path, duration)

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
