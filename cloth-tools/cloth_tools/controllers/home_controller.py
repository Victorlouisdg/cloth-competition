import sys

from airo_drake import animate_dual_joint_trajectory, time_parametrize_toppra
from airo_planner import PlannerError
from cloth_tools.controllers.controller import Controller
from cloth_tools.stations.competition_station import CompetitionStation
from cloth_tools.trajectory_execution import execute_dual_arm_drake_trajectory
from loguru import logger


class HomeController(Controller):
    """
    Opens the grippers and move the arms to their home positions.
    """

    def __init__(
        self,
        station: CompetitionStation,
        move_left_home: bool = True,
        move_right_home: bool = True,
        open_left_gripper: bool = True,
        open_right_gripper: bool = True,
    ):
        self.station = station
        self.move_left_home = move_left_home
        self.move_right_home = move_right_home
        self.open_left_gripper = open_left_gripper
        self.open_right_gripper = open_right_gripper

        # Attributes set by plan()
        self._path = None
        self._trajectory = None

    def plan(self) -> None:
        dual_arm = self.station.dual_arm
        start_joints_left = dual_arm.left_manipulator.get_joint_configuration()
        start_joints_right = dual_arm.right_manipulator.get_joint_configuration()
        goal_joints_left = self.station.home_joints_left
        goal_joints_right = self.station.home_joints_right

        if not self.move_left_home:
            goal_joints_left = None

        if not self.move_right_home:
            goal_joints_right = None

        planner = self.station.planner

        try:
            path = planner.plan_to_joint_configuration(
                start_joints_left, start_joints_right, goal_joints_left, goal_joints_right
            )
            self._path = path
        except PlannerError as e:
            logger.warning(f"Failed to plan home path: {e}")
            self._path = None
            return

        drake_plant = self.station.drake_scene.robot_diagram.plant()
        trajectory = time_parametrize_toppra(drake_plant, path)

        self._trajectory = trajectory

    def visualize_plan(self) -> None:
        scene = self.station.drake_scene

        if self._trajectory is not None:
            animate_dual_joint_trajectory(
                scene.meshcat, scene.robot_diagram, scene.arm_left_index, scene.arm_right_index, self._trajectory
            )

    def execute_plan(self) -> None:
        if self._trajectory is None:
            logger.info("Home not executed because no trajectory was found.")
            return

        dual_arm = self.station.dual_arm
        self._path

        assert dual_arm.left_manipulator.gripper is not None
        assert dual_arm.right_manipulator.gripper is not None

        if self.open_left_gripper:
            left_opened = dual_arm.left_manipulator.gripper.open()
        if self.open_right_gripper:
            right_opened = dual_arm.right_manipulator.gripper.open()
        if self.open_left_gripper:
            left_opened.wait()
        if self.open_right_gripper:
            right_opened.wait()

        execute_dual_arm_drake_trajectory(dual_arm, self._trajectory)

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
