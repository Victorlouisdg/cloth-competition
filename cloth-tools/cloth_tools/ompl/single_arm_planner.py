from typing import Callable, List, Optional

import numpy as np
from airo_typing import HomogeneousMatrixType, JointConfigurationType
from cloth_tools.ompl.state_space import (
    function_numpy_to_ompl,
    numpy_to_ompl_state,
    ompl_path_to_numpy,
    revolute_joints_state_space,
)
from cloth_tools.planning.interfaces import SingleArmMotionPlanner
from loguru import logger
from ompl import base as ob
from ompl import geometric as og

JointConfigurationCheckerType = Callable[[JointConfigurationType], bool]
InverseKinematicsType = Callable[[JointConfigurationType], List[HomogeneousMatrixType]]


class SingleArmOmplPlanner(SingleArmMotionPlanner):
    """Utility class for single-arm motion planning using OMPL.

    This class only plan in joint space.

    The purpose of this class is to make working with OMPL easier. It basically
    just handles the creation of OMPL objects and the conversion between numpy
    arrays and OMPL states and paths. After creating an instance of this class,
    you can also extract the SimpleSetup object and use it directly if you want.
    This can be useful for benchmarking with the OMPL benchmarking tools.
    """

    def __init__(
        self,
        is_state_valid_fn: JointConfigurationCheckerType,
        inverse_kinematics_fn: Optional[InverseKinematicsType] = None,
        max_planning_time: float = 30.0,
        num_interpolated_states: Optional[int] = 500,
    ):
        """Instiatiate a single-arm motion planner that uses OMPL. This creates
        a SimpleSetup object. Note that planning to TCP poses is only possible
        if the inverse kinematics function is provided.

        Args:
            is_state_valid_fn: A function that checks if a given joint configuration is valid.
            inverse_kinematics_fn: A function that computes the inverse kinematics of a given TCP pose.
            max_planning_time: The maximum time allowed for planning.
            num_interpolated_states: The amount of states the solution path should be interpolated to, if None then no interpolation is done.
        """
        self.is_state_valid_fn = is_state_valid_fn
        self.inverse_kinematics_fn = inverse_kinematics_fn

        # Settings
        self.max_planning_time = max_planning_time
        self.num_interpolated_states = num_interpolated_states

        # Currently we only planning for 6 DoF arms
        self.degrees_of_freedom: int = 6

        self._simple_setup = self._create_simple_setup()

        self._path_length: float | None = None

    def _create_simple_setup(self):
        # Create state space and convert to OMPL compatible data types
        space = revolute_joints_state_space(self.degrees_of_freedom)

        is_state_valid_ompl = function_numpy_to_ompl(self.is_state_valid_fn, self.degrees_of_freedom)

        # Configure the SimpleSetup object
        simple_setup = og.SimpleSetup(space)
        simple_setup.setStateValidityChecker(ob.StateValidityCheckerFn(is_state_valid_ompl))

        simple_setup.setOptimizationObjective(ob.PathLengthOptimizationObjective(simple_setup.getSpaceInformation()))

        # TODO: Should investigate effect of this further
        step = float(np.deg2rad(5))
        resolution = step / space.getMaximumExtent()
        simple_setup.getSpaceInformation().setStateValidityCheckingResolution(resolution)

        planner = og.RRTConnect(simple_setup.getSpaceInformation())
        simple_setup.setPlanner(planner)

        return simple_setup

    def _set_start_and_goal_configurations(
        self, start_configuration: JointConfigurationType, goal_configuration: JointConfigurationType
    ) -> None:
        space = self._simple_setup.getStateSpace()
        start_state = numpy_to_ompl_state(start_configuration, space)
        goal_state = numpy_to_ompl_state(goal_configuration, space)
        self._simple_setup.setStartAndGoalStates(start_state, goal_state)

    def plan_to_joint_configuration(
        self, start_configuration: JointConfigurationType, goal_configuration: JointConfigurationType
    ):
        self._simple_setup.clear()  # Needed to support multiple calls with different start/goal configurations

        self._set_start_and_goal_configurations(start_configuration, goal_configuration)
        simple_setup = self._simple_setup

        simple_setup.solve(self.max_planning_time)

        if not simple_setup.haveExactSolutionPath():
            return None

        # Simplify, smooth, simplify again and interpolate the solution path
        # We simply twice because the smoothing changes the density of the
        # states along the path. If you then execute those paths, the robot will
        # move very slow for some parts of the path and very fast for other
        # parts. By simplifying and interpolating, we get a path with much more
        # evenly spaced states.
        simple_setup.simplifySolution()
        path_simplifier = og.PathSimplifier(simple_setup.getSpaceInformation())
        path = simple_setup.getSolutionPath()
        path_simplifier.smoothBSpline(path)
        simple_setup.simplifySolution()
        if self.num_interpolated_states is not None:
            path.interpolate(self.num_interpolated_states)

        self._path_length = path.length()

        path_numpy = ompl_path_to_numpy(path, self.degrees_of_freedom)
        return path_numpy

    def plan_to_tcp_pose(self, start_configuration, tcp_pose_in_base):
        if self.inverse_kinematics_fn is None:
            logger.warning("Planning to TCP pose attempted but inverse_kinematics_fn was provided, returing None.")
            return None

        ik_solutions = self.inverse_kinematics_fn(tcp_pose_in_base)
        if ik_solutions is None or len(ik_solutions) == 0:
            logger.info("IK returned no solutions, returning None.")
            return None

        ik_solutions_valid = [s for s in ik_solutions if self.is_state_valid_fn(s)]
        if len(ik_solutions_valid) == 0:
            logger.info("All IK solutions are invalid, returning None.")
            return None

        # Try solving to each IK solution in joint space.
        paths = []
        path_lengths = []
        for ik_solution in ik_solutions_valid:
            path = self.plan_to_joint_configuration(start_configuration, ik_solution)
            if path is not None:
                paths.append(path)
                path_lengths.append(self._path_length)

        if len(paths) == 0:
            logger.info("No paths founds towards any IK solutions, returning None.")
            return None

        logger.info(f"Found {len(paths)} paths towards IK solutions.")

        # Pick the shortest path
        shortest_path_idx = np.argmin(path_lengths)
        shortest_path = paths[shortest_path_idx]
        return shortest_path
