from typing import Callable, Optional

import numpy as np
from airo_typing import JointConfigurationType
from cloth_tools.ompl.state_space import (
    function_numpy_to_ompl,
    numpy_to_ompl_state,
    ompl_path_to_numpy,
    revolute_joints_state_space,
)
from cloth_tools.planning.interfaces import SingleArmMotionPlanner
from ompl import base as ob
from ompl import geometric as og

JointConfigurationCheckerType = Callable[[JointConfigurationType], bool]


class SingleArmOmplPlanner(SingleArmMotionPlanner):
    def __init__(
        self,
        is_state_valid_fn: JointConfigurationCheckerType,
        max_planning_time: float = 30.0,
        num_interpolated_states: Optional[int] = 100,
    ):
        """Instiatiate a single-arm motion planner that uses OMPL.

        For now, I've chosen to not create the simple setup in the constructor,
        because it requires the state space to be known. Instead, I've chosen
        to create the simple setup in the plan_ methods, so we chould still
        chose between joint or task space planning there.

        Args:
            is_state_valid_fn: A function that checks if a given joint configuration is valid.
            max_planning_time: The maximum time allowed for planning.
            num_interpolated_states: The amount of states the solution path should be interpolated to, if None then no interpolation is done.
        """
        self.is_state_valid_fn = is_state_valid_fn

        # Settings
        self.max_planning_time = max_planning_time
        self.num_interpolated_states = num_interpolated_states

        # Currently we only planning for 6 DoF arms
        self.degrees_of_freedom: int = 6

        self._simple_setup = self._create_simple_setup()

    def _create_simple_setup(self):
        # Create state space and convert to OMPL compatible data types
        space = revolute_joints_state_space(self.degrees_of_freedom)

        is_state_valid_ompl = function_numpy_to_ompl(self.is_state_valid_fn, self.degrees_of_freedom)

        # Configure the SimpleSetup object
        simple_setup = og.SimpleSetup(space)
        simple_setup.setStateValidityChecker(ob.StateValidityCheckerFn(is_state_valid_ompl))

        # TODO: Should investigate effect of this further
        step = float(np.deg2rad(5))
        resolution = step / space.getMaximumExtent()
        simple_setup.getSpaceInformation().setStateValidityCheckingResolution(resolution)

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
        self._set_start_and_goal_configurations(start_configuration, goal_configuration)
        simple_setup = self._simple_setup

        simple_setup.solve(self.max_planning_time)

        if not simple_setup.haveExactSolutionPath():
            return None

        simple_setup.simplifySolution()
        path = simple_setup.getSolutionPath()
        if self.num_interpolated_states is not None:
            path.interpolate(self.num_interpolated_states)

        path_numpy = ompl_path_to_numpy(path, self.degrees_of_freedom)
        return path_numpy

    def plan_to_tcp_pose(self, start_configuration, tcp_pose_in_base):
        raise NotImplementedError
