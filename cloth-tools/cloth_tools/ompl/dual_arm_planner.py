from typing import Callable, List, Optional, Tuple

import numpy as np
from airo_typing import HomogeneousMatrixType, JointConfigurationType
from cloth_tools.ompl.single_arm_planner import SingleArmOmplPlanner
from cloth_tools.ompl.state_space import (
    function_numpy_to_ompl,
    numpy_to_ompl_state,
    ompl_path_to_numpy,
    revolute_joints_state_space,
)
from cloth_tools.planning.interfaces import DualArmMotionPlanner
from ompl import base as ob
from ompl import geometric as og

DualJointConfigurationCheckerType = Callable[[np.ndarray], bool]


class DualArmOmplPlanner(DualArmMotionPlanner):
    def __init__(
        self,
        is_state_valid_fn: DualJointConfigurationCheckerType,
        max_planning_time: float = 30.0,
        num_interpolated_states: Optional[int] = 100,
    ):
        self.is_state_valid_fn = is_state_valid_fn

        # Settings
        self.max_planning_time = max_planning_time
        self.num_interpolated_states = num_interpolated_states

        # Currently we only support planning for two 6 DoF arms
        self.degrees_of_freedom: int = 12

    def _create_simple_setup(
        self,
        left_start_configuration: JointConfigurationType,
        right_start_configuration: JointConfigurationType,
        left_goal_configuration: JointConfigurationType,
        right_goal_configuration: JointConfigurationType,
    ):
        space = revolute_joints_state_space(self.degrees_of_freedom)
        start_configuration = np.concatenate([left_start_configuration, right_start_configuration])
        goal_configuration = np.concatenate([left_goal_configuration, right_goal_configuration])
        start_state = numpy_to_ompl_state(start_configuration, space)
        goal_state = numpy_to_ompl_state(goal_configuration, space)
        is_state_valid_ompl = function_numpy_to_ompl(self.is_state_valid_fn, self.degrees_of_freedom)

        simple_setup = og.SimpleSetup(space)
        simple_setup.setStateValidityChecker(ob.StateValidityCheckerFn(is_state_valid_ompl))
        simple_setup.setStartAndGoalStates(start_state, goal_state)

        # TODO: investigate whether a different reolsution is needed for dual arm planning as opposed to single arm planning
        step = float(np.deg2rad(5))
        resolution = step / space.getMaximumExtent()
        simple_setup.getSpaceInformation().setStateValidityCheckingResolution(resolution)

        planner = og.RRTstar(simple_setup.getSpaceInformation())
        simple_setup.setPlanner(planner)

        return simple_setup

    def _plan_to_joint_configuration_dual_arm(
        self,
        left_start_configuration: JointConfigurationType,
        right_start_configuration: JointConfigurationType,
        left_goal_configuration: JointConfigurationType,
        right_goal_configuration: JointConfigurationType,
    ) -> List[Tuple[JointConfigurationType, JointConfigurationType]] | None:
        simple_setup = self._create_simple_setup(
            left_start_configuration, right_start_configuration, left_goal_configuration, right_goal_configuration
        )

        # Everything below here in this functionis exactly the same as in
        # SingleArmOmplPlanner so we could consider refactoring this
        self._simple_setup = simple_setup  # Save for debugging

        simple_setup.solve(self.max_planning_time)

        if not simple_setup.haveExactSolutionPath():
            return None

        simple_setup.simplifySolution()
        path = simple_setup.getSolutionPath()
        if self.num_interpolated_states is not None:
            path.interpolate(self.num_interpolated_states)

        path_numpy = ompl_path_to_numpy(path, self.degrees_of_freedom)
        path_tuple = [(state[:6], state[6:]) for state in path_numpy]
        return path_tuple

    def _plan_to_joint_configuration_left_arm_only(
        self,
        left_start_configuration: JointConfigurationType,
        right_start_configuration: JointConfigurationType,
        left_goal_configuration: JointConfigurationType,
    ) -> List[Tuple[JointConfigurationType, JointConfigurationType]] | None:
        def is_left_state_valid_fn(left_state):
            return self.is_state_valid_fn(np.concatenate((left_state, right_start_configuration)))

        single_arm_planner = SingleArmOmplPlanner(
            is_left_state_valid_fn, self.max_planning_time, self.num_interpolated_states
        )
        self._single_arm_planner = single_arm_planner  # Save for debugging

        left_path = single_arm_planner.plan_to_joint_configuration(left_start_configuration, left_goal_configuration)
        path = [(left_state, right_start_configuration) for left_state in left_path]
        return path

    def _plan_to_joint_configuration_right_arm_only(
        self,
        left_start_configuration: JointConfigurationType,
        right_start_configuration: JointConfigurationType,
        right_goal_configuration: JointConfigurationType,
    ) -> List[Tuple[JointConfigurationType, JointConfigurationType]] | None:
        def is_right_state_valid_fn(right_state):
            return self.is_state_valid_fn(np.concatenate((left_start_configuration, right_state)))

        single_arm_planner = SingleArmOmplPlanner(
            is_right_state_valid_fn, self.max_planning_time, self.num_interpolated_states
        )
        self._single_arm_planner = single_arm_planner  # Save for debugging

        right_path = single_arm_planner.plan_to_joint_configuration(
            right_start_configuration, right_goal_configuration
        )
        path = [(left_start_configuration, right_state) for right_state in right_path]
        return path

    def plan_to_joint_configuration(
        self,
        left_start_configuration: JointConfigurationType,
        right_start_configuration: JointConfigurationType,
        left_goal_configuration: JointConfigurationType | None,
        right_goal_configuration: JointConfigurationType | None,
    ) -> List[Tuple[JointConfigurationType, JointConfigurationType]] | None:
        if left_goal_configuration is None and right_goal_configuration is None:
            raise ValueError("A goal configurations must be specified for at least one of the arms.")

        if right_goal_configuration is None:
            path = self._plan_to_joint_configuration_left_arm_only(
                left_start_configuration, right_start_configuration, left_goal_configuration
            )
            return path

        if left_goal_configuration is None:
            # Plan for the right arm only
            path = self._plan_to_joint_configuration_right_arm_only(
                left_start_configuration, right_start_configuration, right_goal_configuration
            )
            return path

        # Do 12 DoF dual arm planning
        path = self._plan_to_joint_configuration_dual_arm(
            left_start_configuration, right_start_configuration, left_goal_configuration, right_goal_configuration
        )
        return path

    def plan_to_tcp_pose(
        self,
        left_start_configuration: JointConfigurationType,
        right_start_configuration: JointConfigurationType,
        left_tcp_pose_in_base: HomogeneousMatrixType | None,
        right_tcp_pose_in_base: HomogeneousMatrixType | None,
    ) -> List[JointConfigurationType]:
        raise NotImplementedError("Dual arm planning to TCP poses is not yet implemented.")
