from typing import Callable, List, Optional, Tuple

import numpy as np
from airo_typing import HomogeneousMatrixType, JointConfigurationType
from cloth_tools.ompl.single_arm_planner import InverseKinematicsType, SingleArmOmplPlanner
from cloth_tools.ompl.state_space import (
    function_numpy_to_ompl,
    numpy_to_ompl_state,
    ompl_path_to_numpy,
    revolute_joints_state_space,
)
from cloth_tools.planning.interfaces import DualArmMotionPlanner
from loguru import logger
from ompl import base as ob
from ompl import geometric as og

DualJointConfigurationCheckerType = Callable[[np.ndarray], bool]


class DualArmOmplPlanner(DualArmMotionPlanner):
    def __init__(
        self,
        is_state_valid_fn: DualJointConfigurationCheckerType,
        left_inverse_kinematics_fn: Optional[InverseKinematicsType] = None,
        right_inverse_kinematics_fn: Optional[InverseKinematicsType] = None,
        max_planning_time: float = 30.0,
        num_interpolated_states: Optional[int] = 500,
    ):
        self.is_state_valid_fn = is_state_valid_fn
        self.left_inverse_kinematics_fn = left_inverse_kinematics_fn
        self.right_inverse_kinematics_fn = right_inverse_kinematics_fn

        # Settings
        self.max_planning_time = max_planning_time
        self.num_interpolated_states = num_interpolated_states

        # Currently we only support planning for two 6 DoF arms
        self.degrees_of_freedom: int = 12

        # The OMPL SimpleSetup for dual arm planning
        self._simple_setup = self._create_simple_setup_dual_arm()

        # Let SingleArmOmplPlanner handle planning for a single arm requests
        # Note that we (re)create these when start and goal config are set
        self._single_arm_planner_left: SingleArmOmplPlanner | None = None
        self._single_arm_planner_right: SingleArmOmplPlanner | None = None

        self._path_length_dual: float | None = None

    def _create_simple_setup_dual_arm(
        self,
    ):
        space = revolute_joints_state_space(self.degrees_of_freedom)

        is_state_valid_ompl = function_numpy_to_ompl(self.is_state_valid_fn, self.degrees_of_freedom)

        simple_setup = og.SimpleSetup(space)
        simple_setup.setStateValidityChecker(ob.StateValidityCheckerFn(is_state_valid_ompl))

        # TODO: investigate whether a different reolsution is needed for dual arm planning as opposed to single arm planning
        step = float(np.deg2rad(5))
        resolution = step / space.getMaximumExtent()
        simple_setup.getSpaceInformation().setStateValidityCheckingResolution(resolution)

        planner = og.RRTConnect(simple_setup.getSpaceInformation())
        simple_setup.setPlanner(planner)

        return simple_setup

    def _set_start_and_goal_configurations(
        self,
        left_start_configuration: JointConfigurationType,
        right_start_configuration: JointConfigurationType,
        left_goal_configuration: JointConfigurationType,
        right_goal_configuration: JointConfigurationType,
    ):
        # Set the starts and goals for dual arm planning
        space = self._simple_setup.getStateSpace()
        start_configuration = np.concatenate([left_start_configuration, right_start_configuration])
        goal_configuration = np.concatenate([left_goal_configuration, right_goal_configuration])
        start_state = numpy_to_ompl_state(start_configuration, space)
        goal_state = numpy_to_ompl_state(goal_configuration, space)
        self._simple_setup.setStartAndGoalStates(start_state, goal_state)

        # Replace single arm planners for the left and right arm
        def is_left_state_valid_fn(left_state):
            return self.is_state_valid_fn(np.concatenate((left_state, right_start_configuration)))

        def is_right_state_valid_fn(right_state):
            return self.is_state_valid_fn(np.concatenate((left_start_configuration, right_state)))

        self._single_arm_planner_left = SingleArmOmplPlanner(
            is_left_state_valid_fn,
            self.left_inverse_kinematics_fn,
            self.max_planning_time,
            self.num_interpolated_states,
        )

        self._single_arm_planner_right = SingleArmOmplPlanner(
            is_right_state_valid_fn,
            self.right_inverse_kinematics_fn,
            self.max_planning_time,
            self.num_interpolated_states,
        )

        self._single_arm_planner_left._set_start_and_goal_configurations(
            left_start_configuration, left_goal_configuration
        )

        self._single_arm_planner_right._set_start_and_goal_configurations(
            right_start_configuration, right_goal_configuration
        )

    def _plan_to_joint_configuration_dual_arm(
        self,
        left_start_configuration: JointConfigurationType,
        right_start_configuration: JointConfigurationType,
        left_goal_configuration: JointConfigurationType,
        right_goal_configuration: JointConfigurationType,
    ) -> List[Tuple[JointConfigurationType, JointConfigurationType]] | None:
        self._simple_setup.clear()
        self._set_start_and_goal_configurations(
            left_start_configuration, right_start_configuration, left_goal_configuration, right_goal_configuration
        )

        simple_setup = self._simple_setup

        simple_setup.solve(self.max_planning_time)

        if not simple_setup.haveExactSolutionPath():
            return None

        # Simplify, smooth and interpolate the solution path
        simple_setup.simplifySolution()
        path_simplifier = og.PathSimplifier(simple_setup.getSpaceInformation())
        path = simple_setup.getSolutionPath()
        path_simplifier.smoothBSpline(path)
        simple_setup.simplifySolution()
        if self.num_interpolated_states is not None:
            path.interpolate(self.num_interpolated_states)

        self._path_length_dual = path.length()

        path_numpy = ompl_path_to_numpy(path, self.degrees_of_freedom)
        path_tuple = [(state[:6], state[6:]) for state in path_numpy]
        return path_tuple

    def _plan_to_joint_configuration_left_arm_only(
        self,
        left_start_configuration: JointConfigurationType,
        right_start_configuration: JointConfigurationType,
        left_goal_configuration: JointConfigurationType,
    ) -> List[Tuple[JointConfigurationType, JointConfigurationType]] | None:
        # Set right goal to right start configuration
        self._set_start_and_goal_configurations(
            left_start_configuration, right_start_configuration, left_goal_configuration, right_start_configuration
        )

        left_path = self._single_arm_planner_left.plan_to_joint_configuration(
            left_start_configuration, left_goal_configuration
        )

        if left_path is None:
            return None

        path = [(left_state, right_start_configuration) for left_state in left_path]
        return path

    def _plan_to_joint_configuration_right_arm_only(
        self,
        left_start_configuration: JointConfigurationType,
        right_start_configuration: JointConfigurationType,
        right_goal_configuration: JointConfigurationType,
    ) -> List[Tuple[JointConfigurationType, JointConfigurationType]] | None:
        # Set left goal to left start configuration
        self._set_start_and_goal_configurations(
            left_start_configuration, right_start_configuration, left_start_configuration, right_goal_configuration
        )

        right_path = self._single_arm_planner_right.plan_to_joint_configuration(
            right_start_configuration, right_goal_configuration
        )

        if right_path is None:
            return None

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

    def _plan_to_tcp_pose_left_arm_only(
        self,
        left_start_configuration: JointConfigurationType,
        right_start_configuration: JointConfigurationType,
        left_tcp_pose_in_base: HomogeneousMatrixType,
    ) -> List[Tuple[JointConfigurationType, JointConfigurationType]] | None:
        # Set right goal to right start configuration
        self._set_start_and_goal_configurations(
            left_start_configuration, right_start_configuration, left_start_configuration, right_start_configuration
        )

        left_path = self._single_arm_planner_left.plan_to_tcp_pose(left_start_configuration, left_tcp_pose_in_base)

        if left_path is None:
            return None

        path = [(left_state, right_start_configuration) for left_state in left_path]
        return path

    def _plan_to_tcp_pose_right_arm_only(
        self,
        left_start_configuration: JointConfigurationType,
        right_start_configuration: JointConfigurationType,
        right_tcp_pose_in_base: HomogeneousMatrixType,
    ) -> List[Tuple[JointConfigurationType, JointConfigurationType]] | None:
        # Set left goal to left start configuration
        self._set_start_and_goal_configurations(
            left_start_configuration, right_start_configuration, left_start_configuration, right_start_configuration
        )

        right_path = self._single_arm_planner_right.plan_to_tcp_pose(right_start_configuration, right_tcp_pose_in_base)

        if right_path is None:
            return None

        path = [(left_start_configuration, right_state) for right_state in right_path]
        return path

    def _plan_to_tcp_pose_dual_arm(
        self,
        left_start_configuration: JointConfigurationType,
        right_start_configuration: JointConfigurationType,
        left_tcp_pose_in_base: HomogeneousMatrixType,
        right_tcp_pose_in_base: HomogeneousMatrixType,
    ) -> List[Tuple[JointConfigurationType, JointConfigurationType]] | None:
        # TODO
        # 1. .do IK for both arms
        # 2. create all goal pairs
        # 3. filter out invalid goal pairs
        # 4. for each pair, plan to the goal pair
        # 5. return the shortest path among all the planned paths

        if self.left_inverse_kinematics_fn is None or self.right_inverse_kinematics_fn is None:
            logger.info(
                "Planning to left and right TCP poses attempted but inverse_kinematics_fn was not provided for both arms, returing None."
            )

        # 1. do IK for both arms
        left_ik_solutions = self.left_inverse_kinematics_fn(left_tcp_pose_in_base)
        right_ik_solutions = self.right_inverse_kinematics_fn(right_tcp_pose_in_base)

        if left_ik_solutions is None or len(left_ik_solutions) == 0:
            logger.info("IK for left arm returned no solutions, returning None.")
            return None
        else:
            logger.info(f"Found {len(left_ik_solutions)} IK solutions for left arm.")

        if right_ik_solutions is None or len(right_ik_solutions) == 0:
            logger.info("IK for right arm returned no solutions, returning None.")
            return None
        else:
            logger.info(f"Found {len(right_ik_solutions)} IK solutions for right arm.")

        # 2. create all goal pairs
        goal_configurations = []
        for left_ik_solution in left_ik_solutions:
            for right_ik_solution in right_ik_solutions:
                goal_configurations.append(np.concatenate((left_ik_solution, right_ik_solution)))

        n_goal_configurations = len(goal_configurations)

        # 3. filter out invalid goal pairs
        goal_configurations_valid = [s for s in goal_configurations if self.is_state_valid_fn(s)]
        n_valid_goal_configurations = len(goal_configurations_valid)

        if n_valid_goal_configurations == 0:
            logger.info(f"All {n_goal_configurations} goal pairs are invalid, returning None.")
            return None
        else:
            logger.info(f"Found {n_valid_goal_configurations}/{n_goal_configurations} valid goal pairs.")

        # 4. for each pair, plan to the goal pair
        paths = []
        path_lengths = []
        for goal_configuration in goal_configurations_valid:
            path = self.plan_to_joint_configuration(
                left_start_configuration, right_start_configuration, goal_configuration[:6], goal_configuration[6:]
            )
            if path is not None:
                paths.append(path)
                path_lengths.append(self._path_length_dual)

        if len(paths) == 0:
            logger.info("No paths founds towards any goal pairs, returning None.")
            return None

        logger.info(f"Found {len(paths)} paths towards goal pairs.")

        # 5. return the shortest path among all the planned paths
        shortest_path_idx = np.argmin(path_lengths)
        shortest_path = paths[shortest_path_idx]
        return shortest_path

        # 2. filter out invalid IK solutions
        # left_ik_solutions_valid = [s for s in left_ik_solutions if self.is_state_valid_fn(s)]
        # right_ik_solutions_valid = [s for s in right_ik_solutions if self.is_state_valid_fn(s)]

        # Set left goal to left start configuration
        # self._set_start_and_goal_configurations(
        #     left_start_configuration, right_start_configuration, left_start_configuration, right_start_configuration
        # )

        # path = self._simple_setup.solve(self.max_planning_time)

        # if not self._simple_setup.haveExactSolutionPath():
        #     return None

        # # Simplify, smooth and interpolate the solution path
        # self._simple_setup.simplifySolution()
        # path_simplifier = og.PathSimplifier(self._simple_setup.getSpaceInformation())
        # path = self._simple_setup.getSolutionPath()
        # path_simplifier.smoothBSpline(path)
        # self._simple_setup.simplifySolution()
        # if self.num_interpolated_states is not None:
        #     path.interpolate(self.num_interpolated_states)

        # path_numpy = ompl_path_to_numpy(path, self.degrees_of_freedom)
        # path_tuple = [(state[:6], state[6:]) for state in path_numpy]
        # return path_tuple

    def plan_to_tcp_pose(
        self,
        left_start_configuration: JointConfigurationType,
        right_start_configuration: JointConfigurationType,
        left_tcp_pose_in_base: HomogeneousMatrixType | None,
        right_tcp_pose_in_base: HomogeneousMatrixType | None,
    ) -> List[JointConfigurationType]:
        if left_tcp_pose_in_base is None and right_tcp_pose_in_base is None:
            raise ValueError("A goal TCP pose must be specified for at least one of the arms.")

        if right_tcp_pose_in_base is None:
            path = self._plan_to_tcp_pose_left_arm_only(
                left_start_configuration, right_start_configuration, left_tcp_pose_in_base
            )
            return path

        if left_tcp_pose_in_base is None:
            path = self._plan_to_tcp_pose_right_arm_only(
                left_start_configuration, right_start_configuration, right_tcp_pose_in_base
            )
            return path

        path = self._plan_to_tcp_pose_dual_arm(
            left_start_configuration, right_start_configuration, left_tcp_pose_in_base, right_tcp_pose_in_base
        )

        return path
