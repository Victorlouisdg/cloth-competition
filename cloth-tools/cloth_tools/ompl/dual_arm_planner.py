from typing import Callable, List, Optional, Tuple

import numpy as np
from airo_typing import HomogeneousMatrixType, JointConfigurationType
from cloth_tools.ompl.single_arm_planner import InverseKinematicsType, SingleArmOmplPlanner
from cloth_tools.ompl.state_space import function_numpy_to_ompl, numpy_to_ompl_state, ompl_path_to_numpy
from cloth_tools.planning.interfaces import DualArmMotionPlanner
from loguru import logger
from ompl import base as ob
from ompl import geometric as og

DualJointConfigurationCheckerType = Callable[[np.ndarray], bool]


class DualArmOmplPlanner(DualArmMotionPlanner):
    def __init__(
        self,
        is_state_valid_fn: DualJointConfigurationCheckerType,
        inverse_kinematics_left_fn: Optional[InverseKinematicsType] = None,
        inverse_kinematics_right_fn: Optional[InverseKinematicsType] = None,
        joint_bounds_left: Optional[Tuple[JointConfigurationType, JointConfigurationType]] = None,
        joint_bounds_right: Optional[Tuple[JointConfigurationType, JointConfigurationType]] = None,
        max_planning_time: float = 5.0,
        num_interpolated_states: Optional[int] = 500,
    ):
        self.is_state_valid_fn = is_state_valid_fn
        self.inverse_kinematics_left_fn = inverse_kinematics_left_fn
        self.inverse_kinematics_right_fn = inverse_kinematics_right_fn
        self.joint_bounds_left = joint_bounds_left
        self.joint_bounds_right = joint_bounds_right

        # Currently we only support planning for two 6 DoF arms
        self.degrees_of_freedom: int = 12

        if self.joint_bounds_left is None:
            self.joint_bounds_left = (np.full(6, -2 * np.pi), np.full(6, 2 * np.pi))

        if self.joint_bounds_right is None:
            self.joint_bounds_right = (np.full(6, -2 * np.pi), np.full(6, 2 * np.pi))

        # Settings
        self.max_planning_time = max_planning_time
        self.num_interpolated_states = num_interpolated_states

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
        # Make state space
        space = ob.RealVectorStateSpace(self.degrees_of_freedom)
        bounds = ob.RealVectorBounds(self.degrees_of_freedom)
        for i in range(self.degrees_of_freedom):
            if i < 6:
                bounds.setLow(i, self.joint_bounds_left[0][i])
                bounds.setHigh(i, self.joint_bounds_left[1][i])
            else:
                bounds.setLow(i, self.joint_bounds_right[0][i - 6])
                bounds.setHigh(i, self.joint_bounds_right[1][i - 6])
        space.setBounds(bounds)

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
        start_configuration_left: JointConfigurationType,
        start_configuration_right: JointConfigurationType,
        goal_configuration_left: JointConfigurationType,
        goal_configuration_right: JointConfigurationType,
    ):
        # Set the starts and goals for dual arm planning
        space = self._simple_setup.getStateSpace()
        start_configuration = np.concatenate([start_configuration_left, start_configuration_right])
        goal_configuration = np.concatenate([goal_configuration_left, goal_configuration_right])
        start_state = numpy_to_ompl_state(start_configuration, space)
        goal_state = numpy_to_ompl_state(goal_configuration, space)
        self._simple_setup.setStartAndGoalStates(start_state, goal_state)

        # Replace single arm planners for the left and right arm
        def is_left_state_valid_fn(left_state):
            return self.is_state_valid_fn(np.concatenate((left_state, start_configuration_right)))

        def is_right_state_valid_fn(right_state):
            return self.is_state_valid_fn(np.concatenate((start_configuration_left, right_state)))

        self._single_arm_planner_left = SingleArmOmplPlanner(
            is_left_state_valid_fn,
            self.inverse_kinematics_left_fn,
            self.joint_bounds_left,
            self.max_planning_time,
            self.num_interpolated_states,
        )

        self._single_arm_planner_right = SingleArmOmplPlanner(
            is_right_state_valid_fn,
            self.inverse_kinematics_right_fn,
            self.joint_bounds_right,
            self.max_planning_time,
            self.num_interpolated_states,
        )

        self._single_arm_planner_left._set_start_and_goal_configurations(
            start_configuration_left, goal_configuration_left
        )

        self._single_arm_planner_right._set_start_and_goal_configurations(
            start_configuration_right, goal_configuration_right
        )

    def _plan_to_joint_configuration_dual_arm(
        self,
        start_configuration_left: JointConfigurationType,
        start_configuration_right: JointConfigurationType,
        goal_configuration_left: JointConfigurationType,
        goal_configuration_right: JointConfigurationType,
    ) -> List[Tuple[JointConfigurationType, JointConfigurationType]] | None:
        self._simple_setup.clear()
        self._set_start_and_goal_configurations(
            start_configuration_left, start_configuration_right, goal_configuration_left, goal_configuration_right
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

        # Don't simplify again, seems to make joint velocity jumps worse
        # simple_setup.simplifySolution()
        # if self.num_interpolated_states is not None:
        #     path.interpolate(self.num_interpolated_states)

        self._path_length_dual = path.length()

        path_numpy = ompl_path_to_numpy(path, self.degrees_of_freedom)
        path_tuple = [(state[:6], state[6:]) for state in path_numpy]
        return path_tuple

    def _plan_to_joint_configuration_left_arm_only(
        self,
        start_configuration_left: JointConfigurationType,
        start_configuration_right: JointConfigurationType,
        goal_configuration_left: JointConfigurationType,
    ) -> List[Tuple[JointConfigurationType, JointConfigurationType]] | None:
        # Set right goal to right start configuration
        self._set_start_and_goal_configurations(
            start_configuration_left, start_configuration_right, goal_configuration_left, start_configuration_right
        )

        left_path = self._single_arm_planner_left.plan_to_joint_configuration(
            start_configuration_left, goal_configuration_left
        )

        if left_path is None:
            return None

        path = [(left_state, start_configuration_right) for left_state in left_path]
        return path

    def _plan_to_joint_configuration_right_arm_only(
        self,
        start_configuration_left: JointConfigurationType,
        start_configuration_right: JointConfigurationType,
        goal_configuration_right: JointConfigurationType,
    ) -> List[Tuple[JointConfigurationType, JointConfigurationType]] | None:
        # Set left goal to left start configuration
        self._set_start_and_goal_configurations(
            start_configuration_left, start_configuration_right, start_configuration_left, goal_configuration_right
        )

        right_path = self._single_arm_planner_right.plan_to_joint_configuration(
            start_configuration_right, goal_configuration_right
        )

        if right_path is None:
            return None

        path = [(start_configuration_left, right_state) for right_state in right_path]
        return path

    def plan_to_joint_configuration(
        self,
        start_configuration_left: JointConfigurationType,
        start_configuration_right: JointConfigurationType,
        goal_configuration_left: JointConfigurationType | None,
        goal_configuration_right: JointConfigurationType | None,
    ) -> List[Tuple[JointConfigurationType, JointConfigurationType]] | None:
        if goal_configuration_left is None and goal_configuration_right is None:
            raise ValueError("A goal configurations must be specified for at least one of the arms.")

        if goal_configuration_right is None:
            path = self._plan_to_joint_configuration_left_arm_only(
                start_configuration_left, start_configuration_right, goal_configuration_left
            )
            return path

        if goal_configuration_left is None:
            # Plan for the right arm only
            path = self._plan_to_joint_configuration_right_arm_only(
                start_configuration_left, start_configuration_right, goal_configuration_right
            )
            return path

        # Do 12 DoF dual arm planning
        path = self._plan_to_joint_configuration_dual_arm(
            start_configuration_left, start_configuration_right, goal_configuration_left, goal_configuration_right
        )
        return path

    def _plan_to_tcp_pose_left_arm_only(
        self,
        start_configuration_left: JointConfigurationType,
        start_configuration_right: JointConfigurationType,
        tcp_pose_left_in_base: HomogeneousMatrixType,
        desirable_goal_configurations_left: List[JointConfigurationType] | None = None,
    ) -> List[Tuple[JointConfigurationType, JointConfigurationType]] | None:
        # Set right goal to right start configuration
        self._set_start_and_goal_configurations(
            start_configuration_left, start_configuration_right, start_configuration_left, start_configuration_right
        )

        left_path = self._single_arm_planner_left.plan_to_tcp_pose(
            start_configuration_left, tcp_pose_left_in_base, desirable_goal_configurations_left
        )

        if left_path is None:
            return None

        path = [(left_state, start_configuration_right) for left_state in left_path]
        return path

    def _plan_to_tcp_pose_right_arm_only(
        self,
        start_configuration_left: JointConfigurationType,
        start_configuration_right: JointConfigurationType,
        tcp_pose_right_in_base: HomogeneousMatrixType,
        desirable_goal_configurations_right: List[JointConfigurationType] | None = None,
    ) -> List[Tuple[JointConfigurationType, JointConfigurationType]] | None:
        # Set left goal to left start configuration
        self._set_start_and_goal_configurations(
            start_configuration_left, start_configuration_right, start_configuration_left, start_configuration_right
        )

        right_path = self._single_arm_planner_right.plan_to_tcp_pose(
            start_configuration_right, tcp_pose_right_in_base, desirable_goal_configurations_right
        )

        if right_path is None:
            return None

        path = [(start_configuration_left, right_state) for right_state in right_path]
        return path

    def _plan_to_tcp_pose_dual_arm(  # noqa: C901
        self,
        start_configuration_left: JointConfigurationType,
        start_configuration_right: JointConfigurationType,
        tcp_pose_left_in_base: HomogeneousMatrixType,
        tcp_pose_right_in_base: HomogeneousMatrixType,
    ) -> List[Tuple[JointConfigurationType, JointConfigurationType]] | None:
        if self.inverse_kinematics_left_fn is None or self.inverse_kinematics_right_fn is None:
            logger.info(
                "Planning to left and right TCP poses attempted but inverse_kinematics_fn was not provided for both arms, returing None."
            )

        # 1. do IK for both arms
        ik_solutions_left = self.inverse_kinematics_left_fn(tcp_pose_left_in_base)
        ik_solutions_right = self.inverse_kinematics_right_fn(tcp_pose_right_in_base)

        if ik_solutions_left is None or len(ik_solutions_left) == 0:
            logger.info("IK for left arm returned no solutions, returning None.")
            return None
        else:
            logger.info(f"Found {len(ik_solutions_left)} IK solutions for left arm.")

        if ik_solutions_right is None or len(ik_solutions_right) == 0:
            logger.info("IK for right arm returned no solutions, returning None.")
            return None
        else:
            logger.info(f"Found {len(ik_solutions_right)} IK solutions for right arm.")

        # 2. filter out IK solutions that are outside the joint bounds
        ik_solutions_in_bounds_left = []
        for ik_solution in ik_solutions_left:
            if np.all(ik_solution >= self.joint_bounds_left[0]) and np.all(ik_solution <= self.joint_bounds_left[1]):
                ik_solutions_in_bounds_left.append(ik_solution)

        ik_solutions_in_bounds_right = []
        for ik_solution in ik_solutions_right:
            if np.all(ik_solution >= self.joint_bounds_right[0]) and np.all(ik_solution <= self.joint_bounds_right[1]):
                ik_solutions_in_bounds_right.append(ik_solution)

        if len(ik_solutions_in_bounds_left) == 0:
            logger.info("No IK solutions for left arm are within the joint bounds, returning None.")
            return None
        else:
            logger.info(
                f"Found {len(ik_solutions_in_bounds_left)}/{len(ik_solutions_left)} IK solutions within the joint bounds for left arm."
            )

        if len(ik_solutions_in_bounds_right) == 0:
            logger.info("No IK solutions for right arm are within the joint bounds, returning None.")
            return None
        else:
            logger.info(
                f"Found {len(ik_solutions_in_bounds_right)}/{len(ik_solutions_right)} IK solutions within the joint bounds for right arm."
            )

        # 2. create all goal pairs
        goal_configurations = []
        for ik_solution_left in ik_solutions_in_bounds_left:
            for ik_solution_right in ik_solutions_in_bounds_right:
                goal_configurations.append(np.concatenate((ik_solution_left, ik_solution_right)))

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
                start_configuration_left, start_configuration_right, goal_configuration[:6], goal_configuration[6:]
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

    def plan_to_tcp_pose(
        self,
        start_configuration_left: JointConfigurationType,
        start_configuration_right: JointConfigurationType,
        tcp_pose_left_in_base: HomogeneousMatrixType | None,
        tcp_pose_right_in_base: HomogeneousMatrixType | None,
        desirable_goal_configurations_left: List[JointConfigurationType] | None = None,
        desirable_goal_configurations_right: List[JointConfigurationType] | None = None,
    ) -> List[JointConfigurationType]:
        if tcp_pose_left_in_base is None and tcp_pose_right_in_base is None:
            raise ValueError("A goal TCP pose must be specified for at least one of the arms.")

        if tcp_pose_right_in_base is None:
            path = self._plan_to_tcp_pose_left_arm_only(
                start_configuration_left,
                start_configuration_right,
                tcp_pose_left_in_base,
                desirable_goal_configurations_left,
            )
            return path

        if tcp_pose_left_in_base is None:
            path = self._plan_to_tcp_pose_right_arm_only(
                start_configuration_left,
                start_configuration_right,
                tcp_pose_right_in_base,
                desirable_goal_configurations_right,
            )
            return path

        # TODO use desirable_goal_configurations for dual arm planning
        if desirable_goal_configurations_left is not None or desirable_goal_configurations_right is not None:
            logger.warning(
                "Desirable goal configurations are not implemented yet for dual arm planning, ignoring them."
            )

        path = self._plan_to_tcp_pose_dual_arm(
            start_configuration_left, start_configuration_right, tcp_pose_left_in_base, tcp_pose_right_in_base
        )

        return path
