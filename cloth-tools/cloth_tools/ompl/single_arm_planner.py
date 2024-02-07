from typing import Callable, List, Optional, Tuple

import numpy as np
from airo_typing import HomogeneousMatrixType, JointConfigurationType
from cloth_tools.ompl.state_space import function_numpy_to_ompl, numpy_to_ompl_state, ompl_path_to_numpy
from cloth_tools.planning.interfaces import SingleArmMotionPlanner
from loguru import logger
from ompl import base as ob
from ompl import geometric as og

JointConfigurationCheckerType = Callable[[JointConfigurationType], bool]
InverseKinematicsType = Callable[[HomogeneousMatrixType], List[JointConfigurationType]]


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
        joint_bounds: Tuple[JointConfigurationType, JointConfigurationType] = None,
        max_planning_time: float = 5.0,
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
        self.joint_bounds = joint_bounds

        # Currently we only planning for 6 DoF arms
        self.degrees_of_freedom: int = 6

        if self.joint_bounds is None:
            self.joint_bounds = (np.full(6, -2 * np.pi), np.full(6, 2 * np.pi))

        # Settings
        self.max_planning_time = max_planning_time
        self.num_interpolated_states = num_interpolated_states

        self._simple_setup = self._create_simple_setup()

        self._path_length: float | None = None

    def _create_simple_setup(self):
        # Create state space and convert to OMPL compatible data types
        joint_bounds = self.joint_bounds
        space = ob.RealVectorStateSpace(self.degrees_of_freedom)
        bounds = ob.RealVectorBounds(self.degrees_of_freedom)
        joint_bounds_lower = joint_bounds[0]
        joint_bounds_upper = joint_bounds[1]
        for i in range(self.degrees_of_freedom):
            bounds.setLow(i, joint_bounds_lower[i])
            bounds.setHigh(i, joint_bounds_upper[i])
        space.setBounds(bounds)

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
    ) -> List[JointConfigurationType] | None:
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
        # Don't simplify again, seems to make joint velocity jumps worse
        # simple_setup.simplifySolution()
        # if self.num_interpolated_states is not None:
        #     path.interpolate(self.num_interpolated_states)

        self._path_length = path.length()

        path_numpy = ompl_path_to_numpy(path, self.degrees_of_freedom)
        return path_numpy

    def plan_to_tcp_pose(  # noqa: C901
        self,
        start_configuration: JointConfigurationType,
        tcp_pose_in_base: HomogeneousMatrixType,
        desirable_goal_configurations: Optional[List[JointConfigurationType]] = None,
    ) -> List[JointConfigurationType] | None:
        # TODO: add options for specifying a preferred IK solutions, e.g. min distance to a joint configuration
        # desirable_goal_joint_configurations = Optional[List[JointConfigurationType]]
        # Without this we plan to all joint configs and pick the shortest path
        # With it, we try the closest IK solution first and if it fails we try the next closest etc.
        if self.inverse_kinematics_fn is None:
            logger.warning("Planning to TCP pose attempted but inverse_kinematics_fn was provided, returing None.")
            return None

        ik_solutions = self.inverse_kinematics_fn(tcp_pose_in_base)
        if ik_solutions is None or len(ik_solutions) == 0:
            logger.info("IK returned no solutions, returning None.")
            return None
        else:
            logger.info(f"IK returned {len(ik_solutions)} solutions.")

        ik_solutions_within_bounds = []
        for ik_solution in ik_solutions:
            if np.all(ik_solution >= self.joint_bounds[0]) and np.all(ik_solution <= self.joint_bounds[1]):
                ik_solutions_within_bounds.append(ik_solution)

        if len(ik_solutions_within_bounds) == 0:
            logger.info("No IK solutions are within the joint bounds, returning None.")
            return None
        else:
            logger.info(f"Found {len(ik_solutions_within_bounds)}/{len(ik_solutions)} solutions within joint bounds.")

        ik_solutions_valid = [s for s in ik_solutions_within_bounds if self.is_state_valid_fn(s)]
        if len(ik_solutions_valid) == 0:
            logger.info("All IK solutions within bounds are invalid, returning None.")
            return None
        else:
            logger.info(f"Found {len(ik_solutions_valid)}/{len(ik_solutions_within_bounds)} valid solutions.")

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

        path_distances = [np.linalg.norm(path[-1] - start_configuration) for path in paths]

        path_desirablities = None
        if desirable_goal_configurations is not None:
            path_desirablities = []
            for path in paths:
                min_distance = np.inf
                for desirable_goal in desirable_goal_configurations:
                    distance = np.linalg.norm(path[-1] - desirable_goal)
                    min_distance = min(min_distance, distance)
                path_desirablities.append(min_distance)

        lengths_str = f"{[np.round(l, 3) for l in path_lengths]}"
        distances_str = f"{[np.round(d, 3) for d in path_distances]}"
        logger.info(f"Found {len(paths)} paths towards IK solutions:")
        logger.info(f"Path lengths: {lengths_str}")
        logger.info(f"Path distances: {distances_str}")

        if path_desirablities is not None:
            desirabilities_str = f"{[np.round(d, 3) for d in path_desirablities]}"
            logger.info(f"Path desirabilities: {desirabilities_str}")

        use_desirability = path_desirablities is not None

        if use_desirability:
            idx = np.argmin(path_desirablities)
            logger.info(
                f"Length of chosen solution (= most desirable end): {path_lengths[idx]:.3f}, desirability: {path_desirablities[idx]:.3f}"
            )
        else:
            idx = np.argmin(path_lengths)
            logger.info(f"Length of chosen solution (= shortest path): {path_lengths[idx]:.3f}")

        solution_path = paths[idx]
        return solution_path
