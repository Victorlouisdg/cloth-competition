from functools import partial

import numpy as np
from airo_drake import (
    calculate_valid_joint_paths,
    concatenate_drake_trajectories,
    discretize_drake_pose_trajectory,
    time_parametrize_toppra,
)
from airo_planner import DualArmOmplPlanner, PlannerError, filter_with_distance_to_configurations, stack_joints
from airo_typing import (
    HomogeneousMatrixType,
    InverseKinematicsFunctionType,
    JointConfigurationCheckerType,
    JointConfigurationType,
    PosePathType,
    Vector3DType,
)
from loguru import logger
from pydrake.math import RigidTransform, RotationMatrix
from pydrake.multibody.plant import MultibodyPlant
from pydrake.trajectories import PiecewisePose, Trajectory


class ExhaustedOptionsError(RuntimeError):
    pass


class GraspNotFeasibleError(RuntimeError):
    pass


# TODO consider making a class for this that stores debug information
def plan_pregrasp_and_grasp_trajectory(  # noqa: C901
    planner_pregrasp: DualArmOmplPlanner,
    grasp_pose: HomogeneousMatrixType,
    start_configuration_left: JointConfigurationType,
    start_configuration_right: JointConfigurationType,
    inverse_kinematics_left_fn: InverseKinematicsFunctionType,  # same comment as for is_state_valid_fn_grasp
    inverse_kinematics_right_fn: InverseKinematicsFunctionType,
    is_state_valid_fn_grasp: JointConfigurationCheckerType,  # could make this optional and use planner's by default
    plant_toppra: MultibodyPlant,
    with_left: bool = True,
) -> Trajectory:

    # We add 1.0 so at least one pregrasp distance fails:
    # pregrasp_distances_to_try = [0.05, 0.1, 0.15]  # , 0.2, 0.25]
    distance_min = 0.05
    distance_max = 0.25
    step = 0.01
    steps = int(np.rint((distance_max - distance_min) / step)) + 1
    pregrasp_distances_to_try = np.linspace(0.05, 0.25, steps)

    # is_state_valid_fn_grasp currently still takes a 12-DoF configuration
    def is_single_arm_state_valid_fn_grasp(joint_configuration: JointConfigurationType) -> bool:
        if with_left:
            return is_state_valid_fn_grasp(stack_joints(joint_configuration, start_configuration_right))
        else:
            return is_state_valid_fn_grasp(stack_joints(start_configuration_left, joint_configuration))

    def hardcoded_cost_fn(
        joint_configuration: JointConfigurationType,
        known_joint_configurations: list[JointConfigurationType],
        costs: list[float],
    ) -> float:
        distances = [
            np.linalg.norm(joint_configuration - known_configuration)
            for known_configuration in known_joint_configurations
        ]
        if np.min(distances) > 0.001:
            logger.warning(f"Joint configuration is not close to any known configurations. {joint_configuration} ")
        return costs[np.argmin(distances)]

    def rank_with_cost_fn(
        joint_configurations: list[JointConfigurationType], cost_fn: JointConfigurationCheckerType
    ) -> list[JointConfigurationType]:
        return sorted(joint_configurations, key=cost_fn)

    for distance in pregrasp_distances_to_try:
        logger.info(f"Planning to pregrasp pose at distance {distance}.")
        # 1. Compute pregrasp pose
        pregrasp_pose = grasp_pose.copy()
        pregrasp_pose[0:3, 3] -= distance * pregrasp_pose[0:3, 2]

        pregrasp_pose_left = pregrasp_pose if with_left else None
        pregrasp_pose_right = pregrasp_pose if not with_left else None

        # 2. Compute grasp TCP path
        rigid_transforms = [RigidTransform(pose) for pose in [pregrasp_pose, grasp_pose]]
        times = np.linspace(0, 1, len(rigid_transforms))
        pose_trajectory = PiecewisePose.MakeLinear(times=times, poses=rigid_transforms)
        grasp_tcp_path = discretize_drake_pose_trajectory(pose_trajectory).poses

        # 3 Compute valid grasp joint paths
        inverse_kinematics_fn = inverse_kinematics_left_fn if with_left else inverse_kinematics_right_fn

        grasp_path_single_arm = calculate_valid_joint_paths(
            grasp_tcp_path, inverse_kinematics_fn, is_single_arm_state_valid_fn_grasp
        )

        if len(grasp_path_single_arm) == 0:
            logger.info(f"No valid grasp joint paths found for distance {distance}, continuing to next distance.")
            continue

        if with_left:
            grasp_paths = [stack_joints(path, start_configuration_right) for path in grasp_path_single_arm]
        else:
            grasp_paths = [stack_joints(start_configuration_left, path) for path in grasp_path_single_arm]

        # Create filter function
        grasp_path_starts = [path[0] for path in grasp_paths]
        filter_fn = partial(filter_with_distance_to_configurations, joint_configurations_close=grasp_path_starts)

        # Create rank function
        grasp_trajectories = []
        times = []
        for path in grasp_paths:
            trajectory = time_parametrize_toppra(plant_toppra, path)
            times.append(trajectory.end_time())
            grasp_trajectories.append(trajectory)

        cost_fn = partial(hardcoded_cost_fn, known_joint_configurations=grasp_path_starts, costs=times)
        rank_fn = partial(rank_with_cost_fn, cost_fn=cost_fn)

        # Plan
        planner_pregrasp.filter_goal_configurations_fn = filter_fn
        planner_pregrasp.rank_goal_configurations_fn = rank_fn

        try:
            pregrasp_path = planner_pregrasp.plan_to_tcp_pose(
                start_configuration_left, start_configuration_right, pregrasp_pose_left, pregrasp_pose_right
            )
        except PlannerError as e:
            logger.info(
                f"Failed to plan to pregrasp pose at distance {distance}, continuing to next distance. Exception was:\n {e}."
            )
            continue

        pregrasp_trajectory = time_parametrize_toppra(plant_toppra, pregrasp_path)

        # # Find the grasp trajectory of which the start is closest to the pregrasp path end (=pregrasp end joints)
        # We will likely want an airo-planner helper function for this
        pregrasp_end_joints = pregrasp_path[-1]
        distances = [np.linalg.norm(start - pregrasp_end_joints) for start in grasp_path_starts]
        index_of_closest_start = np.argmin(distances)

        assert np.isclose(distances[index_of_closest_start], 0, atol=0.01)  # sanity check

        grasp_trajectory = grasp_trajectories[index_of_closest_start]

        # final set: concatenate pregrasp and grasp trajectories
        pregrasp_and_grasp_trajectory = concatenate_drake_trajectories([pregrasp_trajectory, grasp_trajectory])

        return pregrasp_and_grasp_trajectory

    raise ExhaustedOptionsError("Grasp planner exhausted all pregrasp poses to try")


def plan_to_grasp_pose_path(  # noqa: C901
    planner_pregrasp: DualArmOmplPlanner,
    grasp_pose_path: PosePathType,
    start_configuration_left: JointConfigurationType,
    start_configuration_right: JointConfigurationType,
    inverse_kinematics_left_fn: InverseKinematicsFunctionType,  # same comment as for is_state_valid_fn_grasp
    inverse_kinematics_right_fn: InverseKinematicsFunctionType,
    is_state_valid_fn_grasp: JointConfigurationCheckerType,  # could make this optional and use planner's by default
    plant_toppra: MultibodyPlant,
    with_left: bool = True,
) -> tuple[Trajectory]:

    # is_state_valid_fn_grasp currently still takes a 12-DoF configuration
    def is_single_arm_state_valid_fn_grasp(joint_configuration: JointConfigurationType) -> bool:
        if with_left:
            return is_state_valid_fn_grasp(stack_joints(joint_configuration, start_configuration_right))
        else:
            return is_state_valid_fn_grasp(stack_joints(start_configuration_left, joint_configuration))

    def hardcoded_cost_fn(
        joint_configuration: JointConfigurationType,
        known_joint_configurations: list[JointConfigurationType],
        costs: list[float],
    ) -> float:
        distances = [
            np.linalg.norm(joint_configuration - known_configuration)
            for known_configuration in known_joint_configurations
        ]
        if np.min(distances) > 0.001:
            logger.warning(f"Joint configuration is not close to any known configurations. {joint_configuration} ")
        return costs[np.argmin(distances)]

    def rank_with_cost_fn(
        joint_configurations: list[JointConfigurationType], cost_fn: JointConfigurationCheckerType
    ) -> list[JointConfigurationType]:
        return sorted(joint_configurations, key=cost_fn)

    # 3 Compute valid grasp joint paths
    inverse_kinematics_fn = inverse_kinematics_left_fn if with_left else inverse_kinematics_right_fn

    grasp_path_single_arm = calculate_valid_joint_paths(
        grasp_pose_path, inverse_kinematics_fn, is_single_arm_state_valid_fn_grasp
    )

    if len(grasp_path_single_arm) == 0:
        raise GraspNotFeasibleError("No valid joint paths found that can execute grasp.")

    if with_left:
        grasp_paths = [stack_joints(path, start_configuration_right) for path in grasp_path_single_arm]
    else:
        grasp_paths = [stack_joints(start_configuration_left, path) for path in grasp_path_single_arm]

    # Create filter function
    grasp_path_starts = [path[0] for path in grasp_paths]
    filter_fn = partial(filter_with_distance_to_configurations, joint_configurations_close=grasp_path_starts)

    # Create rank function
    grasp_trajectories = []
    times = []
    for path in grasp_paths:
        trajectory = time_parametrize_toppra(plant_toppra, path)
        times.append(trajectory.end_time())
        grasp_trajectories.append(trajectory)

    cost_fn = partial(hardcoded_cost_fn, known_joint_configurations=grasp_path_starts, costs=times)
    rank_fn = partial(rank_with_cost_fn, cost_fn=cost_fn)

    # Plan
    planner_pregrasp.filter_goal_configurations_fn = filter_fn
    planner_pregrasp.rank_goal_configurations_fn = rank_fn

    pregrasp_pose = grasp_pose_path[0]
    pregrasp_pose_left = pregrasp_pose if with_left else None
    pregrasp_pose_right = pregrasp_pose if not with_left else None

    try:
        pregrasp_path = planner_pregrasp.plan_to_tcp_pose(
            start_configuration_left, start_configuration_right, pregrasp_pose_left, pregrasp_pose_right
        )
    except PlannerError:
        raise GraspNotFeasibleError()

    pregrasp_trajectory = time_parametrize_toppra(plant_toppra, pregrasp_path)

    # # Find the grasp trajectory of which the start is closest to the pregrasp path end (=pregrasp end joints)
    # We will likely want an airo-planner helper function for this
    pregrasp_end_joints = pregrasp_path[-1]
    distances = [np.linalg.norm(start - pregrasp_end_joints) for start in grasp_path_starts]
    index_of_closest_start = np.argmin(distances)

    assert np.isclose(distances[index_of_closest_start], 0, atol=0.01)  # sanity check

    grasp_trajectory = grasp_trajectories[index_of_closest_start]

    # final set: concatenate pregrasp and grasp trajectories
    pregrasp_and_grasp_trajectory = concatenate_drake_trajectories([pregrasp_trajectory, grasp_trajectory])

    return pregrasp_and_grasp_trajectory


def make_grasp_pose_vertical(
    grasp_location: Vector3DType, gripper_open_direction: Vector3DType
) -> HomogeneousMatrixType:
    # grasp_location = np.array([0.2, 0.0, 0.7])

    gripper_forward_direction = np.array([0, 0, 1])
    Z = gripper_forward_direction / np.linalg.norm(gripper_forward_direction)
    # X = np.array([0, -1, 0])
    X = gripper_open_direction / np.linalg.norm(gripper_open_direction)
    # NOTE: we assume that the given X is perpendicular to Z

    Y = np.cross(Z, X)

    grasp_orientation = np.column_stack([X, Y, Z])
    grasp_pose_vertical = np.identity(4)
    grasp_pose_vertical[0:3, 0:3] = grasp_orientation
    grasp_pose_vertical[0:3, 3] = grasp_location
    return grasp_pose_vertical


def make_lowest_grasp_path_candidates(lowest_point: Vector3DType, grasp_depth: float = 0.0):
    grasp_pose_paths = []

    switch_angle = np.deg2rad(60)

    for angle in np.linspace(np.pi / 4, np.pi / 2, 4):
        gripper_open_direction = np.array([1, 0, 0]) if angle <= switch_angle else np.array([0, -1, 0])
        grasp_pose = make_grasp_pose_vertical(lowest_point, gripper_open_direction)

        global_y_rotation = RotationMatrix.MakeYRotation(angle).matrix()
        grasp_pose[:3, :3] = global_y_rotation @ grasp_pose[:3, :3]

        # Add grasp depth to grasp pose here
        grasp_pose[0:3, 3] += grasp_depth * grasp_pose[0:3, 2]

        for pregrasp_distance in np.linspace(0.05, 0.50, 10):
            distance = pregrasp_distance + grasp_depth
            pregrasp_pose = grasp_pose.copy()
            pregrasp_pose[0:3, 3] -= distance * grasp_pose[0:3, 2]

            rigid_transforms = [RigidTransform(pose) for pose in [pregrasp_pose, grasp_pose]]
            times = np.linspace(0, 1, len(rigid_transforms))
            pose_trajectory = PiecewisePose.MakeLinear(times=times, poses=rigid_transforms)
            steps = max(2, int(distance / 0.01))
            grasp_pose_path = discretize_drake_pose_trajectory(pose_trajectory, steps).poses

            grasp_pose_paths.append(grasp_pose_path)

    return grasp_pose_paths


def plan_lowest_point_grasp(
    lowest_point: Vector3DType,
    grasp_depth: float = 0.0,
    planner_pregrasp: DualArmOmplPlanner = None,
    home_joints_left: JointConfigurationType = None,
    home_joints_right: JointConfigurationType = None,
    inverse_kinematics_left_fn: InverseKinematicsFunctionType = None,
    inverse_kinematics_right_fn: InverseKinematicsFunctionType = None,
    is_state_valid_fn_grasp: JointConfigurationCheckerType = None,
    plant: MultibodyPlant = None,
    with_left: bool = True,
):
    grasp_pose_paths = make_lowest_grasp_path_candidates(lowest_point, grasp_depth)

    trajectory = None
    for i, grasp_pose_path in enumerate(grasp_pose_paths):
        try:
            trajectory = plan_to_grasp_pose_path(
                planner_pregrasp,
                grasp_pose_path,
                home_joints_left,
                home_joints_right,
                inverse_kinematics_left_fn,
                inverse_kinematics_right_fn,
                is_state_valid_fn_grasp,
                plant,
                with_left=with_left,
            )
            logger.success(f"Grasp pose path {i} is feasible.")
            break
        except GraspNotFeasibleError:
            logger.info(f"Grasp pose path {i} is not feasible.")
            continue

    if trajectory is None:
        raise ExhaustedOptionsError("All lowest point grasp paths are infeasible.")
    return trajectory
