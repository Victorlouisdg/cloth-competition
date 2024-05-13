from functools import partial
from pathlib import Path

import numpy as np
from airo_drake import animate_dual_joint_trajectory, time_parametrize_toppra
from airo_planner import DualArmOmplPlanner, PlannerError
from cloth_tools.controllers.grasp_highest_controller import hang_in_the_air_tcp_pose
from cloth_tools.controllers.stretch_controller import StretchController
from cloth_tools.dataset.bookkeeping import ensure_dataset_dir, find_highest_suffix
from cloth_tools.dataset.collection import collect_observation
from cloth_tools.dataset.format import save_competition_observation
from cloth_tools.kinematics.constants import TCP_TRANSFORM
from cloth_tools.kinematics.inverse_kinematics import inverse_kinematics_in_world_post_processed_fn
from cloth_tools.motion_blur_detector import MotionBlurDetector
from cloth_tools.stations.competition_station import CompetitionStation
from cloth_tools.trajectory_execution import execute_dual_arm_drake_trajectory
from linen.elemental.move_backwards import move_pose_backwards
from loguru import logger


def move_to_start_pose(station: CompetitionStation):
    tcp_distance = 0.25

    hang_pose_left = hang_in_the_air_tcp_pose(left=True)
    hang_pose_right = hang_in_the_air_tcp_pose(left=False)

    backwards_shift = 0.9 * (tcp_distance / 2.0)  # set distance to a percentage of the distance when grasping
    stretch_pose_left = move_pose_backwards(hang_pose_left, backwards_shift)
    stretch_pose_right = move_pose_backwards(hang_pose_right, backwards_shift)

    # Move arms closer to the camera
    global_x_shift = -0.4
    stretch_pose_left[:3, 3] += np.array([global_x_shift, 0, 0])
    stretch_pose_right[:3, 3] += np.array([global_x_shift, 0, 0])

    dual_arm = station.dual_arm
    start_joints_left = dual_arm.left_manipulator.get_joint_configuration()
    start_joints_right = dual_arm.right_manipulator.get_joint_configuration()

    X_W_LCB = station.left_arm_pose
    X_W_RCB = station.right_arm_pose

    inverse_kinematics_left_fn = partial(
        inverse_kinematics_in_world_post_processed_fn,
        X_W_CB=X_W_LCB,
        tcp_transform=TCP_TRANSFORM,
        reference_configuration=start_joints_left,
    )
    inverse_kinematics_right_fn = partial(
        inverse_kinematics_in_world_post_processed_fn,
        X_W_CB=X_W_RCB,
        tcp_transform=TCP_TRANSFORM,
        reference_configuration=start_joints_right,
    )

    # Make a custom planner that has post-processed IK functions to reduce cloth twisting
    planner = DualArmOmplPlanner(
        is_state_valid_fn=station.is_state_valid_fn,
        inverse_kinematics_left_fn=inverse_kinematics_left_fn,
        inverse_kinematics_right_fn=inverse_kinematics_right_fn,
        joint_bounds_left=station.joint_bounds_left,
        joint_bounds_right=station.joint_bounds_right,
    )

    try:
        path_to_stretch = planner.plan_to_tcp_pose(
            start_joints_left, start_joints_right, stretch_pose_left, stretch_pose_right
        )
    except PlannerError as e:
        logger.info(f"Failed to plan to stretch pose. Exception was:\n {e}.")
        return

    # Time parametrize the path
    # Move very slowly
    plant = station.drake_scene.robot_diagram.plant()
    trajectory_to_stretch = time_parametrize_toppra(
        plant, path_to_stretch, joint_speed_limit=0.5, joint_acceleration_limit=0.5
    )

    scene = station.drake_scene

    animate_dual_joint_trajectory(
        scene.meshcat,
        scene.robot_diagram,
        scene.arm_left_index,
        scene.arm_right_index,
        trajectory_to_stretch,
    )

    # Wait for user input:
    # - Press Enter to execute the trajectory
    # - Press Ctrl+C to exit
    # input("Press Enter to execute the trajectory...")

    # Execute the trajectory
    execute_dual_arm_drake_trajectory(station.dual_arm, trajectory_to_stretch)


if __name__ == "__main__":
    station = CompetitionStation()
    dataset_dir = Path(ensure_dataset_dir("notebooks/data/cloth_competition_references_0002"))

    while True:
        sample_index = find_highest_suffix(dataset_dir, "sample") + 1

        sample_dir = dataset_dir / f"sample_{sample_index:06d}"
        observation_dir = sample_dir / "observation_result"
        observation_dir.mkdir(parents=True, exist_ok=True)

        move_to_start_pose(station)

        # Close gripper already slightly
        dual_arm = station.dual_arm
        dual_arm.left_manipulator.gripper.move(0.01).wait()
        dual_arm.right_manipulator.gripper.move(0.01).wait()

        dual_arm.left_manipulator.rtde_control.zeroFtSensor()
        dual_arm.right_manipulator.rtde_control.zeroFtSensor()

        # Execute the stretch controller
        stretch_controller = StretchController(station)
        stretch_controller.execute(interactive=True)

        motion_blur_detector = MotionBlurDetector(station.camera, station.hanging_cloth_crop)
        motion_blur_detector.wait_for_blur_to_stabilize(timeout=15)

        # Save the reference observation
        observation_reference = collect_observation(station)
        save_competition_observation(observation_reference, observation_dir)
