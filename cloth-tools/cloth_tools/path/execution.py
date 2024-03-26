import numpy as np
from airo_robots.manipulators.bimanual_position_manipulator import DualArmPositionManipulator
from pydrake.trajectories import Trajectory


def ensure_dual_arm_at_joint_configuration(dual_arm, joints_left, joints_right, tolerance=0.1) -> None:
    """Sanity check that the arm are were you expect them to be,
    e.g. close to that start of a path you are about to execute.

    Raises ValueError if the arms are not at the expected joints.
    """
    current_joints_left = dual_arm.left_manipulator.get_joint_configuration()
    current_joints_right = dual_arm.right_manipulator.get_joint_configuration()

    left_distance = np.linalg.norm(current_joints_left - joints_left)
    right_distance = np.linalg.norm(current_joints_right - joints_right)

    if left_distance > tolerance:
        raise ValueError(
            f"Left arm is at {current_joints_left} but should be at {joints_left}, distance: {left_distance}"
        )
    if right_distance > tolerance:
        raise ValueError(
            f"Right arm is at {current_joints_right} but should be at {joints_right}, distance: {right_distance}"
        )


def execute_dual_arm_trajectory(
    dual_arm: DualArmPositionManipulator, joint_trajectory: Trajectory, time_trajectory: Trajectory
):
    # TODO don't receive joint and time trajectory separately, but as a single PathParametrizedTrajectory
    # TODO use discretize trajectory
    start_joints = joint_trajectory.value(time_trajectory.value(0).item()).squeeze()
    start_joints_left = start_joints[0:6]
    start_joints_right = start_joints[6:12]

    ensure_dual_arm_at_joint_configuration(dual_arm, start_joints_left, start_joints_right)

    period = 0.005
    duration = time_trajectory.end_time()

    n_servos = int(np.ceil(duration / period))
    period_adjusted = duration / n_servos  # can be slightly different from period due to rounding

    for t in np.linspace(0, duration, n_servos):
        joints = joint_trajectory.value(time_trajectory.value(t).item()).squeeze()
        joints_left = joints[0:6]
        joints_right = joints[6:12]
        left_servo = dual_arm.left_manipulator.servo_to_joint_configuration(joints_left, period_adjusted)
        right_servo = dual_arm.right_manipulator.servo_to_joint_configuration(joints_right, period_adjusted)
        left_servo.wait()
        right_servo.wait()

    # This avoids the abrupt stop and "thunk" sounds at the end of paths that end with non-zero velocity
    # However, I believe these functions are blocking, so right only stops after left has stopped.
    dual_arm.left_manipulator.rtde_control.servoStop(2.0)
    dual_arm.right_manipulator.rtde_control.servoStop(2.0)

    left_finished = dual_arm.left_manipulator.move_to_joint_configuration(joints_left)
    right_finished = dual_arm.right_manipulator.move_to_joint_configuration(joints_right)

    left_finished.wait()
    right_finished.wait()
