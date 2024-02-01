import numpy as np


def calculate_path_array_duration(path_array: np.ndarray, max_allowed_speed: float = 0.5) -> float:
    velocities = np.diff(path_array, axis=0)

    v_max = abs(velocities.max())
    v_min = abs(velocities.min())
    v_max_abs = max(abs(v_min), abs(v_max))

    duration_for_1rads = len(path_array) * v_max_abs

    duration = duration_for_1rads / max_allowed_speed
    return duration


def calculate_dual_path_duration(path, max_allowed_speed: float = 0.5) -> float:
    path_array = np.array(path).reshape(-1, 12)
    return calculate_path_array_duration(path_array, max_allowed_speed)


def interpolate_linearly(a, b, t):
    return a + t * (b - a)


def resample_path(path, n):
    m = len(path)
    path_new = []

    # example if m = 2 and n = 3, then i = 0, 1, 2 must produce j = 0, 0.5, 1, this i_to_j = 1/2 = (2-1)/(3-1)
    i_to_j = (m - 1) / (n - 1)

    for i in range(n):
        j_float = i_to_j * i
        j_fractional, j_integral = np.modf(j_float)

        j = int(j_integral)
        j_next = min(j + 1, m - 1)  # If j+1 would be m, then clamping to m-1 will give last element which is desired

        a = path[j]
        b = path[j_next]

        v = interpolate_linearly(a, b, j_fractional)
        path_new.append(v)

    return path_new


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


# def execute_joint_path_naive(dual_arm, path, duration):
#     period = duration / len(path)
#     for joints_left, joints_right in path:
#         left_servo = dual_arm.left_manipulator.servo_to_joint_configuration(joints_left, period)
#         right_servo = dual_arm.right_manipulator.servo_to_joint_configuration(joints_right, period)
#         left_servo.wait()
#         right_servo.wait()


def execute_dual_arm_joint_path(dual_arm, path, joint_speed=0.5):
    ensure_dual_arm_at_joint_configuration(dual_arm, path[0][0], path[0][1])

    duration = calculate_dual_path_duration(path, joint_speed)

    # TODO check whether arms are close to path start?
    period = 0.005  # 200 Hz, e-series should be able to handle 500 Hz

    n_servos = int(np.ceil(duration / period))

    path_left = [joint_left for joint_left, _ in path]
    path_right = [joint_right for _, joint_right in path]
    path_left_resampled = resample_path(path_left, n_servos)
    path_right_resampled = resample_path(path_right, n_servos)
    path_resampled = list(zip(path_left_resampled, path_right_resampled))

    for joints_left, joints_right in path_resampled:
        left_servo = dual_arm.left_manipulator.servo_to_joint_configuration(joints_left, period)
        right_servo = dual_arm.right_manipulator.servo_to_joint_configuration(joints_right, period)
        left_servo.wait()
        right_servo.wait()
