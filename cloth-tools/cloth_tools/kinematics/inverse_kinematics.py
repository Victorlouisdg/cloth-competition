import numpy as np
from airo_typing import HomogeneousMatrixType, JointConfigurationType
from ur_analytic_ik import ur5e


def inverse_kinematics_in_world_fn(
    tcp_pose: HomogeneousMatrixType, X_W_CB: HomogeneousMatrixType, tcp_transform: HomogeneousMatrixType
) -> list[JointConfigurationType]:
    X_W_TCP = tcp_pose
    X_CB_W = np.linalg.inv(X_W_CB)
    solutions_1x6 = ur5e.inverse_kinematics_with_tcp(X_CB_W @ X_W_TCP, tcp_transform)
    solutions = [solution.squeeze() for solution in solutions_1x6]
    return solutions


# TODO document the functions below:
# in short: they change the solutions given by the IK that are within (-pi, pi)
# to lie between (-2pi, 2pi) of the angle of a joint is closer to the angle of a reference (e.g. starting) configuration
# This solved the issue of the robot arm "twisting" the cloth too much
# e.g. when for from -3.13 -> 1.57 instead of -3.13 -> -4.70
def post_process_solution(
    solution: JointConfigurationType, reference_configuration: JointConfigurationType
) -> JointConfigurationType:
    new_solution = np.zeros_like(solution)
    for i in range(len(solution)):
        angle = solution[i]
        alternative_angle = angle + 2 * np.pi if angle < 0 else angle - 2 * np.pi
        reference_angle = reference_configuration[i]
        abs_diff = np.abs(angle - reference_angle)
        abs_diff_new = np.abs(alternative_angle - reference_angle)
        new_solution[i] = alternative_angle if abs_diff_new < abs_diff else angle
    return new_solution


def inverse_kinematics_in_world_post_processed_fn(
    tcp_pose: HomogeneousMatrixType,
    X_W_CB: HomogeneousMatrixType,
    tcp_transform: HomogeneousMatrixType,
    reference_configuration: JointConfigurationType,
) -> list[JointConfigurationType]:
    X_W_TCP = tcp_pose
    X_CB_W = np.linalg.inv(X_W_CB)
    solutions_1x6 = ur5e.inverse_kinematics_with_tcp(X_CB_W @ X_W_TCP, tcp_transform)
    solutions = [solution.squeeze() for solution in solutions_1x6]
    solutions_post_processed = [post_process_solution(s, reference_configuration) for s in solutions]
    return solutions_post_processed
