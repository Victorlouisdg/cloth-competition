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
