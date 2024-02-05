import numpy as np
from airo_typing import HomogeneousMatrixType


def create_egocentric_world_frame(
    camera_pose_in_left: HomogeneousMatrixType, camera_pose_in_right: HomogeneousMatrixType
) -> tuple[HomogeneousMatrixType:, HomogeneousMatrixType, HomogeneousMatrixType]:
    """Creates an egocentric world frame based on the camera poses in the left and right robot base frames.

    This world frame is defined as follows:
    * The world origin is the middle point between the two robot bases.
    * The world Y-axis is the vector from the right robot base to the left robot base.
    * The world Z-axis is the same as the left robot base Z-axis (adjusted slightly to ensure orthogonality).
    * The world X-axis is the cross product of the world Y and Z axes.

    """
    X_LCB_C = camera_pose_in_left
    X_RCB_C = camera_pose_in_right
    X_LCB_RCB = X_LCB_C @ np.linalg.inv(X_RCB_C)
    p_LCB_RCB = X_LCB_RCB[:3, 3]

    # Define the world origin as the middle point between the two robot bases
    p_LCB_W = p_LCB_RCB / 2.0

    # Define the world Y-axis as the vector from the right robot base to the left robot base
    y_LCB_W = -p_LCB_RCB / np.linalg.norm(p_LCB_RCB)

    # Define (for now) that the world Z-axis is the same as the left_base Z-axis
    # We will fix later that it is not exactly orthogonal to the world Y-axis
    z_LCB_W = np.array([0, 0, 1])

    # Define the world X-axis as the cross product of the world Y and Z axes
    x_LCB_W = np.cross(y_LCB_W, z_LCB_W)
    x_LCB_W = x_LCB_W / np.linalg.norm(x_LCB_W)  # normalize because Z was not exactly orthogonal to Y

    # Ensure the world Z-axis is orthogonal to the world X and Y axes
    z_LCB_W = np.cross(x_LCB_W, y_LCB_W)

    # Create the rotation matrix
    R_LCB_W = np.column_stack((x_LCB_W, y_LCB_W, z_LCB_W))

    # Create the homogeneous transformation matrix
    X_LCB_W = np.identity(4)
    X_LCB_W[:3, :3] = R_LCB_W
    X_LCB_W[:3, 3] = p_LCB_W

    X_W_LCB = np.linalg.inv(X_LCB_W)
    X_W_RCB = X_W_LCB @ X_LCB_RCB

    X_W_C = X_W_LCB @ X_LCB_C
    # X_W_C_ = X_W_RCB @ X_RCB_C # Alternative, but should give exactly the same result I belive
    return X_W_C, X_W_LCB, X_W_RCB
