import os
from pathlib import Path
from typing import Tuple

from airo_dataset_tools.data_parsers.pose import Pose
from airo_robots.grippers.hardware.robotiq_2f85_urcap import Robotiq2F85
from airo_robots.manipulators.bimanual_position_manipulator import DualArmPositionManipulator
from airo_robots.manipulators.hardware.ur_rtde import URrtde
from airo_typing import HomogeneousMatrixType

CONFIG_DIR = "CLOTH_TOOLS_CONFIG_DIR"


def get_config_dir() -> str:
    """Path to a directory with configuration files such as camera extrinsics.

    Set in your conda with like so:
    conda env config vars set CLOTH_TOOLS_CONFIG_DIR=/home/victor/cloth-competition/config/

    Returns:
        str: Path to the config directory.
    """
    if CONFIG_DIR not in os.environ:
        raise ValueError(f"Environment variable {CONFIG_DIR} not set")
    return os.environ[CONFIG_DIR]


def load_camera_pose_in_left_and_right() -> Tuple[HomogeneousMatrixType, HomogeneousMatrixType]:
    """Load the camera pose in the left and right robot's base frame.

    Returns:
        camera pose in left and right robot's base frame
    """
    with open(Path(get_config_dir()) / "camera_pose_in_left.json", "r") as f:
        camera_pose_in_left = Pose.model_validate_json(f.read()).as_homogeneous_matrix()

    with open(Path(get_config_dir()) / "camera_pose_in_right.json", "r") as f:
        camera_pose_in_right = Pose.model_validate_json(f.read()).as_homogeneous_matrix()

    return camera_pose_in_left, camera_pose_in_right


def setup_dual_arm_ur5e_in_world(
    X_W_LCB: HomogeneousMatrixType,
    X_W_RCB: HomogeneousMatrixType,
    ip_address_left: str = "10.42.0.163",
    ip_address_right: str = "10.42.0.162",
) -> DualArmPositionManipulator:
    """Connect to the UR5e robots and Robotiq 2F-85 grippers. Sets the world frame to the base frame of the left robot.

    Returns:
        The initialized dual arm.
    """
    gripper_left = Robotiq2F85(ip_address_left)
    robot_left = URrtde(ip_address_left, URrtde.UR3E_CONFIG, gripper_left)

    gripper_right = Robotiq2F85(ip_address_right)
    robot_right = URrtde(ip_address_right, URrtde.UR3E_CONFIG, gripper_right)

    dual_arm = DualArmPositionManipulator(robot_left, X_W_LCB, robot_right, X_W_RCB)
    return dual_arm
