import os
from pathlib import Path
from typing import Tuple

from airo_dataset_tools.data_parsers.pose import Pose
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
