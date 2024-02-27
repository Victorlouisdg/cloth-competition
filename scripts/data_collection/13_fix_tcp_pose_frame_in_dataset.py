# During the data I accidently saved TCP poses expressed in the robot base frames instead of the world frame.
# However this is easy to fix because I also have the robot base poses in the world frame.

import json
import os
from pathlib import Path

from airo_dataset_tools.data_parsers.pose import Pose
from cloth_tools.dataset.format import COMPETITION_OBSERVATION_FILENAMES

script_dir = Path(os.path.realpath(__file__)).parent
data_dir = script_dir / Path("../../notebooks/data2")
data_dir = data_dir.resolve()

dataset_dir = data_dir / "cloth_competition_dataset_0000_0-9"


for sample_dir in sorted(os.listdir(dataset_dir)):
    print("Processing", sample_dir)
    for observation_dirname in ["observation_start", "observation_result"]:
        observation_dir = dataset_dir / sample_dir / observation_dirname
        filepaths = {
            key: str(observation_dir / filename) for key, filename in COMPETITION_OBSERVATION_FILENAMES.items()
        }

        with open(filepaths["arm_left_pose_in_world"], "r") as f:
            arm_left_pose_in_world = Pose.model_validate_json(f.read()).as_homogeneous_matrix()

        with open(filepaths["arm_right_pose_in_world"], "r") as f:
            arm_right_pose_in_world = Pose.model_validate_json(f.read()).as_homogeneous_matrix()

        # these in_world files incorrectly contain the poses in the robot base frame
        with open(filepaths["arm_left_tcp_pose_in_world"], "r") as f:
            arm_left_tcp_pose_in_base = Pose.model_validate_json(f.read()).as_homogeneous_matrix()

        with open(filepaths["arm_right_tcp_pose_in_world"], "r") as f:
            arm_right_tcp_pose_in_base = Pose.model_validate_json(f.read()).as_homogeneous_matrix()

        # Convert to the correct frame
        X_LCB_TCPL = arm_left_tcp_pose_in_base
        X_RCB_TCPR = arm_right_tcp_pose_in_base

        X_W_LCB = arm_left_pose_in_world
        X_W_RCB = arm_right_pose_in_world

        X_W_TCPL = X_W_LCB @ X_LCB_TCPL
        X_W_TCPR = X_W_RCB @ X_RCB_TCPR

        arm_left_tcp_pose_in_world = X_W_TCPL
        arm_right_tcp_pose_in_world = X_W_TCPR

        with open(filepaths["arm_left_tcp_pose_in_world"], "w") as f:
            arm_left_tcp_pose_model = Pose.from_homogeneous_matrix(arm_left_tcp_pose_in_world)
            json.dump(arm_left_tcp_pose_model.model_dump(exclude_none=True), f, indent=4)

        with open(filepaths["arm_right_tcp_pose_in_world"], "w") as f:
            arm_right_tcp_pose_model = Pose.from_homogeneous_matrix(arm_right_tcp_pose_in_world)
            json.dump(arm_right_tcp_pose_model.model_dump(exclude_none=True), f, indent=4)
