from typing import Tuple

import airo_models
import numpy as np
from airo_typing import HomogeneousMatrixType
from cloth_tools.urdf.robotiq import create_static_robotiq_2f_85_urdf
from pydrake.math import RigidTransform, RollPitchYaw
from pydrake.multibody.tree import ModelInstanceIndex
from pydrake.planning import RobotDiagramBuilder

# Some fixed transforms and default poses (e.g. for when you don't have access to the real robot)

# X_CB_B is the 180 rotation between ROS URDF base and the UR control box base
X_CB_B = RigidTransform(rpy=RollPitchYaw([0, 0, np.pi]), p=[0, 0, 0])

ARM_Y_DEFAULT = 0.45

# The default pose of the left robot base when using the ROS URDFs
X_W_L_DEFAULT = RigidTransform(rpy=RollPitchYaw([0, 0, np.pi / 2]), p=[0, ARM_Y_DEFAULT, 0])

# The default pose of the right robot base when using the ROS URDFs
X_W_R_DEFAULT = RigidTransform(rpy=RollPitchYaw([0, 0, np.pi / 2]), p=[0, -ARM_Y_DEFAULT, 0])

X_URTOOL0_ROBOTIQ = RigidTransform(rpy=RollPitchYaw([0, 0, np.pi / 2]), p=[0, 0, 0])


def add_ur5e_and_table_to_builder(
    robot_diagram_builder: RobotDiagramBuilder,
) -> Tuple[ModelInstanceIndex, ModelInstanceIndex]:
    plant = robot_diagram_builder.plant()
    parser = robot_diagram_builder.parser()

    # Load URDF files
    ur5e_urdf_path = airo_models.get_urdf_path("ur5e")
    robotiq_urdf_path = create_static_robotiq_2f_85_urdf()

    table_thickness = 0.2
    table_urdf_path = airo_models.box_urdf_path((2.0, 2.4, table_thickness), "table")

    arm_index = parser.AddModelFromFile(ur5e_urdf_path)
    gripper_index = parser.AddModelFromFile(robotiq_urdf_path)
    table_index = parser.AddModelFromFile(table_urdf_path)

    # Weld some frames together
    world_frame = plant.world_frame()
    table_frame = plant.GetFrameByName("base_link", table_index)
    arm_frame = plant.GetFrameByName("base_link", arm_index)
    arm_tool_frame = plant.GetFrameByName("tool0", arm_index)
    gripper_frame = plant.GetFrameByName("base_link", gripper_index)

    arm_transform = RigidTransform(rpy=RollPitchYaw([0, 0, np.pi]), p=[0, 0, 0])
    table_transform = RigidTransform(p=[0, 0, -table_thickness / 2])

    plant.WeldFrames(world_frame, arm_frame, arm_transform)
    plant.WeldFrames(arm_tool_frame, gripper_frame, X_URTOOL0_ROBOTIQ)
    plant.WeldFrames(world_frame, table_frame, table_transform)

    return arm_index, gripper_index


def add_dual_ur5e_and_table_to_builder(
    robot_diagram_builder: RobotDiagramBuilder,
    X_W_LCB: HomogeneousMatrixType | None = None,
    X_W_RCB: HomogeneousMatrixType | None = None,
) -> Tuple[Tuple[ModelInstanceIndex, ModelInstanceIndex], Tuple[ModelInstanceIndex, ModelInstanceIndex]]:
    plant = robot_diagram_builder.plant()
    parser = robot_diagram_builder.parser()

    # Load URDF files
    ur5e_urdf_path = airo_models.get_urdf_path("ur5e")
    robotiq_urdf_path = create_static_robotiq_2f_85_urdf()

    table_thickness = 0.2
    table_urdf_path = airo_models.box_urdf_path((2.0, 2.4, table_thickness), "table")
    wall_thickness = 0.2
    wall_back_urdf_path = airo_models.box_urdf_path((wall_thickness, 2.7, 2.0), "wall_back")
    wall_left_urdf_path = airo_models.box_urdf_path((2.0, wall_thickness, 2.0), "wall_left")
    wall_right_urdf_path = airo_models.box_urdf_path((2.0, wall_thickness, 2.0), "wall_right")

    arm_left_index = parser.AddModelFromFile(ur5e_urdf_path, model_name="arm_left")
    arm_right_index = parser.AddModelFromFile(ur5e_urdf_path, model_name="arm_right")
    gripper_left_index = parser.AddModelFromFile(robotiq_urdf_path, "gripper_left")
    gripper_right_index = parser.AddModelFromFile(robotiq_urdf_path, model_name="gripper_right")

    table_index = parser.AddModelFromFile(table_urdf_path)
    wall_back_index = parser.AddModelFromFile(wall_back_urdf_path)
    wall_left_index = parser.AddModelFromFile(wall_left_urdf_path)
    wall_right_index = parser.AddModelFromFile(wall_right_urdf_path)

    # Weld some frames together
    world_frame = plant.world_frame()
    arm_left_frame = plant.GetFrameByName("base_link", arm_left_index)
    arm_right_frame = plant.GetFrameByName("base_link", arm_right_index)
    arm_left_tool_frame = plant.GetFrameByName("tool0", arm_left_index)
    arm_right_tool_frame = plant.GetFrameByName("tool0", arm_right_index)
    gripper_left_frame = plant.GetFrameByName("base_link", gripper_left_index)
    gripper_right_frame = plant.GetFrameByName("base_link", gripper_right_index)

    table_frame = plant.GetFrameByName("base_link", table_index)
    wall_back_frame = plant.GetFrameByName("base_link", wall_back_index)
    wall_left_frame = plant.GetFrameByName("base_link", wall_left_index)
    wall_right_frame = plant.GetFrameByName("base_link", wall_right_index)

    X_W_L = X_W_L_DEFAULT if X_W_LCB is None else RigidTransform(X_W_LCB @ X_CB_B.GetAsMatrix4())
    X_W_R = X_W_R_DEFAULT if X_W_RCB is None else RigidTransform(X_W_RCB @ X_CB_B.GetAsMatrix4())

    arm_left_transform = X_W_L
    arm_right_transform = X_W_R
    arm_y = arm_left_transform.translation()[1]

    table_transform = RigidTransform(p=[0, 0, -table_thickness / 2])
    wall_back_transform = RigidTransform(p=[0.9 + wall_thickness / 2, 0, 0])
    wall_left_transform = RigidTransform(p=[0, arm_y + 0.7 + wall_thickness / 2, 0])
    wall_right_transform = RigidTransform(p=[0, -arm_y - 0.7 - wall_thickness / 2, 0])

    plant.WeldFrames(world_frame, arm_left_frame, arm_left_transform)
    plant.WeldFrames(world_frame, arm_right_frame, arm_right_transform)
    plant.WeldFrames(arm_left_tool_frame, gripper_left_frame, X_URTOOL0_ROBOTIQ)
    plant.WeldFrames(arm_right_tool_frame, gripper_right_frame, X_URTOOL0_ROBOTIQ)
    plant.WeldFrames(world_frame, table_frame, table_transform)
    plant.WeldFrames(world_frame, wall_back_frame, wall_back_transform)
    plant.WeldFrames(world_frame, wall_left_frame, wall_left_transform)
    plant.WeldFrames(world_frame, wall_right_frame, wall_right_transform)

    return (arm_left_index, arm_right_index), (gripper_left_index, gripper_right_index)
