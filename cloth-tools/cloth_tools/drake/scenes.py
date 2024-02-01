from typing import Tuple

import airo_models
import numpy as np
from cloth_tools.urdf.robotiq import create_static_robotiq_2f_85_urdf
from pydrake.math import RigidTransform, RollPitchYaw
from pydrake.multibody.tree import ModelInstanceIndex
from pydrake.planning import RobotDiagramBuilder


def add_ur5e_and_table_to_builder(
    robot_diagram_builder: RobotDiagramBuilder,
) -> Tuple[ModelInstanceIndex, ModelInstanceIndex]:
    plant = robot_diagram_builder.plant()
    parser = robot_diagram_builder.parser()

    # Load URDF files
    ur5e_urdf_path = airo_models.get_urdf_path("ur5e")
    robotiq_urdf_path = create_static_robotiq_2f_85_urdf()
    table_urdf_path = "table.urdf"

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
    robotiq_ur_transform = RigidTransform(rpy=RollPitchYaw([0, 0, np.pi / 2]), p=[0, 0, 0])

    plant.WeldFrames(world_frame, arm_frame, arm_transform)
    plant.WeldFrames(arm_tool_frame, gripper_frame, robotiq_ur_transform)
    plant.WeldFrames(world_frame, table_frame)

    return arm_index, gripper_index


def add_dual_ur5e_and_table_to_builder(
    robot_diagram_builder: RobotDiagramBuilder,
) -> Tuple[Tuple[ModelInstanceIndex, ModelInstanceIndex], Tuple[ModelInstanceIndex, ModelInstanceIndex]]:
    plant = robot_diagram_builder.plant()
    parser = robot_diagram_builder.parser()

    # Load URDF files
    ur5e_urdf_path = airo_models.get_urdf_path("ur5e")
    robotiq_urdf_path = create_static_robotiq_2f_85_urdf()
    table_urdf_path = "table.urdf"

    arm_left_index = parser.AddModelFromFile(ur5e_urdf_path, model_name="arm_left")
    arm_right_index = parser.AddModelFromFile(ur5e_urdf_path, model_name="arm_right")
    gripper_left_index = parser.AddModelFromFile(robotiq_urdf_path, "gripper_left")
    gripper_right_index = parser.AddModelFromFile(robotiq_urdf_path, model_name="gripper_right")
    table_index = parser.AddModelFromFile(table_urdf_path)

    # Weld some frames together
    world_frame = plant.world_frame()
    table_frame = plant.GetFrameByName("base_link", table_index)
    arm_left_frame = plant.GetFrameByName("base_link", arm_left_index)
    arm_right_frame = plant.GetFrameByName("base_link", arm_right_index)
    arm_left_tool_frame = plant.GetFrameByName("tool0", arm_left_index)
    arm_right_tool_frame = plant.GetFrameByName("tool0", arm_right_index)
    gripper_left_frame = plant.GetFrameByName("base_link", gripper_left_index)
    gripper_right_frame = plant.GetFrameByName("base_link", gripper_right_index)

    y_distance = 0.45
    arm_left_transform = RigidTransform(rpy=RollPitchYaw([0, 0, np.pi / 2]), p=[0, y_distance, 0])
    arm_right_transform = RigidTransform(rpy=RollPitchYaw([0, 0, np.pi / 2]), p=[0, -y_distance, 0])
    robotiq_ur_transform = RigidTransform(rpy=RollPitchYaw([0, 0, np.pi / 2]), p=[0, 0, 0])

    plant.WeldFrames(world_frame, arm_left_frame, arm_left_transform)
    plant.WeldFrames(world_frame, arm_right_frame, arm_right_transform)
    plant.WeldFrames(arm_left_tool_frame, gripper_left_frame, robotiq_ur_transform)
    plant.WeldFrames(arm_right_tool_frame, gripper_right_frame, robotiq_ur_transform)
    plant.WeldFrames(world_frame, table_frame)

    return (arm_left_index, arm_right_index), (gripper_left_index, gripper_right_index)
