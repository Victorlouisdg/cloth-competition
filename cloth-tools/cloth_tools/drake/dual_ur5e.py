from pathlib import Path
from typing import Tuple

import cloth_tools
import numpy as np
from airo_planner.utils import files
from pydrake.geometry import Meshcat, MeshcatVisualizer
from pydrake.math import RigidTransform, RollPitchYaw
from pydrake.multibody.plant import DiscreteContactSolver
from pydrake.multibody.tree import ModelInstanceIndex
from pydrake.planning import RobotDiagramBuilder


def get_robot_diagram_builder_dual_ur5e() -> Tuple[RobotDiagramBuilder, Tuple[ModelInstanceIndex, ModelInstanceIndex]]:
    """
    Create a Drake setup with two UR5e robots and a table.
    Nothing is built/finalized yet, so you can load more URDFs and weld frames etc.

    Returns:
        robot_diagram_builder: A RobotDiagramBuilder with two UR5e robots and a table.
        arm_left_index: The index of the left arm.
        arm_right_index: The index of the right arm.
    """
    robot_diagram_builder = RobotDiagramBuilder()
    scene_graph = robot_diagram_builder.scene_graph()
    plant = robot_diagram_builder.plant()
    builder = robot_diagram_builder.builder()
    parser = robot_diagram_builder.parser()

    # Add visualizer
    meshcat = Meshcat()
    MeshcatVisualizer.AddToBuilder(builder, scene_graph, meshcat)
    # config = VisualizationConfig(publish_contacts=True, enable_alpha_sliders=True)
    # ApplyVisualizationConfig(config, builder=builder, plant=plant, meshcat=meshcat)

    # This get rid ot the warning for the mimic joints in the Robotiq gripper
    plant.set_discrete_contact_solver(DiscreteContactSolver.kSap)

    # Load URDF files
    resources_root = str(files.get_resources_dir())
    ur5e_urdf = Path(resources_root) / "robots" / "ur5e" / "ur5e.urdf"
    robotiq_2f_85_gripper_urdf = (
        Path(resources_root) / "grippers" / "2f_85_gripper" / "urdf" / "robotiq_2f_85_static.urdf"
    )
    table_urdf = cloth_tools.resources.table

    arm_left_index = parser.AddModelFromFile(str(ur5e_urdf), model_name="arm_left")
    arm_right_index = parser.AddModelFromFile(str(ur5e_urdf), model_name="arm_right")
    gripper_left_index = parser.AddModelFromFile(str(robotiq_2f_85_gripper_urdf), model_name="gripper_left")
    gripper_right_index = parser.AddModelFromFile(str(robotiq_2f_85_gripper_urdf), model_name="gripper_right")
    table_index = parser.AddModelFromFile(str(table_urdf))

    # Weld some frames together
    world_frame = plant.world_frame()
    arm_left_frame = plant.GetFrameByName("base_link", arm_left_index)
    arm_right_frame = plant.GetFrameByName("base_link", arm_right_index)
    arm_left_wrist_frame = plant.GetFrameByName("wrist_3_link", arm_left_index)
    arm_right_wrist_frame = plant.GetFrameByName("wrist_3_link", arm_right_index)
    gripper_left_frame = plant.GetFrameByName("base_link", gripper_left_index)
    gripper_right_frame = plant.GetFrameByName("base_link", gripper_right_index)
    table_frame = plant.GetFrameByName("base_link", table_index)

    distance_between_arms = 0.9
    distance_between_arms_half = distance_between_arms / 2

    plant.WeldFrames(world_frame, arm_left_frame)
    plant.WeldFrames(world_frame, arm_right_frame, RigidTransform([distance_between_arms, 0, 0]))
    plant.WeldFrames(
        arm_left_wrist_frame, gripper_left_frame, RigidTransform(p=[0, 0, 0], rpy=RollPitchYaw([0, 0, np.pi / 2]))
    )
    plant.WeldFrames(
        arm_right_wrist_frame, gripper_right_frame, RigidTransform(p=[0, 0, 0], rpy=RollPitchYaw([0, 0, np.pi / 2]))
    )
    plant.WeldFrames(world_frame, table_frame, RigidTransform([distance_between_arms_half, 0, 0]))

    return robot_diagram_builder, (arm_left_index, arm_right_index)
