from pathlib import Path
from typing import Optional, Tuple

import cloth_tools
import numpy as np
from airo_planner.utils import files
from pydrake.geometry import Meshcat, MeshcatVisualizer, MeshcatVisualizerParams, Role
from pydrake.math import RigidTransform, RollPitchYaw
from pydrake.multibody.tree import ModelInstanceIndex
from pydrake.planning import RobotDiagramBuilder
from pydrake.systems.framework import Context, Diagram
from pydrake.visualization import ApplyVisualizationConfig, VisualizationConfig


def add_meshcat_to_builder(robot_diagram_builder: RobotDiagramBuilder) -> Meshcat:
    scene_graph = robot_diagram_builder.scene_graph()
    builder = robot_diagram_builder.builder()

    # Adding Meshcat must also be done before finalizing
    meshcat = Meshcat()
    MeshcatVisualizer.AddToBuilder(builder, scene_graph, meshcat)

    # Add visualizer for proximity/collision geometry
    collision_params = MeshcatVisualizerParams(role=Role.kProximity, prefix="collision", visible_by_default=False)
    MeshcatVisualizer.AddToBuilder(builder, scene_graph.get_query_output_port(), meshcat, collision_params)
    # meshcat.SetProperty("collision", "visible", False) # overwritten by .Build() I believe

    return meshcat


def add_dual_ur5e_and_table_to_builder(
    robot_diagram_builder: RobotDiagramBuilder,
) -> Tuple[ModelInstanceIndex, ModelInstanceIndex]:
    plant = robot_diagram_builder.plant()
    parser = robot_diagram_builder.parser()

    # This should get rid ot the warning for the mimic joints in the Robotiq gripper
    # But it doesn't seem to work
    # plant.set_discrete_contact_solver(DiscreteContactSolver.kSap)

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

    return (arm_left_index, arm_right_index)


def finish_build(
    robot_diagram_builder: RobotDiagramBuilder, meshcat: Optional[MeshcatVisualizer] = None
) -> Tuple[Diagram, Context]:
    """Finish building the diagram and create a context.

    Note that after finishing the build, we can no longer add new objects to the Drake scene.
    However this needs to be done to be able to use many functionalities, e.g. collision checking.
    This is the standard workflow in Drake and a known "limitation".

    Args:
        robot_diagram_builder: The RobotDiagramBuilder object to which all models have already been added.
        meshcat: The MeshcatVisualizer object.

    Returns:
        diagram: The diagram.
        context: A default context that you can use as you wish.
    """
    plant = robot_diagram_builder.plant()

    # These 4 lines are only for collision visualization
    if meshcat is not None:
        builder = robot_diagram_builder.builder()
        plant.Finalize()
        config = VisualizationConfig(publish_contacts=True, enable_alpha_sliders=True)
        ApplyVisualizationConfig(config, builder=builder, plant=plant, meshcat=meshcat)

    # A diagram is needed in the constructor of the SceneGraphCollisionChecker
    # However, calling .Build() prevents us from adding more models, e.g. runtime obstacles
    diagram = robot_diagram_builder.Build()

    # Create default contexts ~= state
    context = diagram.CreateDefaultContext()
    diagram.ForcedPublish(context)  # From this point we can see a visualization in Meshcat

    return diagram, context
