from typing import Optional, Tuple

from pydrake.geometry import Meshcat, MeshcatVisualizer, MeshcatVisualizerParams, Role
from pydrake.planning import RobotDiagram, RobotDiagramBuilder
from pydrake.systems.framework import Context
from pydrake.visualization import ApplyVisualizationConfig, VisualizationConfig


def add_meshcat_to_builder(robot_diagram_builder: RobotDiagramBuilder) -> Meshcat:
    scene_graph = robot_diagram_builder.scene_graph()
    builder = robot_diagram_builder.builder()

    # Adding Meshcat must also be done before finalizing
    meshcat = Meshcat()
    meshcat.SetCameraPose([-2.0, 0, 1.0], [0, 0, 0])
    MeshcatVisualizer.AddToBuilder(builder, scene_graph, meshcat)

    # Add visualizer for proximity/collision geometry
    collision_params = MeshcatVisualizerParams(role=Role.kProximity, prefix="collision", visible_by_default=False)
    MeshcatVisualizer.AddToBuilder(builder, scene_graph.get_query_output_port(), meshcat, collision_params)

    return meshcat


def finish_build(
    robot_diagram_builder: RobotDiagramBuilder, meshcat: Optional[MeshcatVisualizer] = None
) -> Tuple[RobotDiagram, Context]:
    """Finish building the diagram and create a context.

    Note that after finishing the build, we can no longer add new objects to the Drake scene.
    However this needs to be done to be able to use many functionalities, e.g. collision checking.
    This is the standard workflow in Drake and a known "limitation".

    Args:
        robot_diagram_builder: The RobotDiagramBuilder object to which all models have already been added.
        meshcat: The MeshcatVisualizer object.

    Returns:
        diagram: The robot diagram.
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
