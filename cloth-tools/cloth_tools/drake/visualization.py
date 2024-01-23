from typing import List, Tuple

from airo_typing import JointConfigurationType
from pydrake.geometry import Meshcat
from pydrake.multibody.tree import ModelInstanceIndex
from pydrake.systems.framework import Context, Diagram


def publish_joint_path(
    joint_path: List[JointConfigurationType],
    duration: float,
    meshcat: Meshcat,
    diagram: Diagram,
    context: Context,
    arm_index: ModelInstanceIndex,
) -> None:
    plant = diagram.plant()
    plant_context = plant.GetMyContextFromRoot(context)

    meshcat.StartRecording(set_visualizations_while_recording=False)

    period = duration / len(joint_path)
    t = 0.0

    for joint_configuration in joint_path:
        context.SetTime(t)
        plant.SetPositions(plant_context, arm_index, joint_configuration)
        diagram.ForcedPublish(context)
        t += period

    meshcat.StopRecording()
    meshcat.PublishRecording()


def publish_dual_arm_joint_path(
    dual_arm_joint_path: List[Tuple[JointConfigurationType, JointConfigurationType]],
    duration: float,
    meshcat: Meshcat,
    diagram: Diagram,
    context: Context,
    arm_left_index: ModelInstanceIndex,
    arm_right_index: ModelInstanceIndex,
) -> None:
    # TODO consider reducing duplication with publish_joint_path
    plant = diagram.plant()
    plant_context = plant.GetMyContextFromRoot(context)

    meshcat.StartRecording(set_visualizations_while_recording=False)

    period = duration / len(dual_arm_joint_path)
    t = 0.0

    for joint_configuration_left, joint_configuration_right in dual_arm_joint_path:
        context.SetTime(t)
        plant.SetPositions(plant_context, arm_left_index, joint_configuration_left)
        plant.SetPositions(plant_context, arm_right_index, joint_configuration_right)
        diagram.ForcedPublish(context)
        t += period

    meshcat.StopRecording()
    meshcat.PublishRecording()
