from typing import List, Tuple

import numpy as np
from airo_typing import JointConfigurationType
from loguru import logger
from pydrake.geometry import Cylinder, Meshcat, Rgba
from pydrake.math import RigidTransform, RotationMatrix
from pydrake.multibody.tree import ModelInstanceIndex
from pydrake.planning import RobotDiagram
from pydrake.systems.framework import Context


def publish_joint_path(
    joint_path: List[JointConfigurationType],
    duration: float,
    meshcat: Meshcat,
    diagram: RobotDiagram,
    context: Context,
    arm_index: ModelInstanceIndex,
) -> None:
    """See PublishPositionTrajectory in Tedrake manipulation repo."""

    if joint_path is None:
        logger.warning("path is None, not publishing anything.")
        return

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
    diagram: RobotDiagram,
    context: Context,
    arm_left_index: ModelInstanceIndex,
    arm_right_index: ModelInstanceIndex,
) -> None:
    if dual_arm_joint_path is None:
        logger.warning("path is None, not publishing anything.")
        return

    # TODO consider reducing duplication with publish_joint_path
    plant = diagram.plant()
    plant_context = plant.GetMyContextFromRoot(context)

    # meshcat.DeleteRecording() # Doesn't seem necessary, old one is overwritten
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


def publish_ik_solutions(
    solutions: List[JointConfigurationType],
    time_per_solution: float,
    meshcat: Meshcat,
    diagram: RobotDiagram,
    context: Context,
    arm_index: ModelInstanceIndex,
):
    if solutions is None:
        logger.warning("solutions is None, not publishing anything.")
        return

    plant = diagram.plant()
    plant_context = plant.GetMyContextFromRoot(context)

    fps = 60.0
    meshcat.StartRecording(set_visualizations_while_recording=False, frames_per_second=fps)

    t = 0.0

    for joint_configuration in solutions:
        for _ in range(int(time_per_solution * fps)):
            context.SetTime(t)
            plant.SetPositions(plant_context, arm_index, joint_configuration.squeeze())
            diagram.ForcedPublish(context)
            t += 1.0 / fps

    meshcat.StopRecording()
    meshcat.PublishRecording()


def add_meshcat_triad(
    meshcat, path, length=0.05, radius=0.002, opacity=1.0, X_W_Triad=RigidTransform(), rgba_xyz=None
):
    if rgba_xyz is None:
        rgba_xyz = [[1, 0, 0, opacity], [0, 1, 0, opacity], [0, 0, 1, opacity]]

    meshcat.SetTransform(path, X_W_Triad)
    # x-axis
    X_TG = RigidTransform(RotationMatrix.MakeYRotation(np.pi / 2), [length / 2.0, 0, 0])
    meshcat.SetTransform(path + "/x-axis", X_TG)
    meshcat.SetObject(path + "/x-axis", Cylinder(radius, length), Rgba(*rgba_xyz[0]))

    # y-axis
    X_TG = RigidTransform(RotationMatrix.MakeXRotation(np.pi / 2), [0, length / 2.0, 0])
    meshcat.SetTransform(path + "/y-axis", X_TG)
    meshcat.SetObject(path + "/y-axis", Cylinder(radius, length), Rgba(*rgba_xyz[1]))

    # z-axis
    X_TG = RigidTransform([0, 0, length / 2.0])
    meshcat.SetTransform(path + "/z-axis", X_TG)
    meshcat.SetObject(path + "/z-axis", Cylinder(radius, length), Rgba(*rgba_xyz[2]))
