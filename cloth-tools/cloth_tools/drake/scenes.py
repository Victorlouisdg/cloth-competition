import tempfile

import airo_models
import numpy as np
import open3d as o3d
from airo_camera_toolkit.point_clouds.conversions import point_cloud_to_open3d
from airo_camera_toolkit.point_clouds.operations import crop_point_cloud
from airo_drake import X_URBASE_ROSBASE, DualArmScene, add_floor, add_manipulator, add_meshcat, add_wall, finish_build
from airo_models import mesh_urdf_path
from airo_typing import HomogeneousMatrixType, JointConfigurationType, PointCloud
from cloth_tools.bounding_boxes import BBOX_CLOTH_IN_THE_AIR
from cloth_tools.dataset.format import CompetitionObservation
from pydrake.math import RigidTransform, RollPitchYaw
from pydrake.multibody.tree import ModelInstanceIndex
from pydrake.planning import RobotDiagramBuilder, SceneGraphCollisionChecker

# Some fixed transforms and default poses (e.g. for when you don't have access to the real robot)
ARM_Y_DEFAULT = 0.45

X_CB_B = X_URBASE_ROSBASE

# The default pose of the left robot base when using the ROS URDFs
X_W_L_DEFAULT = RigidTransform(rpy=RollPitchYaw([0, 0, np.pi / 2]), p=[0, ARM_Y_DEFAULT, 0])
X_W_LCB_DEFAULT = X_W_L_DEFAULT.GetAsMatrix4() @ np.linalg.inv(X_CB_B.GetAsMatrix4())

# The default pose of the right robot base when using the ROS URDFs
X_W_R_DEFAULT = RigidTransform(rpy=RollPitchYaw([0, 0, np.pi / 2]), p=[0, -ARM_Y_DEFAULT, 0])
X_W_RCB_DEFAULT = X_W_R_DEFAULT.GetAsMatrix4() @ np.linalg.inv(X_CB_B.GetAsMatrix4())


def add_cloth_competition_dual_ur5e_scene(
    robot_diagram_builder: RobotDiagramBuilder,
    X_W_LCB: HomogeneousMatrixType | None = None,
    X_W_RCB: HomogeneousMatrixType | None = None,
) -> tuple[tuple[ModelInstanceIndex, ModelInstanceIndex], tuple[ModelInstanceIndex, ModelInstanceIndex]]:
    parser = robot_diagram_builder.parser()
    parser.SetAutoRenaming(True)

    X_W_L = X_W_L_DEFAULT if X_W_LCB is None else RigidTransform(X_W_LCB @ X_CB_B.GetAsMatrix4())
    X_W_R = X_W_R_DEFAULT if X_W_RCB is None else RigidTransform(X_W_RCB @ X_CB_B.GetAsMatrix4())

    add_floor(robot_diagram_builder, y_size=2.4)

    # Add three safety walls
    arm_y = X_W_L.translation()[1]
    wall_thickness = 0.2
    wall_left_position = np.array([0, arm_y + 0.7 + wall_thickness / 2, 0])
    wall_right_position = np.array([0, -arm_y - 0.7 - wall_thickness / 2, 0])
    wall_back_position = np.array([0.9 + wall_thickness / 2, 0, 0])
    add_wall(robot_diagram_builder, "XZ", 2.0, 2.0, wall_thickness, wall_left_position, "wall_left")
    add_wall(robot_diagram_builder, "XZ", 2.0, 2.0, wall_thickness, wall_right_position, "wall_right")
    add_wall(robot_diagram_builder, "YZ", 2.0, 2.7, wall_thickness, wall_back_position, "wall_back")

    # The robot arms
    arm_left_index, gripper_left_index = add_manipulator(
        robot_diagram_builder, "ur5e", "robotiq_2f_85", X_W_L, static_gripper=True
    )
    arm_right_index, gripper_right_index = add_manipulator(
        robot_diagram_builder, "ur5e", "robotiq_2f_85", X_W_R, static_gripper=True
    )

    return (arm_left_index, arm_right_index), (gripper_left_index, gripper_right_index)


def add_cloth_obstacle_to_builder(
    robot_diagram_builder: RobotDiagramBuilder, point_cloud_cloth: PointCloud
) -> tuple[ModelInstanceIndex, o3d.t.geometry.TriangleMesh]:

    # Create mesh URDF of the convex hull of the cloth point cloud
    pcd = point_cloud_to_open3d(point_cloud_cloth)
    hull = pcd.compute_convex_hull()

    # Scaling disable for now because this might make it collide with that gripper that's holding it in the air
    # hull.scale(1.1, hull.get_center())  # make 10% larger

    hull_file = tempfile.NamedTemporaryFile(prefix="cloth_hull_", suffix=".obj", delete=False)
    hull_filename = hull_file.name
    o3d.t.io.write_triangle_mesh(hull_filename, hull)  # equiv? hull.write_obj(hull_filename)

    hull_urdf_path = mesh_urdf_path(hull_filename, "cloth_hull")

    # Add the cloth hull to the scene
    plant = robot_diagram_builder.plant()
    parser = robot_diagram_builder.parser()
    hull_index = parser.AddModels(hull_urdf_path)[0]

    world_frame = plant.world_frame()
    hull_frame = plant.GetFrameByName("base_link", hull_index)
    plant.WeldFrames(world_frame, hull_frame)

    return hull_index, hull


def add_safety_wall_to_builder(
    robot_diagram_builder: RobotDiagramBuilder, X_W_TCP: HomogeneousMatrixType
) -> ModelInstanceIndex:
    plant = robot_diagram_builder.plant()
    parser = robot_diagram_builder.parser()
    safety_wall_thickness = 0.05
    safety_wall_urdf_path = airo_models.box_urdf_path((0.2, safety_wall_thickness, 0.3), "safety_wall")

    # Position the safety wall in front of the left TCP
    shift = safety_wall_thickness / 2 + 0.01
    p_W_S = X_W_TCP[:3, 3] + X_W_TCP[:3, 2] * shift
    X_W_S = RigidTransform(p=p_W_S)

    # Add the cloth hull to the scene
    safety_wall_index = parser.AddModels(safety_wall_urdf_path)[0]
    world_frame = plant.world_frame()
    safety_wall_frame = plant.GetFrameByName("base_link", safety_wall_index)
    plant.WeldFrames(world_frame, safety_wall_frame, X_W_S)

    return safety_wall_index


def make_drake_scene(
    X_W_LCB: HomogeneousMatrixType,
    X_W_RCB: HomogeneousMatrixType,
    point_cloud: PointCloud = None,
    X_W_TCP: HomogeneousMatrixType = None,
) -> DualArmScene:
    robot_diagram_builder = RobotDiagramBuilder()

    meshcat = add_meshcat(robot_diagram_builder)
    meshcat.SetCameraPose([-1.5, 0, 1.0], [0, 0, 0])

    (arm_left_index, arm_right_index), (
        gripper_left_index,
        gripper_right_index,
    ) = add_cloth_competition_dual_ur5e_scene(robot_diagram_builder, X_W_LCB, X_W_RCB)

    if point_cloud is not None:
        point_cloud_cropped = crop_point_cloud(point_cloud, BBOX_CLOTH_IN_THE_AIR)
        add_cloth_obstacle_to_builder(robot_diagram_builder, point_cloud_cropped)

    if X_W_TCP is not None:
        add_safety_wall_to_builder(robot_diagram_builder, X_W_TCP)

    robot_diagram, _ = finish_build(robot_diagram_builder, meshcat)

    scene = DualArmScene(
        robot_diagram, arm_left_index, arm_right_index, gripper_left_index, gripper_right_index, meshcat
    )
    return scene


def set_dual_arm_joints(
    scene: DualArmScene, arm_left_joints: JointConfigurationType, arm_right_joints: JointConfigurationType
) -> None:
    """Set the joint positions of the dual arm scene to the given joint positions.

    Note that this is only for visualization.
    """
    robot_diagram = scene.robot_diagram
    context = robot_diagram.CreateDefaultContext()
    plant = robot_diagram.plant()
    plant_context = plant.GetMyContextFromRoot(context)
    plant.SetPositions(plant_context, scene.arm_left_index, arm_left_joints)
    plant.SetPositions(plant_context, scene.arm_right_index, arm_right_joints)
    robot_diagram.ForcedPublish(context)


def make_drake_scene_from_observation(
    observation: CompetitionObservation, include_cloth_obstacle=True
) -> DualArmScene:
    X_W_LCB = observation.arm_left_pose_in_world
    X_W_RCB = observation.arm_right_pose_in_world

    point_cloud = None
    X_W_TCP = None
    if include_cloth_obstacle:
        point_cloud = observation.point_cloud
        X_W_TCP = observation.arm_left_tcp_pose_in_world

    scene = make_drake_scene(X_W_LCB, X_W_RCB, point_cloud, X_W_TCP)

    set_dual_arm_joints(scene, observation.arm_left_joints, observation.arm_right_joints)
    return scene


def make_dual_arm_collision_checker(scene: DualArmScene) -> SceneGraphCollisionChecker:
    collision_checker = SceneGraphCollisionChecker(
        model=scene.robot_diagram,
        robot_model_instances=[
            scene.arm_left_index,
            scene.arm_right_index,
            scene.gripper_left_index,
            scene.gripper_right_index,
        ],
        edge_step_size=0.125,  # Arbitrary value: we don't use the CheckEdgeCollisionFree
        env_collision_padding=0.005,
        self_collision_padding=0.005,
    )
    return collision_checker
