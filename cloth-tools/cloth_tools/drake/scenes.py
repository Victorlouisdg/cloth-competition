import tempfile
from typing import Tuple

import airo_models
import numpy as np
import open3d as o3d
from airo_camera_toolkit.point_clouds.conversions import point_cloud_to_open3d
from airo_models import mesh_urdf_path
from airo_typing import HomogeneousMatrixType, PointCloud
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
    parser.SetAutoRenaming(True)

    # Load URDF files
    ur5e_urdf_path = airo_models.get_urdf_path("ur5e")
    robotiq_urdf_path = create_static_robotiq_2f_85_urdf()

    table_thickness = 0.2
    table_urdf_path = airo_models.box_urdf_path((2.0, 2.4, table_thickness), "table")

    arm_index = parser.AddModels(ur5e_urdf_path)[0]
    gripper_index = parser.AddModels(robotiq_urdf_path)[0]
    table_index = parser.AddModels(table_urdf_path)[0]

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
    parser.SetAutoRenaming(True)

    # Load URDF files
    ur5e_urdf_path = airo_models.get_urdf_path("ur5e")
    robotiq_urdf_path = create_static_robotiq_2f_85_urdf()

    table_thickness = 0.2
    table_urdf_path = airo_models.box_urdf_path((2.0, 2.4, table_thickness), "table")
    wall_thickness = 0.2
    wall_back_urdf_path = airo_models.box_urdf_path((wall_thickness, 2.7, 2.0), "wall_back")
    wall_left_urdf_path = airo_models.box_urdf_path((2.0, wall_thickness, 2.0), "wall_left")
    wall_right_urdf_path = airo_models.box_urdf_path((2.0, wall_thickness, 2.0), "wall_right")

    arm_left_index = parser.AddModels(ur5e_urdf_path)[0]
    arm_right_index = parser.AddModels(ur5e_urdf_path)[0]
    gripper_left_index = parser.AddModels(robotiq_urdf_path)[0]
    gripper_right_index = parser.AddModels(robotiq_urdf_path)[0]

    table_index = parser.AddModels(table_urdf_path)[0]
    wall_back_index = parser.AddModels(wall_back_urdf_path)[0]
    wall_left_index = parser.AddModels(wall_left_urdf_path)[0]
    wall_right_index = parser.AddModels(wall_right_urdf_path)[0]

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

    # Save hull attributes for Rerun visualization
    # hull.compute_vertex_normals()
    # self._hull_vertex_positions = hull.vertex.positions.numpy()
    # self._hull_vertex_normals = hull.vertex.normals.numpy()
    # self._hull_triangle_indices = hull.triangle.indices.numpy()

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
