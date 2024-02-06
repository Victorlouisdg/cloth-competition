import sys
import tempfile
from functools import partial
from typing import Any, Optional, Tuple

import cv2
import numpy as np
import rerun as rr
from airo_camera_toolkit.point_clouds.conversions import point_cloud_to_open3d
from airo_camera_toolkit.point_clouds.operations import crop_point_cloud
from airo_camera_toolkit.utils.image_converter import ImageConverter
from airo_models import mesh_urdf_path
from airo_typing import (
    BoundingBox3DType,
    HomogeneousMatrixType,
    OpenCVIntImageType,
    PointCloud,
    RotationMatrixType,
    Vector3DType,
)
from cloth_tools.bounding_boxes import BBOX_CLOTH_IN_THE_AIR, bbox_to_mins_and_sizes
from cloth_tools.controllers.controller import Controller
from cloth_tools.controllers.grasp_highest_controller import hang_in_the_air_tcp_pose
from cloth_tools.controllers.home_controller import HomeController
from cloth_tools.drake.building import add_meshcat_to_builder, finish_build
from cloth_tools.drake.scenes import add_dual_ur5e_and_table_to_builder
from cloth_tools.drake.visualization import publish_dual_arm_joint_path
from cloth_tools.ompl.dual_arm_planner import DualArmOmplPlanner
from cloth_tools.path.execution import calculate_dual_path_duration, execute_dual_arm_joint_path
from cloth_tools.point_clouds.camera import get_image_and_filtered_point_cloud
from cloth_tools.point_clouds.operations import lowest_point
from cloth_tools.stations.competition_station import CompetitionStation, inverse_kinematics_in_world_fn
from cloth_tools.stations.dual_arm_station import DualArmStation
from cloth_tools.visualization.opencv import draw_point_3d, draw_pose
from cloth_tools.visualization.rerun import rr_log_camera
from linen.elemental.move_backwards import move_pose_backwards
from loguru import logger
from pydrake.planning import RobotDiagramBuilder, SceneGraphCollisionChecker

import open3d as o3d  # isort:skip


def flatter_orientation(gripper_forward_direction: Vector3DType) -> RotationMatrixType:
    Z = gripper_forward_direction / np.linalg.norm(gripper_forward_direction)
    Y = np.array([0, 0, -1])  # 0, 0, 1 is also an option
    X = np.cross(Y, Z)
    return np.column_stack([X, Y, Z])


def lowest_point_grasp_pose(
    lowest_point: Vector3DType, grasp_depth: float = 0.08, height_offset: float = 0.025
) -> HomogeneousMatrixType:
    """TODO: docstring

    Args:
        lowest_point: The lowest point of the cloth.

    Returns:
        The grasp pose.
    """
    gripper_forward_direction = np.array([1, 0, 0])
    grasp_orientation = flatter_orientation(gripper_forward_direction)
    grasp_location = lowest_point.copy()
    grasp_location += gripper_forward_direction * grasp_depth
    grasp_location[2] += height_offset  # Move the TCP pose a bit up into the cloth
    grasp_pose = np.identity(4)
    grasp_pose[0:3, 0:3] = grasp_orientation
    grasp_pose[0:3, 3] = grasp_location
    return grasp_pose


class GraspLowestController(Controller):
    """
    Grasps the lowest point of a piece of cloth and lift the cloth into the air.

    Currently always uses the left arm to grasp.
    """

    def __init__(self, station: DualArmStation, bbox: BoundingBox3DType):
        self.station = station
        self.bbox = bbox

        self.joint_speed = 1.0  # rad/s

        # Attributes that will be set in plan()
        self._image: Optional[OpenCVIntImageType] = None
        self._grasp_pose: Optional[HomogeneousMatrixType] = None
        self._point_cloud: Optional[PointCloud] = None
        self._lowest_point: Optional[Vector3DType] = None
        self._pregrasp_pose: Optional[HomogeneousMatrixType] = None
        self._path_pregrasp: Optional[Any] = None
        self._path_right_home: Optional[Any] = None
        self._path_left_hang: Optional[Any] = None
        self._hull_vertex_positions: Optional[np.ndarray] = None
        self._hull_vertex_normals: Optional[np.ndarray] = None
        self._hull_triangle_indices: Optional[np.ndarray] = None

        camera = self.station.camera
        camera_pose = self.station.camera_pose

        # Setting up the OpenCV window
        window_name = self.__class__.__name__
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 1280, 720)

        # Setting up Rerun
        rr.init(window_name, spawn=True)
        rr_log_camera(camera, camera_pose)
        bbox_color = (122, 173, 255)  # blue
        bbox_mins, bbox_sizes = bbox_to_mins_and_sizes(bbox)
        rr_bbox = rr.Boxes3D(mins=bbox_mins, sizes=bbox_sizes, colors=bbox_color)
        rr.log("world/bbox", rr_bbox)

    def execute_handover(self) -> None:
        dual_arm = self.station.dual_arm

        # Execute the path to the pregrasp pose
        execute_dual_arm_joint_path(dual_arm, self._path_pregrasp, self.joint_speed)

        # Execute the grasp
        dual_arm.move_linear_to_tcp_pose(self._grasp_pose, None, linear_speed=0.2).wait()
        dual_arm.left_manipulator.gripper.close().wait()
        dual_arm.move_linear_to_tcp_pose(self._pregrasp_pose, None, linear_speed=0.2).wait()

        # Open the right gripper of the cloth be released
        dual_arm.right_manipulator.gripper.open().wait()
        execute_dual_arm_joint_path(dual_arm, self._path_right_home, self.joint_speed)

        # Move the left arm to the hang pose
        execute_dual_arm_joint_path(dual_arm, self._path_left_hang, self.joint_speed)

    def _create_cloth_obstacle_planner(self, point_cloud_cloth: PointCloud) -> DualArmOmplPlanner:
        """Very similar to the method in CompetitionStation, but with the cloth as a runtime obstacle."""

        X_W_LCB = self.station.left_arm_pose
        X_W_RCB = self.station.right_arm_pose

        robot_diagram_builder = RobotDiagramBuilder()
        meshcat = add_meshcat_to_builder(robot_diagram_builder)
        arm_indices, gripper_indices = add_dual_ur5e_and_table_to_builder(robot_diagram_builder, X_W_LCB, X_W_RCB)

        pcd = point_cloud_to_open3d(point_cloud_cloth)
        hull = pcd.compute_convex_hull()
        hull.scale(1.1, hull.get_center())  # make 10% larger

        hull_file = tempfile.NamedTemporaryFile(prefix="cloth_hull_", suffix=".obj", delete=False)
        hull_filename = hull_file.name
        o3d.t.io.write_triangle_mesh(hull_filename, hull)  # equiv? hull.write_obj(hull_filename)

        hull_urdf_path = mesh_urdf_path(hull_filename, "cloth_hull")

        # Save hull attributes for Rerun visualization
        hull.compute_vertex_normals()
        self._hull_vertex_positions = hull.vertex.positions.numpy()
        self._hull_vertex_normals = hull.vertex.normals.numpy()
        self._hull_triangle_indices = hull.triangle.indices.numpy()

        # Add the cloth hull to the scene
        plant = robot_diagram_builder.plant()
        parser = robot_diagram_builder.parser()
        hull_index = parser.AddModels(hull_urdf_path)[0]
        world_frame = plant.world_frame()
        hull_frame = plant.GetFrameByName("base_link", hull_index)
        plant.WeldFrames(world_frame, hull_frame)

        diagram, context = finish_build(robot_diagram_builder, meshcat)

        collision_checker = SceneGraphCollisionChecker(
            model=diagram,
            robot_model_instances=[*arm_indices, *gripper_indices],
            edge_step_size=0.125,  # Arbitrary value: we don't use the CheckEdgeCollisionFree
            env_collision_padding=0.005,
            self_collision_padding=0.005,
        )

        is_state_valid_fn = collision_checker.CheckConfigCollisionFree

        inverse_kinematics_left_fn = partial(inverse_kinematics_in_world_fn, X_W_CB=X_W_LCB)
        inverse_kinematics_right_fn = partial(inverse_kinematics_in_world_fn, X_W_CB=X_W_RCB)

        # expose these things for visualization
        self._diagram = diagram
        self._context = context
        self._collision_checker = collision_checker
        self._meshcat = meshcat
        self._arm_indices = arm_indices
        self._gripper_indices = gripper_indices

        planner = DualArmOmplPlanner(
            is_state_valid_fn,
            inverse_kinematics_left_fn,
            inverse_kinematics_right_fn,
            self.station.joint_bounds_left,
            self.station.joint_bounds_right,
        )

        return planner

    def plan(self) -> None:
        logger.info(f"{self.__class__.__name__}: Creating new plan.")

        camera = self.station.camera
        camera_pose = self.station.camera_pose

        image_rgb, point_cloud = get_image_and_filtered_point_cloud(camera, camera_pose)
        image = ImageConverter.from_numpy_int_format(image_rgb).image_in_opencv_format

        self._image = image
        self._point_cloud = point_cloud

        point_cloud_cropped = crop_point_cloud(point_cloud, self.bbox)

        if len(point_cloud_cropped.points) == 0:
            self._lowest_point = None
            self._grasp_pose = None
            return

        lowest_point_ = lowest_point(point_cloud_cropped.points)
        grasp_pose = lowest_point_grasp_pose(lowest_point_)

        self._lowest_point = lowest_point_
        self._grasp_pose = grasp_pose

        logger.info(f"Found lowest point in bbox at: {lowest_point_}")

        pregrasp_pose = move_pose_backwards(grasp_pose, 0.15)
        self._pregrasp_pose = pregrasp_pose

        # Create planner with cloth obstacle
        planner_cloth = self._create_cloth_obstacle_planner(point_cloud_cropped)

        dual_arm = self.station.dual_arm
        start_joints_left = dual_arm.left_manipulator.get_joint_configuration()
        start_joints_right = dual_arm.right_manipulator.get_joint_configuration()

        # For debugging
        # plant = self._diagram.plant()
        # plant_context = plant.GetMyContextFromRoot(self._context)
        # plant.SetPositions(plant_context, self._arm_indices[1], start_joints_right.squeeze())

        # X_W_LCB = self.station.left_arm_pose
        # inverse_kinematics_left_fn = partial(inverse_kinematics_in_world_fn, X_W_CB=X_W_LCB)
        # ik_solutions = inverse_kinematics_left_fn(pregrasp_pose)
        # publish_ik_solutions(ik_solutions, 2.0, self._meshcat, self._diagram, self._context, self._arm_indices[0])
        # input("Press Enter to continue...")

        path_pregrasp = planner_cloth.plan_to_tcp_pose(start_joints_left, start_joints_right, pregrasp_pose, None)
        self._path_pregrasp = path_pregrasp

        if path_pregrasp is None:
            return

        # Here the right arm opens its gripper and moves to its home position
        # Plan for the right arm to move home (don't consider obstacles)
        planner = self.station.planner

        pregrasp_joints_left = path_pregrasp[-1][0]

        home_joints_right = self.station.home_joints_right
        path_right_home = planner.plan_to_joint_configuration(
            pregrasp_joints_left, start_joints_right, None, home_joints_right
        )
        self._path_right_home = path_right_home

        # Plan for the left arm to move to the hang pose
        hang_pose = hang_in_the_air_tcp_pose(left=True)
        path_left_hang = planner.plan_to_tcp_pose(pregrasp_joints_left, home_joints_right, hang_pose, None)
        self._path_left_hang = path_left_hang

    def _can_execute(self) -> bool:
        # maybe this should just be a property?
        if self._path_pregrasp is None:
            return False

        if self._path_right_home is None:
            return False

        if self._path_left_hang is None:
            return False

        return True

    def visualize_plan(self) -> Tuple[OpenCVIntImageType, Any]:
        if self._image is None:
            raise RuntimeError("You must call plan() before visualize_plan().")

        image = self._image
        camera_pose = self.station.camera_pose
        intrinsics = self.station.camera.intrinsics_matrix()

        if self._lowest_point is not None:
            lowest = self._lowest_point
            draw_point_3d(image, lowest, intrinsics, camera_pose, (0, 255, 0))

            rr.log("world/lowest_point", rr.Points3D(positions=[lowest], colors=[(0, 1, 0)], radii=0.02))

        if self._grasp_pose is not None:
            grasp_pose = self._grasp_pose
            draw_pose(image, grasp_pose, intrinsics, camera_pose)

            rr_grasp_pose = rr.Transform3D(translation=grasp_pose[0:3, 3], mat3x3=grasp_pose[0:3, 0:3])
            rr.log("world/grasp_pose", rr_grasp_pose)

        if self._pregrasp_pose is not None:
            pregrasp_pose = self._pregrasp_pose
            # draw_pose(image, pregrasp_pose, intrinsics, camera_pose)

            rr_pregrasp_pose = rr.Transform3D(translation=pregrasp_pose[0:3, 3], mat3x3=pregrasp_pose[0:3, 0:3])
            rr.log("world/pregrasp_pose", rr_pregrasp_pose)

        if self._path_pregrasp is not None:
            path = self._path_pregrasp
            duration = calculate_dual_path_duration(path, self.joint_speed)
            publish_dual_arm_joint_path(
                path, duration, self._meshcat, self._diagram, self._context, *self._arm_indices
            )

        if self._path_left_hang is not None:
            path = self._path_left_hang
            station = self.station
            duration = calculate_dual_path_duration(path, self.joint_speed)
            publish_dual_arm_joint_path(
                path, duration, station._meshcat, station._diagram, station._context, *station._arm_indices
            )

        if self._point_cloud is not None:
            point_cloud = self._point_cloud
            rr_point_cloud = rr.Points3D(positions=point_cloud.points, colors=point_cloud.colors)
            rr.log("world/point_cloud", rr_point_cloud)

        if self._hull_vertex_positions is not None:
            hull_color = (1.0, 0.2, 0.0, 0.5)

            rr_mesh_material = rr.Material(hull_color)
            rr_mesh = rr.Mesh3D(
                vertex_positions=self._hull_vertex_positions,
                vertex_normals=self._hull_vertex_normals,
                indices=self._hull_triangle_indices,
                mesh_material=rr_mesh_material,
            )

            rr.log("world/hull", rr_mesh)

        # Duplicated from GraspHighestController
        if self._can_execute():
            image_annotated = image.copy()
            text = "Execute? Press (y/n)"
            cv2.putText(image_annotated, text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3, cv2.LINE_AA)
            logger.info(f"{self.__class__.__name__}: {text} in OpenCV window.")
            cv2.imshow(self.__class__.__name__, image_annotated)
            key = cv2.waitKey(0)
            return image, key
        else:
            cv2.imshow(self.__class__.__name__, image)
            return image, cv2.waitKey(1)

    def execute_plan(self) -> None:
        # TODO bring execute_handover() here
        if self._grasp_pose is None:
            logger.info("Grasp and hang not executed because no grasp pose was found.")
            return
        self.execute_handover()

    # TODO: remove this duplication, maybe through inheritance?
    def execute_interactive(self) -> None:
        while True:
            self.plan()
            _, key = self.visualize_plan()
            if key == ord("p"):
                key = cv2.waitKey(0)
            elif key == ord("y"):
                self.execute_plan()
                return
            elif key == ord("n"):
                continue
            elif key == ord("q"):
                sys.exit(0)

    def execute(self, interactive: bool = True) -> None:
        logger.info(f"{self.__class__.__name__} started.")

        if interactive:
            self.execute_interactive()
        else:
            # Autonomous execution
            self.plan()
            self.visualize_plan()
            self.execute_plan()

        # Close cv2 window to reduce clutter
        cv2.destroyWindow(self.__class__.__name__)

        logger.info(f"{self.__class__.__name__} finished.")


if __name__ == "__main__":
    station = CompetitionStation()
    dual_arm = station.dual_arm

    # Move only left arm to home. right might be holding the cloth
    home_controller = HomeController(station, move_right_home=False, open_right_gripper=False)
    home_controller.execute(interactive=True)

    # Assumes the right arm is holding the cloth in the air
    grasp_lowest_controller = GraspLowestController(station, BBOX_CLOTH_IN_THE_AIR)
    grasp_lowest_controller.execute(interactive=True)
