import sys
from functools import partial
from typing import Any

import cv2
import numpy as np
import rerun as rr
from airo_camera_toolkit.point_clouds.operations import crop_point_cloud
from airo_camera_toolkit.utils.image_converter import ImageConverter
from airo_drake import (
    DualArmScene,
    add_meshcat,
    animate_dual_joint_trajectory,
    concatenate_drake_trajectories,
    finish_build,
    time_parametrize_toppra,
)
from airo_planner import DualArmOmplPlanner, PlannerError, rank_by_distance_to_desirable_configurations, stack_joints
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
from cloth_tools.drake.scenes import (
    add_cloth_competition_dual_ur5e_scene,
    add_cloth_obstacle_to_builder,
    add_safety_wall_to_builder,
)
from cloth_tools.kinematics.constants import TCP_TRANSFORM
from cloth_tools.point_clouds.camera import get_image_and_filtered_point_cloud
from cloth_tools.point_clouds.operations import lowest_point
from cloth_tools.stations.competition_station import CompetitionStation, inverse_kinematics_in_world_fn
from cloth_tools.stations.dual_arm_station import DualArmStation
from cloth_tools.trajectory_execution import execute_dual_arm_drake_trajectory
from cloth_tools.visualization.opencv import draw_point_3d, draw_pose
from cloth_tools.visualization.rerun import rr_log_camera
from linen.elemental.move_backwards import move_pose_backwards
from loguru import logger
from pydrake.planning import RobotDiagramBuilder, SceneGraphCollisionChecker
from pydrake.trajectories import Trajectory

import open3d as o3d  # isort:skip


def flatter_orientation(gripper_forward_direction: Vector3DType) -> RotationMatrixType:
    Z = gripper_forward_direction / np.linalg.norm(gripper_forward_direction)
    Y = np.array([0, 0, -1])  # 0, 0, 1 is also an option
    X = np.cross(Y, Z)
    return np.column_stack([X, Y, Z])


def lowest_point_grasp_pose(
    lowest_point: Vector3DType, grasp_depth: float = 0.1, height_offset: float = 0.025
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


def create_cloth_obstacle_planner(
    station: CompetitionStation, point_cloud_cloth: PointCloud, left_hanging: bool = True
) -> tuple[DualArmOmplPlanner, DualArmScene, o3d.t.geometry.TriangleMesh]:
    X_W_LCB = station.left_arm_pose
    X_W_RCB = station.right_arm_pose

    if left_hanging:
        X_LCB_TCP = station.dual_arm.left_manipulator.get_tcp_pose()
        X_W_TCP = X_W_LCB @ X_LCB_TCP
    else:
        X_RCB_TCP = station.dual_arm.right_manipulator.get_tcp_pose()
        X_W_TCP = X_W_RCB @ X_RCB_TCP

    robot_diagram_builder = RobotDiagramBuilder()

    meshcat = add_meshcat(robot_diagram_builder)
    meshcat.SetCameraPose([-1.0, 0, 1.0], [0, 0, 0])

    (arm_left_index, arm_right_index), (
        gripper_left_index,
        gripper_right_index,
    ) = add_cloth_competition_dual_ur5e_scene(robot_diagram_builder, X_W_LCB, X_W_RCB)

    _, hull = add_cloth_obstacle_to_builder(robot_diagram_builder, point_cloud_cloth)
    add_safety_wall_to_builder(robot_diagram_builder, X_W_TCP)

    robot_diagram, _ = finish_build(robot_diagram_builder, meshcat)

    scene = DualArmScene(
        robot_diagram, arm_left_index, arm_right_index, gripper_left_index, gripper_right_index, meshcat
    )

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

    inverse_kinematics_left_fn = partial(inverse_kinematics_in_world_fn, X_W_CB=X_W_LCB, tcp_transform=TCP_TRANSFORM)
    inverse_kinematics_right_fn = partial(inverse_kinematics_in_world_fn, X_W_CB=X_W_RCB, tcp_transform=TCP_TRANSFORM)

    planner = DualArmOmplPlanner(
        is_state_valid_fn=collision_checker.CheckConfigCollisionFree,
        inverse_kinematics_left_fn=inverse_kinematics_left_fn,
        inverse_kinematics_right_fn=inverse_kinematics_right_fn,
        joint_bounds_left=station.joint_bounds_left,
        joint_bounds_right=station.joint_bounds_right,
    )
    return planner, scene, hull


class GraspLowestController(Controller):
    """
    Grasps the lowest point of a piece of cloth and lift the cloth into the air.

    Currently always uses the left arm to grasp.
    """

    def __init__(self, station: DualArmStation, bbox: BoundingBox3DType):
        self.station = station
        self.bbox = bbox

        # Attributes that will be set in plan()
        self._image: OpenCVIntImageType | None = None
        self._grasp_pose: HomogeneousMatrixType | None = None
        self._point_cloud: PointCloud | None = None
        self._lowest_point: Vector3DType | None = None
        self._pregrasp_pose: HomogeneousMatrixType | None = None

        self._trajectory_pregrasp: Trajectory | None = None
        self._trajectory_right_home: Trajectory | None = None
        self._trajectory_hang_left: Trajectory | None = None
        self._hull: o3d.t.geometry.TriangleMesh | None = None

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
        execute_dual_arm_drake_trajectory(dual_arm, self._trajectory_pregrasp)

        # Execute the grasp
        dual_arm.move_linear_to_tcp_pose(self._grasp_pose, None, linear_speed=0.2).wait()
        dual_arm.left_manipulator.gripper.close().wait()
        dual_arm.move_linear_to_tcp_pose(self._pregrasp_pose, None, linear_speed=0.2).wait()

        # Open the right gripper of the cloth be released
        dual_arm.right_manipulator.gripper.open().wait()
        execute_dual_arm_drake_trajectory(dual_arm, self._trajectory_right_home)

        # Move the left arm to the hang pose
        execute_dual_arm_drake_trajectory(dual_arm, self._trajectory_hang_left)

    def _create_cloth_obstacle_planner(self, point_cloud_cloth: PointCloud) -> DualArmOmplPlanner:
        planner, scene, hull = create_cloth_obstacle_planner(self.station, point_cloud_cloth, left_hanging=False)
        # Save this for visualization
        self._hull = hull
        self.drake_scene = scene

        return planner

    def plan(self) -> None:
        logger.info(f"{self.__class__.__name__}: Creating new plan.")

        camera = self.station.camera
        camera_pose = self.station.camera_pose

        image_rgb, _, point_cloud = get_image_and_filtered_point_cloud(camera, camera_pose)
        image = ImageConverter.from_numpy_int_format(image_rgb).image_in_opencv_format

        self._image = image
        self._point_cloud = point_cloud

        point_cloud_cropped = crop_point_cloud(point_cloud, self.bbox)

        if len(point_cloud_cropped.points) == 0:
            logger.info("No points in the cropped point cloud.")
            self._lowest_point = None
            self._grasp_pose = None
            return

        lowest_point_ = lowest_point(point_cloud_cropped.points)
        grasp_pose = lowest_point_grasp_pose(lowest_point_)

        self._lowest_point = lowest_point_
        self._grasp_pose = grasp_pose

        logger.info(f"Found lowest point in bbox at: {lowest_point_}")

        planner_with_cloth_obstacle = self._create_cloth_obstacle_planner(point_cloud_cropped)
        plant = self.station.drake_scene.robot_diagram.plant()  # For use with TOPP-RA

        # Create planner with cloth obstacle
        dual_arm = self.station.dual_arm
        start_joints_left = dual_arm.left_manipulator.get_joint_configuration()
        start_joints_right = dual_arm.right_manipulator.get_joint_configuration()

        # Try to avoid the shoulder from pointing towards the camera
        # This can in rare cases trigger the robot to go into protective stop when grasping
        # due to risk of clamping.
        desirable_configurations = [stack_joints(self.station.home_joints_left, start_joints_right)]
        rank_fn = partial(
            rank_by_distance_to_desirable_configurations, desirable_configurations=desirable_configurations
        )
        planner_with_cloth_obstacle.rank_goal_configurations_fn = rank_fn

        # Try moving pregrasp pose to several distances from the grasp pose
        path_pregrasp = None  # To prevent UnboundLocalError
        distances_to_try = [0.15, 0.20, 0.1, 0.25, 0.05, 0.01]
        for distance in distances_to_try:
            logger.info(f"Planning to pregrasp pose at distance {distance}.")
            pregrasp_pose = move_pose_backwards(grasp_pose, distance)
            self._pregrasp_pose = pregrasp_pose

            try:
                path_pregrasp = planner_with_cloth_obstacle.plan_to_tcp_pose(
                    start_joints_left,
                    start_joints_right,
                    pregrasp_pose,
                    None,
                    # desirable_goal_configurations_left=[
                    #     self.station.home_joints_left
                    # ],
                )
                break
            except PlannerError:
                logger.info(f"No path found to pregrasp pose with distance {distance}.")
                continue

        if path_pregrasp is None:
            logger.warning("Failed to create path to any of the tried pregrasp poses.")
            self._trajectory_pregrasp = None
            self._trajectory_right_home = None
            self._trajectory_hang_left = None
            return

        trajectory_pregrasp = time_parametrize_toppra(plant, path_pregrasp)
        self._trajectory_pregrasp = trajectory_pregrasp

        # Here the right arm opens its gripper and moves to its home position
        # Plan for the right arm to move home (don't consider obstacles)
        planner = self.station.planner

        # we moveL back to the pregrasp pose after grasping
        pregrasp_joints_left = path_pregrasp[-1][:6]

        home_joints_right = self.station.home_joints_right
        path_home_right = planner.plan_to_joint_configuration(
            pregrasp_joints_left, start_joints_right, None, home_joints_right
        )
        trajectory_right_home = time_parametrize_toppra(plant, path_home_right)

        self._trajectory_right_home = trajectory_right_home

        home_joints_wrist_flipped_left = self.station.home_joints_left.copy()
        home_joints_wrist_flipped_left[5] = -home_joints_wrist_flipped_left[5]

        desirable_configurations_hang = [stack_joints(self.station.home_joints_left, home_joints_wrist_flipped_left)]
        rank_fn_hang = partial(
            rank_by_distance_to_desirable_configurations, desirable_configurations=desirable_configurations_hang
        )

        old_rank_fn = planner.rank_goal_configurations_fn
        planner.rank_goal_configurations_fn = rank_fn_hang

        # Plan for the left arm to move to the hang pose
        hang_pose = hang_in_the_air_tcp_pose(left=True)
        path_hang_left = planner.plan_to_tcp_pose(
            pregrasp_joints_left,
            home_joints_right,
            hang_pose,
            None,
            # desirable_goal_configurations_left=[self.station.home_joints_left, home_joints_wrist_flipped_left],
        )

        # reset this for future controller that might use it, might be better to have a separate planner for each controller
        planner.rank_goal_configurations_fn = old_rank_fn

        self._path_hang_left = path_hang_left

        # Limit the acceleration of the joints to avoid the cloth swinging too much
        trajectory_hang_left = time_parametrize_toppra(plant, path_hang_left, joint_acceleration_limit=0.5)
        self._trajectory_hang_left = trajectory_hang_left

    def _can_execute(self) -> bool:
        # maybe this should just be a property?
        if self._trajectory_pregrasp is None:
            return False

        if self._trajectory_right_home is None:
            return False

        if self._trajectory_hang_left is None:
            return False

        return True

    def visualize_plan(self) -> tuple[OpenCVIntImageType, Any]:
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

        if (
            self._trajectory_pregrasp is not None
            and self._trajectory_right_home is not None
            and self._trajectory_hang_left is not None
        ):
            scene = self.drake_scene
            trajectory_concatenated = concatenate_drake_trajectories(
                [self._trajectory_pregrasp, self._trajectory_right_home, self._trajectory_hang_left]
            )
            animate_dual_joint_trajectory(
                scene.meshcat,
                scene.robot_diagram,
                scene.arm_left_index,
                scene.arm_right_index,
                trajectory_concatenated,
            )

        if self._point_cloud is not None:
            point_cloud = self._point_cloud
            rr_point_cloud = rr.Points3D(positions=point_cloud.points, colors=point_cloud.colors)
            rr.log("world/point_cloud", rr_point_cloud)

        if self._hull is not None:
            self._hull.compute_vertex_normals()
            hull_vertex_positions = self._hull.vertex.positions.numpy()
            hull_vertex_normals = self._hull.vertex.normals.numpy()
            hull_triangle_indices = self._hull.triangle.indices.numpy()
            hull_color = (1.0, 0.2, 0.0, 0.5)
            rr_mesh_material = rr.Material(hull_color)
            rr_mesh = rr.Mesh3D(
                vertex_positions=hull_vertex_positions,
                vertex_normals=hull_vertex_normals,
                indices=hull_triangle_indices,
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

        if not self._can_execute():
            logger.info("Grasp and hang not executed because the plan is not complete.")
            return

        self.execute_handover()

    # TODO: remove this duplication, maybe through inheritance?
    def execute_interactive(self) -> None:
        while True:
            self.plan()
            _, key = self.visualize_plan()
            if key == ord("p"):
                key = cv2.waitKey(0)
            elif key == ord("y") and self._can_execute():
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
            while not self._can_execute():
                self.plan()
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
