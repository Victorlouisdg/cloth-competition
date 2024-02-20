import sys
from pathlib import Path
from typing import Any, Optional, Tuple

import cv2
import numpy as np
import rerun as rr
from airo_camera_toolkit.point_clouds.operations import crop_point_cloud
from airo_camera_toolkit.utils.image_converter import ImageConverter
from airo_typing import BoundingBox3DType, HomogeneousMatrixType, OpenCVIntImageType, PointCloud
from cloth_tools.annotation.grasp_annotation import get_manual_grasp_annotation, save_grasp_info
from cloth_tools.bounding_boxes import BBOX_CLOTH_IN_THE_AIR, bbox_to_mins_and_sizes
from cloth_tools.controllers.controller import Controller
from cloth_tools.controllers.grasp_lowest_controller import create_cloth_obstacle_planner
from cloth_tools.controllers.home_controller import HomeController
from cloth_tools.drake.visualization import publish_dual_arm_trajectory, publish_ik_solutions
from cloth_tools.ompl.dual_arm_planner import DualArmOmplPlanner
from cloth_tools.path.execution import execute_dual_arm_trajectory, time_parametrize_toppra
from cloth_tools.point_clouds.camera import get_image_and_filtered_point_cloud
from cloth_tools.stations.competition_station import CompetitionStation
from cloth_tools.stations.dual_arm_station import DualArmStation
from cloth_tools.visualization.opencv import draw_pose
from cloth_tools.visualization.rerun import rr_log_camera
from linen.elemental.move_backwards import move_pose_backwards
from loguru import logger

import open3d as o3d  # isort:skip


class GraspHangingController(Controller):
    """
    Grasps the lowest point of a piece of cloth and lift the cloth into the air.

    Currently always uses the left arm to grasp.
    """

    GOOD_GRASP_JOINTS_RIGHT_0 = np.deg2rad([-124, -124, -75.5, -164, 56, 180])  # recorded in freedrive

    def __init__(self, station: DualArmStation, bbox: BoundingBox3DType, sample_dir: str = None):
        self.station = station
        self.bbox = bbox

        self.sample_dir = sample_dir  # Quick fix for data collection, saving grasp will be done elsewhere later

        # Attributes that will be set in plan()
        self._image: Optional[OpenCVIntImageType] = None
        self._grasp_pose: Optional[HomogeneousMatrixType] = None
        self._point_cloud: Optional[PointCloud] = None
        self._pregrasp_pose: Optional[HomogeneousMatrixType] = None
        self._path_pregrasp: Optional[Any] = None
        self._trajectory_pregrasp: Optional[Any] = None
        self._time_trajectory_pregrasp: Optional[Any] = None

        self._hull: Optional[o3d.t.geometry.TriangleMesh] = None
        self._planner_cloth: Optional[DualArmOmplPlanner] = None
        self._start_joints_left: Optional[Any] = None

        self._diagram: Optional[Any] = None
        self._context: Optional[Any] = None
        self._collision_checker: Optional[Any] = None
        self._meshcat: Optional[Any] = None
        self._arm_indices: Optional[Tuple[int, int]] = None
        self._gripper_indices: Optional[Tuple[int, int]] = None

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

    def _create_cloth_obstacle_planner(self, point_cloud_cloth: PointCloud) -> DualArmOmplPlanner:
        (
            planner,
            hull,
            diagram,
            context,
            collision_checker,
            meshcat,
            arm_indices,
            gripper_indices,
        ) = create_cloth_obstacle_planner(self.station, point_cloud_cloth, left_hanging=True)
        # Save all this for visualization
        self._hull = hull
        self._diagram = diagram
        self._context = context
        self._collision_checker = collision_checker
        self._meshcat = meshcat
        self._arm_indices = arm_indices
        self._gripper_indices = gripper_indices

        return planner

    def plan(self) -> None:
        logger.info(f"{self.__class__.__name__}: Creating new plan.")

        camera = self.station.camera
        camera_pose = self.station.camera_pose

        image_rgb, depth_map, point_cloud = get_image_and_filtered_point_cloud(camera, camera_pose)
        image = ImageConverter.from_numpy_int_format(image_rgb).image_in_opencv_format

        self._image = image
        self._point_cloud = point_cloud

        point_cloud_cropped = crop_point_cloud(point_cloud, self.bbox)

        if len(point_cloud_cropped.points) == 0:
            logger.info("No points in the cropped point cloud.")
            self._grasp_pose = None
            return

        grasp_info = get_manual_grasp_annotation(
            image_rgb, depth_map, point_cloud, self.station.camera_pose, camera.intrinsics_matrix(), log_to_rerun=True
        )
        grasp_pose = grasp_info.grasp_pose

        self._grasp_info = grasp_info  # Save for data collection
        self._grasp_pose = grasp_pose

        if grasp_pose is None:
            logger.info("Recieved no grasp pose.")
            self._pregrasp_pose = None
            self._path_pregrasp = None
            self._trajectory_pregrasp = None
            self._time_trajectory_pregrasp = None
            return

        # DETERMINE GRASP POSE HERE
        planner_cloth = self._create_cloth_obstacle_planner(point_cloud_cropped)
        self._planner_cloth = planner_cloth  # Save for debugging

        # Create planner with cloth obstacle

        dual_arm = self.station.dual_arm
        start_joints_left = dual_arm.left_manipulator.get_joint_configuration()
        start_joints_right = dual_arm.right_manipulator.get_joint_configuration()

        # Try moving pregrasp pose to several distances from the grasp pose
        distances_to_try = [0.15, 0.20, 0.1, 0.25, 0.05, 0.01]
        for d in distances_to_try:
            pregrasp_pose = move_pose_backwards(grasp_pose, d)
            self._pregrasp_pose = pregrasp_pose

            self._start_joints_left = start_joints_left

            path_pregrasp = planner_cloth.plan_to_tcp_pose(
                start_joints_left,
                start_joints_right,
                None,
                pregrasp_pose,
                desirable_goal_configurations_right=[
                    self.GOOD_GRASP_JOINTS_RIGHT_0,
                ],  # Try to avoid the shoulder from pointing towards the camera
            )
            self._path_pregrasp = path_pregrasp

            if path_pregrasp is not None:
                logger.info(f"Found path to pregrasp pose with distance {d}.")

                trajectory_pregrasp, time_trajectory_pregrasp = time_parametrize_toppra(
                    path_pregrasp, self._diagram.plant()
                )
                self._trajectory_pregrasp = trajectory_pregrasp
                self._time_trajectory_pregrasp = time_trajectory_pregrasp
                break
            else:
                logger.info(f"No path to pregrasp pose with distance {d}.")

        if path_pregrasp is None:
            logger.info("No path to any of the tried pregrasp poses found.")
            # Maybe reset all state at the beginning of each plan call?
            self._path_pregrasp = None
            self._trajectory_pregrasp = None
            self._time_trajectory_pregrasp = None
            return

    def execute_grasp(self) -> None:
        if self.sample_dir is not None:
            # Save the grasp we are about to execute
            sample_dir = self.sample_dir
            grasp_info = self._grasp_info
            grasp_dir = Path(sample_dir) / "grasp"
            save_grasp_info(str(grasp_dir), grasp_info)

        dual_arm = self.station.dual_arm

        # Execute the path to the pregrasp pose
        execute_dual_arm_trajectory(dual_arm, self._trajectory_pregrasp, self._time_trajectory_pregrasp)

        # Execute the grasp
        dual_arm.move_linear_to_tcp_pose(None, self._grasp_pose, linear_speed=0.2).wait()
        dual_arm.right_manipulator.gripper.close().wait()

    def _can_execute(self) -> bool:
        # maybe this should just be a property?
        if self._path_pregrasp is None:
            return False

        return True

    def visualize_plan(self) -> Tuple[OpenCVIntImageType, Any]:  # noqa C901
        if self._image is None:
            raise RuntimeError("You must call plan() before visualize_plan().")

        image = self._image
        camera_pose = self.station.camera_pose
        intrinsics = self.station.camera.intrinsics_matrix()

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
            trajectory = self._trajectory_pregrasp
            time_trajectory = self._time_trajectory_pregrasp
            publish_dual_arm_trajectory(
                trajectory,
                time_trajectory,
                self._meshcat,
                self._diagram,
                self._context,
                *self._arm_indices,
            )
        else:
            # Publish IK solutions for debugging
            if self._planner_cloth is not None:
                if self._planner_cloth._single_arm_planner_right is not None:
                    if self._planner_cloth._single_arm_planner_right._ik_solutions is not None:
                        logger.info("Publishing IK solutions for right arm.")

                        # Set positiosn of left arm first
                        plant = self._diagram.plant()
                        plant_context = plant.GetMyContextFromRoot(self._context)
                        plant.SetPositions(plant_context, self._arm_indices[0], self._start_joints_left)

                        publish_ik_solutions(
                            self._planner_cloth._single_arm_planner_right._ik_solutions,
                            2.0,
                            self._meshcat,
                            self._diagram,
                            self._context,
                            self._arm_indices[1],
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
            logger.info(f"{self.__class__.__name__}: Execution not possible yet. Press any key to plan again.")
            return image, cv2.waitKey(0)

    def execute_plan(self) -> None:
        # TODO bring execute_handover() here
        if self._grasp_pose is None:
            logger.info("Grasp not executed because no grasp pose was found.")
            return

        if not self._can_execute():
            logger.info("Grasp not executed because the plan is not complete.")
            return

        self.execute_grasp()

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
    home_controller = HomeController(station, move_left_home=False, open_left_gripper=False)
    home_controller.execute(interactive=True)

    # Assumes the right arm is holding the cloth in the air
    grasp_hanging_controller = GraspHangingController(station, BBOX_CLOTH_IN_THE_AIR)
    grasp_hanging_controller.execute(interactive=True)
