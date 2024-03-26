import sys
from functools import partial
from pathlib import Path
from typing import Any, Tuple

import cv2
import rerun as rr
from airo_camera_toolkit.point_clouds.operations import crop_point_cloud
from airo_camera_toolkit.utils.image_converter import ImageConverter
from airo_drake import DualArmScene, animate_dual_joint_trajectory
from airo_planner import DualArmOmplPlanner
from airo_typing import BoundingBox3DType, HomogeneousMatrixType, OpenCVIntImageType, PointCloud
from cloth_tools.annotation.grasp_annotation import get_manual_grasp_annotation, save_grasp_info
from cloth_tools.bounding_boxes import BBOX_CLOTH_IN_THE_AIR, bbox_to_mins_and_sizes
from cloth_tools.controllers.controller import Controller
from cloth_tools.controllers.grasp_lowest_controller import create_cloth_obstacle_planner
from cloth_tools.controllers.home_controller import HomeController
from cloth_tools.kinematics.constants import TCP_TRANSFORM
from cloth_tools.kinematics.inverse_kinematics import inverse_kinematics_in_world_fn
from cloth_tools.planning.grasp_planning import plan_pregrasp_and_grasp_trajectory
from cloth_tools.point_clouds.camera import get_image_and_filtered_point_cloud
from cloth_tools.stations.competition_station import CompetitionStation
from cloth_tools.stations.dual_arm_station import DualArmStation
from cloth_tools.trajectory_execution import execute_dual_arm_drake_trajectory
from cloth_tools.visualization.opencv import draw_pose
from cloth_tools.visualization.rerun import rr_log_camera
from loguru import logger

import open3d as o3d  # isort:skip


class GraspHangingController(Controller):
    """
    Grasps the lowest point of a piece of cloth and lift the cloth into the air.

    Currently always uses the left arm to grasp.
    """

    def __init__(self, station: DualArmStation, bbox: BoundingBox3DType, sample_dir: str = None):
        self.station = station
        self.bbox = bbox

        self.sample_dir = sample_dir  # Quick fix for data collection, saving grasp will be done elsewhere later

        # Attributes that will be set in plan()
        self._image: OpenCVIntImageType | None = None
        self._grasp_pose: HomogeneousMatrixType | None = None
        self._point_cloud: PointCloud | None = None
        self._pregrasp_pose: HomogeneousMatrixType | None = None

        self._trajectory_pregrasp_and_grasp: Any | None = None

        self._hull: o3d.t.geometry.TriangleMesh | None = None
        self._drake_scene: DualArmScene | None = None
        self._planner_with_cloth_obstacle: DualArmOmplPlanner | None = None
        # self._start_joints_left: Any | None = None

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
        planner, scene, hull = create_cloth_obstacle_planner(self.station, point_cloud_cloth, left_hanging=True)
        # Save this for visualization
        self._hull = hull
        self._drake_scene = scene

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
            logger.warning("Recieved no grasp pose.")
            return

        planner_cloth = self._create_cloth_obstacle_planner(point_cloud_cropped)
        self._planner_with_cloth_obstacle = planner_cloth  # Save for debugging

        # Create planner with cloth obstacle

        dual_arm = self.station.dual_arm
        start_joints_left = dual_arm.left_manipulator.get_joint_configuration()
        start_joints_right = dual_arm.right_manipulator.get_joint_configuration()

        X_W_LCB = self.station.left_arm_pose
        X_W_RCB = self.station.right_arm_pose

        inverse_kinematics_left_fn = partial(
            inverse_kinematics_in_world_fn, X_W_CB=X_W_LCB, tcp_transform=TCP_TRANSFORM
        )
        inverse_kinematics_right_fn = partial(
            inverse_kinematics_in_world_fn, X_W_CB=X_W_RCB, tcp_transform=TCP_TRANSFORM
        )

        plant_default = self.station.drake_scene.robot_diagram.plant()
        is_state_valid_fn_default = self.station.planner.is_state_valid_fn

        try:
            trajectory = plan_pregrasp_and_grasp_trajectory(
                planner_cloth,
                grasp_pose,
                start_joints_left,
                start_joints_right,
                inverse_kinematics_left_fn,
                inverse_kinematics_right_fn,
                is_state_valid_fn_default,
                plant_default,
                with_left=False,
            )
        except Exception as e:
            logger.warning(f"Failed to plan grasp. Exception was:\n {e}.")
            self._grasp_info = None
            self._grasp_pose = None
            return

        self._trajectory_pregrasp_and_grasp = trajectory

    def execute_grasp(self) -> None:
        if self.sample_dir is not None:
            # Save the grasp we are about to execute
            sample_dir = self.sample_dir
            grasp_info = self._grasp_info
            grasp_dir = Path(sample_dir) / "grasp"
            save_grasp_info(str(grasp_dir), grasp_info)

        dual_arm = self.station.dual_arm

        # Execute the path to the pregrasp pose
        execute_dual_arm_drake_trajectory(dual_arm, self._trajectory_pregrasp_and_grasp)

        # Execute the grasp
        dual_arm.right_manipulator.gripper.close().wait()

    def _can_execute(self) -> bool:
        # maybe this should just be a property?
        if self._trajectory_pregrasp_and_grasp is None:
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

        if self._trajectory_pregrasp_and_grasp is not None:
            scene = self._drake_scene
            animate_dual_joint_trajectory(
                scene.meshcat,
                scene.robot_diagram,
                scene.arm_left_index,
                scene.arm_right_index,
                self._trajectory_pregrasp_and_grasp,
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
