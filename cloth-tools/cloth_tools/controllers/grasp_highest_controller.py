import sys
import time
from typing import Any, List, Optional, Tuple

import cv2
import numpy as np
import rerun as rr
from airo_camera_toolkit.point_clouds.operations import crop_point_cloud
from airo_camera_toolkit.utils.image_converter import ImageConverter
from airo_typing import (
    BoundingBox3DType,
    HomogeneousMatrixType,
    JointConfigurationType,
    OpenCVIntImageType,
    PointCloud,
    RotationMatrixType,
    Vector3DType,
)
from cloth_tools.bounding_boxes import BBOX_CLOTH_ON_TABLE, bbox_to_mins_and_sizes
from cloth_tools.controllers.controller import Controller
from cloth_tools.controllers.home_controller import HomeController
from cloth_tools.drake.visualization import publish_dual_arm_trajectory
from cloth_tools.path.execution import execute_dual_arm_trajectory, time_parametrize_toppra
from cloth_tools.point_clouds.camera import get_image_and_filtered_point_cloud
from cloth_tools.point_clouds.operations import highest_point
from cloth_tools.stations.competition_station import CompetitionStation
from cloth_tools.visualization.opencv import draw_point_3d, draw_pose
from cloth_tools.visualization.rerun import rr_log_camera
from linen.elemental.move_backwards import move_pose_backwards
from linen.geometry.orientation import top_down_orientation
from loguru import logger
from pydrake.trajectories import Trajectory


def highest_point_grasp_pose(highest_point: Vector3DType, grasp_depth: float = 0.05) -> HomogeneousMatrixType:
    """Returns a top-down grasp pose for the highest point of a piece of cloth.
    The grasp height will be at least 1 cm to avoid collisions with the table.
    The gripper will open along the global x-axis.

    Args:
        highest_point: The highest point of the cloth.

    Returns:
        The grasp pose.
    """
    grasp_orientation = top_down_orientation([0, -1, 0])
    grasp_location = highest_point.copy()
    grasp_location[2] -= grasp_depth
    grasp_location[2] = max(grasp_location[2], 0.01)  # grasp at least 1cm above robot base
    grasp_pose = np.identity(4)
    grasp_pose[0:3, 0:3] = grasp_orientation
    grasp_pose[0:3, 3] = grasp_location
    return grasp_pose


def hang_in_the_air_tcp_orientation(left: bool) -> RotationMatrixType:
    gripper_forward_direction = np.array([0, -1, 0]) if left else np.array([0, 1, 0])
    Z = gripper_forward_direction / np.linalg.norm(gripper_forward_direction)
    X = np.array([0, 0, 1]) if left else np.array([0, 0, -1])
    Y = np.cross(Z, X)
    return np.column_stack([X, Y, Z])


def hang_in_the_air_tcp_pose(left: bool) -> HomogeneousMatrixType:
    """Hardcoded TCP poses for the left arm right arm to hang the cloth in the air.

    Assumes the world frame lies between the robot arms, and the left arm stand on the positive y-axis and the right arm
    on the negative y-axis.

    Args:
        left: Whether the left or right arm will do the hanging.

    Returns:
        The TCP pose.
    """
    position = np.array([0, 0, 0.9])  # 1 m is too close to a singularity
    gripper_orientation = hang_in_the_air_tcp_orientation(left)

    gripper_pose = np.identity(4)
    gripper_pose[0:3, 0:3] = gripper_orientation
    gripper_pose[0:3, 3] = position
    return gripper_pose


class GraspHighestController(Controller):
    """
    Grasps the highest point of a piece of cloth and lift the cloth into the air.

    Currently always uses the right arm to grasp and hang the cloth.
    """

    def __init__(self, station: CompetitionStation, bbox: BoundingBox3DType):
        self.station = station
        self.bbox = bbox

        # Attributes that will be set in plan()
        self._image: Optional[OpenCVIntImageType] = None
        self._grasp_pose: Optional[HomogeneousMatrixType] = None
        self._pregrasp_pose: Optional[HomogeneousMatrixType] = None
        self._point_cloud: Optional[PointCloud] = None
        self._highest_point: Optional[Vector3DType] = None
        self._path_pregrasp: Optional[List[Tuple[JointConfigurationType, JointConfigurationType]]] = None
        self._path_hang: Optional[List[Tuple[JointConfigurationType, JointConfigurationType]]] = None
        self._hang_tcp_pose: Optional[HomogeneousMatrixType] = None
        self._trajectory_pregrasp: Optional[Trajectory] = None
        self._time_trajectory_pregrasp: Optional[Trajectory] = None
        self._trajectory_hang: Optional[Trajectory] = None
        self._time_trajectory_hang: Optional[Trajectory] = None

        camera = self.station.camera
        camera_pose = self.station.camera_pose

        # Setting up the OpenCV window
        window_name = self.__class__.__name__
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 1280, 720)

        # Setting up Rerun
        rr.init(window_name, spawn=True)
        rr_log_camera(camera, camera_pose)
        bbox_color = (255, 231, 122)  # yellow
        bbox_mins, bbox_sizes = bbox_to_mins_and_sizes(bbox)
        rr_bbox = rr.Boxes3D(mins=bbox_mins, sizes=bbox_sizes, colors=bbox_color)
        rr.log("world/bbox", rr_bbox)

    def execute_grasp_and_hang(self, grasp_pose: HomogeneousMatrixType, pregrasp_pose: HomogeneousMatrixType) -> None:
        dual_arm = self.station.dual_arm

        assert dual_arm.right_manipulator.gripper is not None  # For mypy

        # Execute the path to the pregrasp pose
        execute_dual_arm_trajectory(dual_arm, self._trajectory_pregrasp, self._time_trajectory_pregrasp)

        # Execute the grasp
        dual_arm.move_linear_to_tcp_pose(None, grasp_pose, linear_speed=0.2).wait()
        dual_arm.right_manipulator.gripper.close().wait()
        dual_arm.move_linear_to_tcp_pose(None, pregrasp_pose, linear_speed=0.2).wait()

        # Hang the cloth in the air (do this a bit slower to limit the cloth swinging)
        execute_dual_arm_trajectory(dual_arm, self._trajectory_hang, self._time_trajectory_hang)

    def plan(self) -> None:
        logger.info(f"{self.__class__.__name__}: Creating new plan.")

        camera = self.station.camera
        camera_pose = self.station.camera_pose

        time.sleep(1.0)  # without this sleep I've noticed motion blur on the images e.g. is the cloth has just fallen
        image_rgb, _, point_cloud = get_image_and_filtered_point_cloud(camera, camera_pose)
        image = ImageConverter.from_numpy_int_format(image_rgb).image_in_opencv_format

        self._image = image
        self._point_cloud = point_cloud

        point_cloud_cropped = crop_point_cloud(point_cloud, self.bbox)

        if len(point_cloud_cropped.points) == 0:
            self._highest_point = None
            self._grasp_pose = None
            return

        highest_point_ = highest_point(point_cloud_cropped.points)
        grasp_pose = highest_point_grasp_pose(highest_point_)

        logger.info(f"{self.__class__.__name__}: Found highest point in bbox at {highest_point_}.")

        self._highest_point = highest_point_
        self._grasp_pose = grasp_pose

        pregrasp_pose = move_pose_backwards(grasp_pose, 0.1)
        self._pregrasp_pose = pregrasp_pose

        logger.info(f"{self.__class__.__name__}: Planning from start joints to pregrasp pose.")

        planner = self.station.planner
        dual_arm = self.station.dual_arm
        start_joints_left = dual_arm.left_manipulator.get_joint_configuration()
        start_joints_right = dual_arm.right_manipulator.get_joint_configuration()
        path_pregrasp = planner.plan_to_tcp_pose(start_joints_left, start_joints_right, None, pregrasp_pose)
        self._path_pregrasp = path_pregrasp

        if path_pregrasp is None:
            return

        trajectory_pregrasp, time_trajectory_pregrasp = time_parametrize_toppra(
            path_pregrasp, self.station._diagram.plant()
        )
        self._trajectory_pregrasp = trajectory_pregrasp
        self._time_trajectory_pregrasp = time_trajectory_pregrasp

        logger.info(f"{self.__class__.__name__}: Planning from pregrasp pose to hang pose.")

        # we operate under the assumption that the after the grasp the robot is back at the pregrasp pose with the same joint config
        pregrasp_joints_right = path_pregrasp[-1][1]

        # Warning: only implemented for right arm at the moment, when implementing for left arm, change args in plan_()
        self._hang_tcp_pose = hang_in_the_air_tcp_pose(left=False)
        path_hang = planner.plan_to_tcp_pose(start_joints_left, pregrasp_joints_right, None, self._hang_tcp_pose)
        self._path_hang = path_hang

        # Lower joint acceleration limit to avoid swinging
        trajectory_hang, time_trajectory_hang = time_parametrize_toppra(
            path_hang, self.station._diagram.plant(), joint_acceleration_limit=0.5
        )
        self._trajectory_hang = trajectory_hang
        self._time_trajectory_hang = time_trajectory_hang

    def _can_execute(self) -> bool:
        # maybe this should just be a property?
        if self._path_pregrasp is None:
            return False

        if self._path_hang is None:
            return False

        return True

    def visualize_plan(self) -> Tuple[OpenCVIntImageType, Any]:
        if self._image is None:
            raise RuntimeError("You must call plan() before visualize_plan().")

        image = self._image
        camera_pose = self.station.camera_pose
        intrinsics = self.station.camera.intrinsics_matrix()

        if self._highest_point is not None:
            highest = self._highest_point
            draw_point_3d(image, highest, intrinsics, camera_pose, (0, 255, 0))

        if self._grasp_pose is not None:
            grasp_pose = self._grasp_pose
            draw_pose(image, grasp_pose, intrinsics, camera_pose)

            rr_grasp_pose = rr.Transform3D(translation=grasp_pose[0:3, 3], mat3x3=grasp_pose[0:3, 0:3])
            rr.log("world/grasp_pose", rr_grasp_pose)

        if self._hang_tcp_pose is not None:
            hang_tcp_pose = self._hang_tcp_pose
            draw_pose(image, hang_tcp_pose, intrinsics, camera_pose)

            rr_hang_tcp_pose = rr.Transform3D(translation=hang_tcp_pose[0:3, 3], mat3x3=hang_tcp_pose[0:3, 0:3])
            rr.log("world/hang_tcp_pose", rr_hang_tcp_pose)

        if self._path_pregrasp is not None:
            station = self.station
            trajectory = self._trajectory_pregrasp
            time_trajectory = self._time_trajectory_pregrasp

            publish_dual_arm_trajectory(
                trajectory,
                time_trajectory,
                station._meshcat,
                station._diagram,
                station._context,
                *station._arm_indices,
            )
            # TODO find a way to also publish the hang path, maybe just append it?

        if self._point_cloud is not None:
            point_cloud = self._point_cloud
            rr_point_cloud = rr.Points3D(positions=point_cloud.points, colors=point_cloud.colors)
            rr.log("world/point_cloud", rr_point_cloud)

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
        if self._grasp_pose is None:
            logger.info("Grasp and hang not executed because no grasp pose was found.")
            return
        self.execute_grasp_and_hang(self._grasp_pose, self._pregrasp_pose)

    def execute_interactive(self) -> None:
        while True:
            self.plan()
            _, key = self.visualize_plan()
            if key == ord("y"):
                self.execute_plan()
                return
            elif key == ord("n"):
                self._path_pregrasp = None
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
    # Move the arms to their home positions
    home_controller = HomeController(station)
    home_controller.execute(interactive=True)

    grasp_highest_controller = GraspHighestController(station, BBOX_CLOTH_ON_TABLE)
    grasp_highest_controller.execute(interactive=True)
