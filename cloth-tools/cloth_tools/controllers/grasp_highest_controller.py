import sys
from typing import Any, Optional, Tuple

import cv2
import numpy as np
import rerun as rr
from airo_camera_toolkit.point_clouds.operations import crop_point_cloud
from airo_camera_toolkit.utils import ImageConverter
from airo_typing import (
    BoundingBox3DType,
    HomogeneousMatrixType,
    JointConfigurationType,
    OpenCVIntImageType,
    PointCloud,
    Vector3DType,
)
from cloth_tools.bounding_boxes import BBOX_CLOTH_ON_TABLE
from cloth_tools.controllers.basic_home_controller import BasicHomeController
from cloth_tools.controllers.controller import Controller
from cloth_tools.point_clouds.camera import get_image_and_filtered_point_cloud
from cloth_tools.point_clouds.operations import highest_point
from cloth_tools.stations.competition_station import CompetitionStation
from cloth_tools.stations.dual_arm_station import DualArmStation
from cloth_tools.visualization.opencv import draw_point_3d, draw_pose
from cloth_tools.visualization.rerun import rr_log_camera
from linen.elemental.move_backwards import move_pose_backwards
from linen.geometry.orientation import top_down_orientation
from loguru import logger


def highest_point_grasp_pose(highest_point: Vector3DType, grasp_depth: float = 0.05) -> HomogeneousMatrixType:
    """Returns a top-down grasp pose for the highest point of a piece of cloth.
    The grasp height will be at least 1 cm to avoid collisions with the table.
    The gripper will open along the global x-axis.

    Args:
        highest_point: The highest point of the cloth.

    Returns:
        The grasp pose.
    """
    grasp_orientation = top_down_orientation([1, 0, 0])
    grasp_location = highest_point.copy()
    grasp_location[2] -= grasp_depth
    grasp_location[2] = max(grasp_location[2], 0.01)  # grasp at least 1cm above robot base
    grasp_pose = np.identity(4)
    grasp_pose[0:3, 0:3] = grasp_orientation
    grasp_pose[0:3, 3] = grasp_location
    return grasp_pose


def hang_in_the_air_joints(left: bool) -> JointConfigurationType:
    """Hardcoded joint poses for the left arm right arm to hang the cloth in the air.

    Args:
        left: Whether the left or right arm will do the hanging.

    Returns:
        The joint angles.
    """
    if left:
        return np.deg2rad([180, -90, 30, -120, -90, -90])
    else:
        return np.deg2rad([-180, -90, -30, -60, 90, 90])


class GraspHighestController(Controller):
    """
    Grasps the highest point of a piece of cloth and lift the cloth into the air.

    Currently always uses the right arm to grasp and hang the cloth.
    """

    def __init__(self, station: DualArmStation, bbox: BoundingBox3DType):
        self.station = station
        self.bbox = bbox
        self.hang_joints = hang_in_the_air_joints(left=False)

        # Attributes that will be set in plan()
        self._image: Optional[OpenCVIntImageType] = None
        self._grasp_pose: Optional[HomogeneousMatrixType] = None
        self._point_cloud: Optional[PointCloud] = None
        self._highest_point: Optional[Vector3DType] = None

        camera = self.station.camera
        camera_pose = self.station.camera_pose

        # Setting up the OpenCV window
        window_name = self.__class__.__name__
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 1280, 720)

        # Setting up Rerun
        rr.init(window_name, spawn=True)
        rr_log_camera(camera, camera_pose)

    def execute_grasp_and_hang(self, grasp_pose: HomogeneousMatrixType) -> None:
        dual_arm = self.station.dual_arm

        assert dual_arm.right_manipulator.gripper is not None  # For mypy

        # Grasp with a pregrasp pose, which also serves as retreat pose
        pregrasp_pose = move_pose_backwards(grasp_pose, 0.1)
        dual_arm.move_linear_to_tcp_pose(None, pregrasp_pose).wait()  # TODO: make this a move_to_tcp_pose
        dual_arm.move_linear_to_tcp_pose(None, grasp_pose, linear_speed=0.2).wait()
        dual_arm.right_manipulator.gripper.close().wait()
        dual_arm.move_linear_to_tcp_pose(None, pregrasp_pose, linear_speed=0.2).wait()

        # Hang the cloth in the air
        dual_arm.right_manipulator.move_to_joint_configuration(self.hang_joints, joint_speed=0.3).wait()

    def plan(self) -> None:
        camera = self.station.camera
        camera_pose = self.station.camera_pose

        image_rgb, point_cloud = get_image_and_filtered_point_cloud(camera, camera_pose)
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

        self._highest_point = highest_point_
        self._grasp_pose = grasp_pose

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

        if self._point_cloud is not None:
            point_cloud = self._point_cloud
            rr_point_cloud = rr.Points3D(positions=point_cloud.points, colors=point_cloud.colors)
            rr.log("world/point_cloud", rr_point_cloud)

        cv2.imshow(self.__class__.__name__, image)
        key = cv2.waitKey(1)
        return image, key

    def execute_plan(self) -> None:
        if self._grasp_pose is None:
            logger.info("Grasp and hang not executed because no grasp pose was found.")
            return
        self.execute_grasp_and_hang(self._grasp_pose)

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

        logger.info(f"{self.__class__.__name__} finished.")


if __name__ == "__main__":
    station = CompetitionStation()

    dual_arm = station.dual_arm
    # Move the arms to their home positions
    home_controller = BasicHomeController(station)
    home_controller.execute()

    grasp_highest_controller = GraspHighestController(station, BBOX_CLOTH_ON_TABLE)
    grasp_highest_controller.execute(interactive=True)
