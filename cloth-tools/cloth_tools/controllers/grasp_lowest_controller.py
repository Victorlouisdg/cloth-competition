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
    OpenCVIntImageType,
    PointCloud,
    RotationMatrixType,
    Vector3DType,
)
from cloth_tools.bounding_boxes import BBOX_CLOTH_IN_THE_AIR
from cloth_tools.controllers.controller import Controller
from cloth_tools.controllers.home_controller import HomeController
from cloth_tools.point_clouds.camera import get_image_and_filtered_point_cloud
from cloth_tools.point_clouds.operations import lowest_point
from cloth_tools.stations.competition_station import CompetitionStation
from cloth_tools.stations.dual_arm_station import DualArmStation
from cloth_tools.visualization.opencv import draw_point_3d, draw_pose
from cloth_tools.visualization.rerun import rr_log_camera
from loguru import logger


def flatter_orientation(gripper_forward_direction: Vector3DType) -> RotationMatrixType:
    Z = gripper_forward_direction / np.linalg.norm(gripper_forward_direction)
    Y = np.array([0, 0, 1])
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
    grasp_orientation = flatter_orientation(np.array([0, 1, 0]))
    grasp_location = lowest_point.copy()
    grasp_location[1] += grasp_depth
    grasp_location[2] += height_offset
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

        # Attributes that will be set in plan()
        self._image: Optional[OpenCVIntImageType] = None
        self._grasp_pose: Optional[HomogeneousMatrixType] = None
        self._point_cloud: Optional[PointCloud] = None
        self._lowest_point: Optional[Vector3DType] = None

        camera = self.station.camera
        camera_pose = self.station.camera_pose

        # Setting up the OpenCV window
        window_name = self.__class__.__name__
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 1280, 720)

        # Setting up Rerun
        rr.init(window_name, spawn=True)
        rr_log_camera(camera, camera_pose)

    def execute_handover(self, grasp_pose: HomogeneousMatrixType) -> None:
        self.station.dual_arm

        # assert dual_arm.left_manipulator.gripper is not None

        # pregrasp_pose = move_pose_backwards(grasp_pose, 0.1)
        # dual_arm.move_linear_to_tcp_pose(pregrasp_pose, None).wait()  # TODO make this a move_to_tcp_pose

        # # Execute the grasp
        # dual_arm.move_linear_to_tcp_pose(grasp_pose, None, linear_speed=0.2)
        # dual_arm.left_manipulator.gripper.close().wait()

        # Open the right gripper of the cloth be released
        # dual_arm.right_manipulator.gripper.open().wait()

    def plan(self) -> None:
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

    def visualize_plan(self) -> Tuple[OpenCVIntImageType, Any]:
        if self._image is None:
            raise RuntimeError("You must call plan() before visualize_plan().")

        image = self._image
        camera_pose = self.station.camera_pose
        intrinsics = self.station.camera.intrinsics_matrix()

        if self._lowest_point is not None:
            highest = self._lowest_point
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
        self.execute_handover(self._grasp_pose)

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

        logger.info(f"{self.__class__.__name__} finished.")


if __name__ == "__main__":
    station = CompetitionStation()

    dual_arm = station.dual_arm

    # Only move left arm home
    # assert dual_arm.left_manipulator.gripper is not None
    # dual_arm.left_manipulator.gripper.open().wait()
    # dual_arm.left_manipulator.move_to_joint_configuration(station.home_joints_left).wait()

    # Move only left arm to home. right might be holding the cloth
    home_controller = HomeController(station, move_right_home=False)
    home_controller.execute(interactive=True)

    # Assumes the right arm is holding the cloth in the air
    grasp_lowest_controller = GraspLowestController(station, BBOX_CLOTH_IN_THE_AIR)
    # grasp_lowest_controller.execute(interactive=True)
