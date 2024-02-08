import sys
from typing import Any, Optional, Tuple

import cv2
import numpy as np
import rerun as rr
from airo_camera_toolkit.utils.image_converter import ImageConverter
from airo_typing import HomogeneousMatrixType, OpenCVIntImageType, PointCloud
from cloth_tools.controllers.controller import Controller
from cloth_tools.controllers.grasp_highest_controller import hang_in_the_air_tcp_pose
from cloth_tools.controllers.grasp_lowest_controller import create_cloth_obstacle_planner
from cloth_tools.drake.visualization import publish_dual_arm_trajectory
from cloth_tools.ompl.dual_arm_planner import DualArmOmplPlanner
from cloth_tools.path.execution import execute_dual_arm_trajectory, time_parametrize_toppra
from cloth_tools.point_clouds.camera import get_image_and_filtered_point_cloud
from cloth_tools.stations.competition_station import CompetitionStation
from cloth_tools.stations.dual_arm_station import DualArmStation
from cloth_tools.visualization.opencv import draw_pose
from cloth_tools.visualization.rerun import rr_log_camera
from linen.elemental.move_backwards import move_pose_backwards
from loguru import logger
from pydrake.trajectories import Trajectory


class StretchController(Controller):
    """
    Grasps the lowest point of a piece of cloth and lift the cloth into the air.

    Currently always uses the left arm to grasp.
    """

    def __init__(self, station: DualArmStation):
        self.station = station

        # Attributes that will be set in plan()
        self._image: Optional[OpenCVIntImageType] = None
        self._point_cloud: Optional[PointCloud] = None
        self._stretch_pose_left: Optional[HomogeneousMatrixType] = None
        self._stretch_pose_right: Optional[HomogeneousMatrixType] = None
        self._path_to_stretch: Optional[Any] = None
        self._trajectory_to_stretch: Optional[Trajectory] = None
        self._time_trajectory_to_stretch: Optional[Trajectory] = None

        camera = self.station.camera
        camera_pose = self.station.camera_pose

        # Setting up the OpenCV window
        window_name = self.__class__.__name__
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 1280, 720)

        # Setting up Rerun
        rr.init(window_name, spawn=True)
        rr_log_camera(camera, camera_pose)

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

        # These are retrieved just for visualization
        image_rgb, _, point_cloud = get_image_and_filtered_point_cloud(camera, camera_pose)
        image = ImageConverter.from_numpy_int_format(image_rgb).image_in_opencv_format
        self._image = image
        self._point_cloud = point_cloud

        hang_pose_left = hang_in_the_air_tcp_pose(left=True)
        hang_pose_right = hang_in_the_air_tcp_pose(left=False)

        X_LCB_TCP = self.station.dual_arm.left_manipulator.get_tcp_pose()
        X_RCB_TCP = self.station.dual_arm.right_manipulator.get_tcp_pose()
        X_W_LCB = self.station.left_arm_pose
        X_W_RCB = self.station.right_arm_pose
        X_W_LTCP = X_W_LCB @ X_LCB_TCP
        X_W_RTCP = X_W_RCB @ X_RCB_TCP

        tcp_distance = np.linalg.norm(X_W_LTCP[:3, 3] - X_W_RTCP[:3, 3])
        logger.info(f"{self.__class__.__name__}: Distance between TCPs at start: {tcp_distance:.3f} meters")

        backwards_shift = tcp_distance / 2.0  # keep end distance same as distance at grasp
        stretch_pose_left = move_pose_backwards(hang_pose_left, backwards_shift)
        stretch_pose_right = move_pose_backwards(hang_pose_right, backwards_shift)

        # Move arms closer to the camera
        global_x_shift = -0.4
        stretch_pose_left[:3, 3] += np.array([global_x_shift, 0, 0])
        stretch_pose_right[:3, 3] += np.array([global_x_shift, 0, 0])

        self._stretch_pose_left = stretch_pose_left
        self._stretch_pose_right = stretch_pose_right

        dual_arm = self.station.dual_arm
        start_joints_left = dual_arm.left_manipulator.get_joint_configuration()
        start_joints_right = dual_arm.right_manipulator.get_joint_configuration()

        planner = self.station.planner

        path_to_stretch = planner.plan_to_tcp_pose(
            start_joints_left, start_joints_right, stretch_pose_left, stretch_pose_right
        )
        self._path_to_stretch = path_to_stretch

        if path_to_stretch is None:
            return

        # Time parametrize the path
        # Move very slowly
        trajectory_to_stretch, time_trajectory_to_stretch = time_parametrize_toppra(
            path_to_stretch, self.station._diagram.plant(), joint_speed_limit=0.5, joint_acceleration_limit=0.5
        )
        self._trajectory_to_stretch = trajectory_to_stretch
        self._time_trajectory_to_stretch = time_trajectory_to_stretch

    def execute_stretch(self) -> None:
        dual_arm = self.station.dual_arm

        # Execute the path to the pregrasp pose
        execute_dual_arm_trajectory(dual_arm, self._trajectory_to_stretch, self._time_trajectory_to_stretch)

    def _can_execute(self) -> bool:
        # maybe this should just be a property?
        if self._trajectory_to_stretch is None:
            return False

        return True

    def visualize_plan(self) -> Tuple[OpenCVIntImageType, Any]:  # noqa C901
        if self._image is None:
            raise RuntimeError("You must call plan() before visualize_plan().")

        image = self._image
        camera_pose = self.station.camera_pose
        intrinsics = self.station.camera.intrinsics_matrix()

        if self._stretch_pose_left is not None:
            pose = self._stretch_pose_left
            draw_pose(image, pose, intrinsics, camera_pose)

            rr_pose = rr.Transform3D(translation=pose[0:3, 3], mat3x3=pose[0:3, 0:3] @ np.identity(3) * 0.1)
            rr.log("world/stretch_pose_left", rr_pose)

        if self._stretch_pose_right is not None:
            pose = self._stretch_pose_right
            draw_pose(image, pose, intrinsics, camera_pose)

            rr_pose = rr.Transform3D(translation=pose[0:3, 3], mat3x3=pose[0:3, 0:3] @ np.identity(3) * 0.1)
            rr.log("world/stretch_pose_right", rr_pose)

        if self._path_to_stretch is not None:
            trajectory = self._trajectory_to_stretch
            time_trajectory = self._time_trajectory_to_stretch
            publish_dual_arm_trajectory(
                trajectory,
                time_trajectory,
                self.station._meshcat,
                self.station._diagram,
                self.station._context,
                *self.station._arm_indices,
            )

        if self._point_cloud is not None:
            point_cloud = self._point_cloud
            rr_point_cloud = rr.Points3D(positions=point_cloud.points, colors=point_cloud.colors)
            rr.log("world/point_cloud", rr_point_cloud)

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
        if not self._can_execute():
            logger.warn(f"{self.__class__.__name__}: not executing because the plan is not complete.")
            return

        self.execute_stretch()

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

    # Assumes both arms are aleady holding the cloth
    stretch_controller = StretchController(station)
    stretch_controller.execute(interactive=True)
