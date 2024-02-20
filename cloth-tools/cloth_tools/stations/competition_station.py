import multiprocessing
from functools import partial
from typing import List

import numpy as np
from airo_camera_toolkit.cameras.multiprocess.multiprocess_stereo_rgbd_camera import (
    MultiprocessStereoRGBDPublisher,
    MultiprocessStereoRGBDReceiver,
)
from airo_camera_toolkit.cameras.zed.zed2i import Zed2i
from airo_camera_toolkit.image_transforms.transforms.crop import Crop
from airo_camera_toolkit.interfaces import StereoRGBDCamera
from airo_typing import HomogeneousMatrixType, JointConfigurationType
from cloth_tools.config import load_camera_pose_in_left_and_right, setup_dual_arm_ur5e_in_world
from cloth_tools.drake.building import add_meshcat_to_builder, finish_build
from cloth_tools.drake.scenes import add_dual_ur5e_and_table_to_builder
from cloth_tools.ompl.dual_arm_planner import DualArmOmplPlanner
from cloth_tools.planning.interfaces import DualArmMotionPlanner
from cloth_tools.stations.coordinate_frames import create_egocentric_world_frame
from cloth_tools.stations.dual_arm_station import DualArmStation
from loguru import logger
from pydrake.planning import RobotDiagramBuilder, SceneGraphCollisionChecker
from ur_analytic_ik import ur5e

# Hardcoded for now
tcp_transform = np.identity(4)
tcp_transform[2, 3] = 0.175


def inverse_kinematics_in_world_fn(
    tcp_pose: HomogeneousMatrixType, X_W_CB: HomogeneousMatrixType
) -> List[JointConfigurationType]:
    X_W_TCP = tcp_pose
    X_CB_W = np.linalg.inv(X_W_CB)
    solutions_1x6 = ur5e.inverse_kinematics_with_tcp(X_CB_W @ X_W_TCP, tcp_transform)
    solutions = [solution.squeeze() for solution in solutions_1x6]
    return solutions


def check_zed_point_cloud_completeness(camera: Zed2i):
    # Check whether the point cloud is complete, i.e. if there are any points closers than 1.0 meters
    point_cloud = camera.get_colored_point_cloud()
    image_rgb = camera.get_rgb_image_as_int()
    image_right_rgb = camera._retrieve_rgb_image_as_int(StereoRGBDCamera.RIGHT_RGB)
    confidence_map = camera._retrieve_confidence_map()
    depth_map = camera._retrieve_depth_map()
    depth_image = camera._retrieve_depth_image()

    distances = np.linalg.norm(point_cloud.points, axis=1)
    if not np.any(distances < 1.2):
        print(distances)
        logger.info("The point cloud is not complete, logging it to rerun.")
        import rerun as rr

        rr.init("Competition Station - Point cloud", spawn=True)
        rr.log("world/point_cloud", rr.Points3D(positions=point_cloud.points, colors=point_cloud.colors))
        rr.log("image", rr.Image(image_rgb).compress(jpeg_quality=90))
        rr.log("image_right", rr.Image(image_right_rgb).compress(jpeg_quality=90))
        rr.log("depth_image", rr.Image(depth_image).compress(jpeg_quality=90))
        rr.log("depth_map", rr.DepthImage(depth_map))
        rr.log("confidence_map", rr.Image(confidence_map))

        raise RuntimeError("The point cloud is incomplete. Restart the ZED2i camera.")


class CompetitionStation(DualArmStation):
    """
    This station specifically contains the hardware setup for the ICRA 2024 competition.
    It (currently) consists of two UR5e robots and a single ZED2i camera.
    The robots are mounted approximately 0.9 meter apart.
    """

    def __init__(self, create_multiprocess_camera: bool = True) -> None:

        # Setting up the camera
        camera_kwargs = {
            "resolution": Zed2i.RESOLUTION_2K,
            "depth_mode": Zed2i.NEURAL_DEPTH_MODE,
            "fps": 15,
        }
        self.camera_publisher = None

        if create_multiprocess_camera:
            multiprocessing.set_start_method("spawn")

            # Running the camera in a seperate process enables us to record videos even if the main process is blocking
            self.camera_publisher = MultiprocessStereoRGBDPublisher(Zed2i, camera_kwargs)
            # self.camera_publisher.publish_depth_image = False

            self.camera_publisher.start()
            camera = MultiprocessStereoRGBDReceiver("camera")
        else:
            camera = Zed2i(**camera_kwargs)

        check_zed_point_cloud_completeness(camera)

        # Image crop used to check motion blur of hanging cloth
        # Note that parts of the cloth may be outside the crop
        image_rgb = camera.get_rgb_image_as_int()
        self.hanging_cloth_crop = Crop(image_rgb.shape, x=1000, y=200, w=200, h=500)

        camera_pose_in_left, camera_pose_in_right = load_camera_pose_in_left_and_right()

        X_W_C, X_W_LCB, X_W_RCB = create_egocentric_world_frame(camera_pose_in_left, camera_pose_in_right)

        camera_pose = X_W_C  # this must be consistent with the setup_dual_arm_ur5e call below
        self.left_arm_pose = X_W_LCB
        self.right_arm_pose = X_W_RCB

        dual_arm = setup_dual_arm_ur5e_in_world(X_W_LCB, X_W_RCB)
        super().__init__(dual_arm, camera, camera_pose)

        # Setting some home joints
        self.home_joints_left = np.deg2rad([180, -120, 60, -30, -90, -90])
        self.home_joints_right = np.deg2rad([-180, -60, -60, -150, 90, 90])

        joint_bounds_lower = np.deg2rad([-360, -180, -160, -360, -360, -360])
        joint_bounds_upper = np.deg2rad([360, 0, 160, 360, 360, 360])
        joint_bounds = (joint_bounds_lower, joint_bounds_upper)
        self.joint_bounds_left = joint_bounds
        self.joint_bounds_right = joint_bounds

        # Planner for the two arms without obstacles (only the table)
        self.planner: DualArmMotionPlanner = self._setup_planner(X_W_LCB, X_W_RCB)

        # This is purely for visualization, but read the robot joints and publish them to meshcat
        diagram = self._diagram
        context = self._context
        arm_indices = self._arm_indices
        self.home_joints_left
        self.home_joints_right

        # Publishing the current joint is purely for debugging. This way you can check in meshcat if the robot is
        # mounted the same way as in the real world as in the simulation.
        current_joints_left = dual_arm.left_manipulator.get_joint_configuration()
        current_joints_right = dual_arm.right_manipulator.get_joint_configuration()
        plant = diagram.plant()
        plant_context = plant.GetMyContextFromRoot(context)
        arm_left_index, arm_right_index = arm_indices
        plant.SetPositions(plant_context, arm_left_index, current_joints_left)
        plant.SetPositions(plant_context, arm_right_index, current_joints_right)
        diagram.ForcedPublish(context)

        logger.info("CompetitionStation initialized.")

    def __del__(self):
        if self.camera_publisher is not None:
            self.camera_publisher.stop()
            self.camera_publisher.join()

    def _setup_planner(self, X_W_LCB, X_W_RCB) -> DualArmOmplPlanner:
        # Creating the default scene
        robot_diagram_builder = RobotDiagramBuilder()
        meshcat = add_meshcat_to_builder(robot_diagram_builder)

        arm_indices, gripper_indices = add_dual_ur5e_and_table_to_builder(robot_diagram_builder, X_W_LCB, X_W_RCB)
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
            self.joint_bounds_left,
            self.joint_bounds_right,
        )
        return planner


if __name__ == "__main__":
    import cv2
    import rerun as rr
    from airo_camera_toolkit.utils.image_converter import ImageConverter
    from cloth_tools.visualization.opencv import draw_pose

    # Check whether all hardware is connected
    station = CompetitionStation()
    camera = station.camera
    dual_arm = station.dual_arm

    X_W_C = station.camera_pose
    X_W_LCB = station.left_arm_pose
    X_W_RCB = station.right_arm_pose

    window_name = "Competiton station"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1600, 800)

    rr.init("Competition Station - Point cloud", spawn=True)

    while True:
        image_rgb = camera.get_rgb_image_as_int()
        image_right_rgb = camera._retrieve_rgb_image_as_int(StereoRGBDCamera.RIGHT_RGB)
        point_cloud = camera._retrieve_colored_point_cloud()
        confidence_map = camera._retrieve_confidence_map()
        depth_map = camera._retrieve_depth_map()
        depth_image = camera._retrieve_depth_image()

        image_bgr = ImageConverter.from_numpy_int_format(image_rgb).image_in_opencv_format

        X_CB_LTCP = dual_arm.left_manipulator.get_tcp_pose()
        X_CB_RTCP = dual_arm.right_manipulator.get_tcp_pose()

        intrinsics = camera.intrinsics_matrix()
        draw_pose(image_bgr, np.identity(4), intrinsics, X_W_C, 0.25)
        draw_pose(image_bgr, X_W_LCB, intrinsics, X_W_C)
        draw_pose(image_bgr, X_W_RCB, intrinsics, X_W_C)
        draw_pose(image_bgr, X_W_LCB @ X_CB_LTCP, intrinsics, X_W_C, 0.05)
        draw_pose(image_bgr, X_W_RCB @ X_CB_RTCP, intrinsics, X_W_C, 0.05)

        cv2.imshow(window_name, image_bgr)
        key = cv2.waitKey(1)
        if key == ord("q"):
            break

        # Unfiltered point cloud in camera frame
        rr_point_cloud = rr.Points3D(positions=point_cloud.points, colors=point_cloud.colors)
        rr.log("world/point_cloud", rr_point_cloud)
        rr.log("image", rr.Image(image_rgb).compress(jpeg_quality=90))
        rr.log("image_right", rr.Image(image_right_rgb).compress(jpeg_quality=90))
        rr.log("depth_image", rr.Image(depth_image).compress(jpeg_quality=90))
        rr.log("depth_map", rr.DepthImage(depth_map))
        rr.log("confidence_map", rr.Image(confidence_map))
