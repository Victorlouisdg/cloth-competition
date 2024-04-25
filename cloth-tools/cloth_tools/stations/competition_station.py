import multiprocessing
from functools import partial

import numpy as np
from airo_camera_toolkit.cameras.multiprocess.multiprocess_stereo_rgbd_camera import (
    MultiprocessStereoRGBDPublisher,
    MultiprocessStereoRGBDReceiver,
)
from airo_camera_toolkit.cameras.zed.zed2i import Zed2i
from airo_camera_toolkit.image_transforms.transforms.crop import Crop
from airo_camera_toolkit.interfaces import StereoRGBDCamera
from airo_drake import DualArmScene, add_meshcat, finish_build
from airo_planner import DualArmOmplPlanner, DualArmPlanner
from cloth_tools.config import load_camera_pose_in_left_and_right, setup_dual_arm_ur5e_in_world
from cloth_tools.drake.scenes import add_cloth_competition_dual_ur5e_scene
from cloth_tools.kinematics.constants import JOINT_BOUNDS, TCP_TRANSFORM
from cloth_tools.kinematics.inverse_kinematics import inverse_kinematics_in_world_fn
from cloth_tools.stations.coordinate_frames import create_egocentric_world_frame
from cloth_tools.stations.dual_arm_station import DualArmStation
from loguru import logger
from pydrake.planning import RobotDiagramBuilder, SceneGraphCollisionChecker


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

        for i in range(5):
            logger.info(f"Initializing camera... (attempt: {i})")
            if create_multiprocess_camera:
                if i == 0:
                    multiprocessing.set_start_method("spawn")

                # Running the camera in a seperate process enables us to record videos even if the main process is blocking
                self.camera_publisher = MultiprocessStereoRGBDPublisher(Zed2i, camera_kwargs)
                self.camera_publisher.start()
                camera = MultiprocessStereoRGBDReceiver("camera")
            else:
                camera = Zed2i(**camera_kwargs)

            try:
                check_zed_point_cloud_completeness(camera)
                break
            except RuntimeError as e:
                logger.error("Failed to initialize camera. Retrying...")
                logger.warning(e)
                if self.camera_publisher is not None:
                    self.camera_publisher.stop()
                    self.camera_publisher.join()
                    del camera
                if i == 4:
                    raise e

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

        self.joint_bounds_left = JOINT_BOUNDS
        self.joint_bounds_right = JOINT_BOUNDS

        # Planner for the two arms without obstacles (only the table)
        self.planner: DualArmPlanner = self._setup_planner(X_W_LCB, X_W_RCB)

        # Publishing the current joint is purely for debugging. This way you can check in meshcat if the robot is
        # mounted the same way as in the real world as in the simulation.
        robot_diagram = self.drake_scene.robot_diagram
        context = robot_diagram.CreateDefaultContext()
        current_joints_left = dual_arm.left_manipulator.get_joint_configuration()
        current_joints_right = dual_arm.right_manipulator.get_joint_configuration()
        plant = robot_diagram.plant()
        plant_context = plant.GetMyContextFromRoot(context)
        plant.SetPositions(plant_context, self.drake_scene.arm_left_index, current_joints_left)
        plant.SetPositions(plant_context, self.drake_scene.arm_right_index, current_joints_right)
        robot_diagram.ForcedPublish(context)

        logger.success("CompetitionStation initialized.")

    def __del__(self):
        if self.camera_publisher is not None:
            self.camera_publisher.stop()
            self.camera_publisher.join()

    def _setup_planner(self, X_W_LCB, X_W_RCB) -> DualArmOmplPlanner:
        # Creating the default scene

        robot_diagram_builder = RobotDiagramBuilder()

        meshcat = add_meshcat(robot_diagram_builder)
        meshcat.SetCameraPose([-1.5, 0, 1.0], [0, 0, 0])

        (arm_left_index, arm_right_index), (
            gripper_left_index,
            gripper_right_index,
        ) = add_cloth_competition_dual_ur5e_scene(robot_diagram_builder, X_W_LCB, X_W_RCB)
        robot_diagram, _ = finish_build(robot_diagram_builder, meshcat)

        scene = DualArmScene(
            robot_diagram, arm_left_index, arm_right_index, gripper_left_index, gripper_right_index, meshcat
        )
        self.drake_scene = scene

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

        inverse_kinematics_left_fn = partial(
            inverse_kinematics_in_world_fn, X_W_CB=X_W_LCB, tcp_transform=TCP_TRANSFORM
        )
        inverse_kinematics_right_fn = partial(
            inverse_kinematics_in_world_fn, X_W_CB=X_W_RCB, tcp_transform=TCP_TRANSFORM
        )

        # Save  these so the Controllers can create their own planners
        self.inverse_kinematics_left_fn = inverse_kinematics_left_fn
        self.inverse_kinematics_right_fn = inverse_kinematics_right_fn
        self.is_state_valid_fn = collision_checker.CheckConfigCollisionFree

        planner = DualArmOmplPlanner(
            is_state_valid_fn=collision_checker.CheckConfigCollisionFree,
            inverse_kinematics_left_fn=inverse_kinematics_left_fn,
            inverse_kinematics_right_fn=inverse_kinematics_right_fn,
            joint_bounds_left=self.joint_bounds_left,
            joint_bounds_right=self.joint_bounds_right,
        )
        return planner


if __name__ == "__main__":
    import cv2
    import rerun as rr
    from airo_camera_toolkit.utils.image_converter import ImageConverter
    from cloth_tools.visualization.opencv import draw_pose

    rr.init("Competition Station - Point cloud")
    rr.spawn(memory_limit="25%")

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

    # rr.spawn(memory_limit="25%")
    # rr.init("Competition Station - Point cloud")

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
