from airo_camera_toolkit.interfaces import StereoRGBDCamera
from airo_camera_toolkit.point_clouds.conversions import open3d_to_point_cloud, point_cloud_to_open3d
from cloth_tools.dataset.format import CompetitionObservation
from cloth_tools.stations.competition_station import CompetitionStation


def collect_observation(station: CompetitionStation) -> CompetitionObservation:
    camera = station.camera

    camera_pose_in_world = station.camera_pose
    arm_left_pose_in_world = station.left_arm_pose
    arm_right_pose_in_world = station.right_arm_pose
    right_camera_pose_in_left_camera = camera.pose_of_right_view_in_left_view

    camera_intrinsics = camera.intrinsics_matrix()
    camera_resolution = camera.resolution

    image_left = camera.get_rgb_image_as_int()
    image_right = camera._retrieve_rgb_image_as_int(StereoRGBDCamera.RIGHT_RGB)
    depth_map = camera._retrieve_depth_map()
    depth_image = camera._retrieve_depth_image()
    confidence_map = camera._retrieve_confidence_map()

    point_cloud_in_camera = camera._retrieve_colored_point_cloud()
    pcd_in_camera = point_cloud_to_open3d(point_cloud_in_camera)
    pcd = pcd_in_camera.transform(camera_pose_in_world)  # transform to world frame (= base frame of left robot)
    point_cloud = open3d_to_point_cloud(pcd)

    arm_left_tcp_pose_in_base = station.dual_arm.left_manipulator.get_tcp_pose()
    arm_left_joints = station.dual_arm.left_manipulator.get_joint_configuration()
    arm_right_tcp_pose_in_base = station.dual_arm.right_manipulator.get_tcp_pose()
    arm_right_joints = station.dual_arm.right_manipulator.get_joint_configuration()

    # Convert TCP poses from robot base frames to world frame
    X_LCB_TCPL = arm_left_tcp_pose_in_base
    X_RCB_TCPR = arm_right_tcp_pose_in_base

    X_W_LCB = arm_left_pose_in_world
    X_W_RCB = arm_right_pose_in_world

    X_W_TCPL = X_W_LCB @ X_LCB_TCPL
    X_W_TCPR = X_W_RCB @ X_RCB_TCPR

    arm_left_tcp_pose_in_world = X_W_TCPL
    arm_right_tcp_pose_in_world = X_W_TCPR

    observation = CompetitionObservation(
        image_left=image_left,
        image_right=image_right,
        depth_map=depth_map,
        point_cloud=point_cloud,
        depth_image=depth_image,
        confidence_map=confidence_map,
        camera_pose_in_world=camera_pose_in_world,
        arm_left_pose_in_world=arm_left_pose_in_world,
        arm_right_pose_in_world=arm_right_pose_in_world,
        arm_left_joints=arm_left_joints,
        arm_right_joints=arm_right_joints,
        arm_left_tcp_pose_in_world=arm_left_tcp_pose_in_world,
        arm_right_tcp_pose_in_world=arm_right_tcp_pose_in_world,
        right_camera_pose_in_left_camera=right_camera_pose_in_left_camera,
        camera_intrinsics=camera_intrinsics,
        camera_resolution=camera_resolution,
    )
    return observation
