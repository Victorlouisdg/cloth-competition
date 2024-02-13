import json
import os
from dataclasses import dataclass
from pathlib import Path

import cv2
import open3d as o3d
from airo_camera_toolkit.point_clouds.conversions import point_cloud_to_open3d
from airo_camera_toolkit.utils.image_converter import ImageConverter
from airo_dataset_tools.data_parsers.camera_intrinsics import CameraIntrinsics
from airo_dataset_tools.data_parsers.pose import Pose
from airo_typing import (
    CameraExtrinsicMatrixType,
    CameraIntrinsicsMatrixType,
    CameraResolutionType,
    HomogeneousMatrixType,
    JointConfigurationType,
    NumpyDepthMapType,
    NumpyIntImageType,
    PointCloud,
)
from loguru import logger


@dataclass
class CompetitionObservation:
    # Data
    image_left: NumpyIntImageType
    image_right: NumpyIntImageType
    depth_map: NumpyDepthMapType  # Depth should be aligned with the left image
    point_cloud: PointCloud  # Expressed in the world frame
    depth_image: NumpyIntImageType | None  # Optional depth image for visualization
    confidence_map: NumpyDepthMapType | None  # Confidence of the depth map as returned by the ZED SDK

    # Poses
    camera_pose_in_world: CameraExtrinsicMatrixType
    arm_left_pose_in_world: HomogeneousMatrixType
    arm_right_pose_in_world: HomogeneousMatrixType
    arm_left_joints: JointConfigurationType
    arm_right_joints: JointConfigurationType
    arm_left_tcp_pose_in_world: HomogeneousMatrixType
    arm_right_tcp_pose_in_world: HomogeneousMatrixType
    right_camera_pose_in_left_camera: HomogeneousMatrixType

    # Camera intrinsics
    camera_intrinsics_left: CameraIntrinsicsMatrixType
    camera_intrinsics_right: CameraIntrinsicsMatrixType
    camera_resolution: CameraResolutionType


# We put these in a dictionary so both the saving and loading code can use the same file names and extensions.
COMPETITION_OBSERVATION_FILENAMES = {
    "image_left": "image_left.png",
    "image_right": "image_right.png",
    "depth_map": "depth_map.tiff",
    "point_cloud": "point_cloud.ply",
    "depth_image": "depth_image.png",
    "confidence_map": "confidence_map.tiff",
    "camera_pose_in_world": "camera_pose_in_world.json",
    "arm_left_pose_in_world": "arm_left_pose_in_world.json",
    "arm_right_pose_in_world": "arm_right_pose_in_world.json",
    "arm_left_joints": "arm_left_joints.json",
    "arm_right_joints": "arm_right_joints.json",
    "arm_left_tcp_pose_in_world": "arm_left_tcp_pose_in_world.json",
    "arm_right_tcp_pose_in_world": "arm_right_tcp_pose_in_world.json",
    "right_camera_pose_in_left_camera": "right_camera_pose_in_left_camera.json",
    "camera_intrinsics_left": "camera_intrinsics_left.json",
    "camera_intrinsics_right": "camera_intrinsics_right.json",
    "camera_resolution": "camera_resolution.json",
}


def save_competition_observation(observation: CompetitionObservation, observation_dir: str) -> None:
    os.makedirs(observation_dir, exist_ok=True)

    # Turn filenames into filepaths
    filepaths = {
        key: str(Path(observation_dir) / filename) for key, filename in COMPETITION_OBSERVATION_FILENAMES.items()
    }

    # Convert images from RGB to BGR
    image_left = ImageConverter.from_numpy_int_format(observation.image_left).image_in_opencv_format
    image_right = ImageConverter.from_numpy_int_format(observation.image_right).image_in_opencv_format

    cv2.imwrite(filepaths["image_left"], image_left)
    cv2.imwrite(filepaths["image_right"], image_right)
    cv2.imwrite(filepaths["depth_map"], observation.depth_map)

    pcd = point_cloud_to_open3d(observation.point_cloud)
    o3d.t.io.write_point_cloud(filepaths["point_cloud"], pcd)

    if observation.confidence_map is not None:
        cv2.imwrite(filepaths["confidence_map"], observation.confidence_map)

    if observation.depth_image is not None:
        depth_image = ImageConverter.from_numpy_int_format(observation.depth_image).image_in_opencv_format
        cv2.imwrite(filepaths["depth_image"], depth_image)

    with open(filepaths["camera_intrinsics_left"], "w") as f:
        intrinsics_model_left = CameraIntrinsics.from_matrix_and_resolution(
            observation.camera_intrinsics_left, observation.camera_resolution
        )
        json.dump(intrinsics_model_left.model_dump(exclude_none=True), f, indent=4)

    with open(filepaths["camera_intrinsics_right"], "w") as f:
        intrinsics_model_right = CameraIntrinsics.from_matrix_and_resolution(
            observation.camera_intrinsics_right, observation.camera_resolution
        )
        json.dump(intrinsics_model_right.model_dump(exclude_none=True), f, indent=4)

    with open(filepaths["camera_pose_in_world"], "w") as f:
        camera_pose_model = Pose.from_homogeneous_matrix(observation.camera_pose_in_world)
        json.dump(camera_pose_model.model_dump(exclude_none=True), f, indent=4)

    with open(filepaths["arm_left_pose_in_world"], "w") as f:
        arm_left_pose_model = Pose.from_homogeneous_matrix(observation.arm_left_pose_in_world)
        json.dump(arm_left_pose_model.model_dump(exclude_none=True), f, indent=4)

    with open(filepaths["arm_right_pose_in_world"], "w") as f:
        arm_right_pose_model = Pose.from_homogeneous_matrix(observation.arm_right_pose_in_world)
        json.dump(arm_right_pose_model.model_dump(exclude_none=True), f, indent=4)

    with open(filepaths["arm_left_joints"], "w") as f:
        # json.dump(observation.arm_left_joints, f, indent=4)
        logger.warning("Not saving arm_left_joints")

    with open(filepaths["arm_right_joints"], "w") as f:
        # json.dump(observation.arm_right_joints, f, indent=4)
        logger.warning("Not saving arm_right_joints")

    with open(filepaths["arm_left_tcp_pose_in_world"], "w") as f:
        arm_left_tcp_pose_model = Pose.from_homogeneous_matrix(observation.arm_left_tcp_pose_in_world)
        json.dump(arm_left_tcp_pose_model.model_dump(exclude_none=True), f, indent=4)

    with open(filepaths["arm_right_tcp_pose_in_world"], "w") as f:
        arm_right_tcp_pose_model = Pose.from_homogeneous_matrix(observation.arm_right_tcp_pose_in_world)
        json.dump(arm_right_tcp_pose_model.model_dump(exclude_none=True), f, indent=4)


# def load_competition_input_sample(dataset_dir: str, sample_index: int) -> CompetitionObservation:
#     """Loads a competition input sample from a directory.

#     Args:
#         dataset_dir: The directory containing the sample directory.
#         sample_index: The index of the sample, must be unique per dataset.

#     Returns:
#         A CompetitionObservation instance.
#     """
#     sample_dir = Path(dataset_dir) / f"sample_{sample_index:06d}"
#     # filenames = competition_input_sample_filenames(sample_index)
#     # filepaths = {key: str(sample_dir / filename) for key, filename in filenames.items()}

#     image_left = cv2.imread(filepaths["image_left"])
#     image_right = cv2.imread(filepaths["image_right"])
#     depth_map = cv2.imread(filepaths["depth_map"], cv2.IMREAD_ANYDEPTH)
#     depth_image = cv2.imread(filepaths["depth_image"])
#     confidence_map = cv2.imread(filepaths["confidence_map"], cv2.IMREAD_ANYDEPTH)

#     with open(filepaths["camera_pose_in_world"], "r") as f:
#         camera_pose_in_world = Pose.model_validate_json(f.read()).as_homogeneous_matrix()

#     with open(filepaths["arm_left_pose_in_world"], "r") as f:
#         arm_left_pose_in_world = Pose.model_validate_json(f.read()).as_homogeneous_matrix()

#     with open(filepaths["arm_right_pose_in_world"], "r") as f:
#         arm_right_pose_in_world = Pose.model_validate_json(f.read()).as_homogeneous_matrix()

#     with open(filepaths["camera_intrinsics_left"], "r") as f:
#         intrinsics_model = CameraIntrinsics.model_validate_json(f.read())
#         camera_instrinsics_left = intrinsics_model.as_matrix()
#         camera_resolution = intrinsics_model.image_resolution.as_tuple()

#     with open(filepaths["camera_intrinsics_right"], "r") as f:
#         intrinsics_model = CameraIntrinsics.model_validate_json(f.read())
#         camera_instrinsics_right = intrinsics_model.as_matrix()

#     # Convert images from BGR to RGB
#     image_left = ImageConverter.from_opencv_format(image_left).image_in_numpy_int_format
#     image_right = ImageConverter.from_opencv_format(image_right).image_in_numpy_int_format
#     depth_image = ImageConverter.from_opencv_format(
#         depth_image
#     ).image_in_numpy_int_format  # in case it's not grayscale

#     pcd = o3d.t.io.read_point_cloud(filepaths["point_cloud"])
#     point_cloud = open3d_to_point_cloud(pcd)

#     return CompetitionObservation(
#         image_left=image_left,
#         image_right=image_right,
#         depth_map=depth_map,
#         point_cloud=point_cloud,
#         depth_image=depth_image,
#         confidence_map=confidence_map,
#         camera_pose_in_world=camera_pose_in_world,
#         arm_left_pose_in_world=arm_left_pose_in_world,
#         arm_right_pose_in_world=arm_right_pose_in_world,
#         camera_intrinsics_left=camera_instrinsics_left,
#         camera_intrinsics_right=camera_instrinsics_left,
#         camera_resolution=camera_resolution,
#     )


# def competition_input_sample_filenames(sample_index: int) -> dict[str, str]:
#     """Returns a dictionary of filenames for a given grasp index. Useful when collecting additional data.
#     The keys are the same as the fields of the corresponding dataclass and the values are the file names.
#     The data that is different for each grasp is suffixed with the zero-padded grasp index.

#     Args:
#         sample_index: The index of the sample, must be unique per dataset.

#     Returns:
#         A dictionary of filenames.
#     """
#     sample_index_padded = f"{sample_index:06d}"

#     return {
#         "image_left": f"image_left_{sample_index_padded}.png",
#         "image_right": f"image_right_{sample_index_padded}.png",
#         "depth_map": f"depth_map_{sample_index_padded}.tiff",
#         "point_cloud": f"point_cloud_{sample_index_padded}.ply",
#         "depth_image": f"depth_image_{sample_index_padded}.png",
#         "confidence_map": f"confidence_map_{sample_index_padded}.tiff",
#         "camera_pose_in_world": "camera_pose_in_world.json",
#         "arm_left_pose_in_world": "arm_left_pose_in_world.json",
#         "arm_right_pose_in_world": "arm_right_pose_in_world.json",
#         "camera_intrinsics_left": "camera_intrinsics_left.json",
#         "camera_intrinsics_right": "camera_intrinsics_right.json",
#         "camera_resolution": "camera_resolution.json",
#     }
