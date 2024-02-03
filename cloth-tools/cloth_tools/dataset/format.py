import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import open3d as o3d
from airo_camera_toolkit.point_clouds.conversions import open3d_to_point_cloud, point_cloud_to_open3d
from airo_camera_toolkit.utils.image_converter import ImageConverter
from airo_dataset_tools.data_parsers.camera_intrinsics import CameraIntrinsics
from airo_dataset_tools.data_parsers.pose import Pose
from airo_typing import (
    CameraExtrinsicMatrixType,
    CameraIntrinsicsMatrixType,
    CameraResolutionType,
    NumpyDepthMapType,
    NumpyIntImageType,
    PointCloud,
)


@dataclass
class CompetitionInputSample:
    image_left: NumpyIntImageType
    image_right: NumpyIntImageType
    depth_map: NumpyDepthMapType
    point_cloud: PointCloud
    depth_image: NumpyIntImageType | None  # Optional depth image for visualization
    confidence_map: NumpyDepthMapType | None  # Confidence of the depth map as returned by the ZED SDK
    camera_pose: CameraExtrinsicMatrixType
    camera_intrinsics: CameraIntrinsicsMatrixType
    camera_resolution: CameraResolutionType


def save_competition_input_sample(sample: CompetitionInputSample, dataset_dir: str, sample_index: int) -> None:
    sample_dir = Path(dataset_dir) / f"sample_{sample_index:06d}"
    sample_dir.mkdir(parents=True, exist_ok=True)
    filenames = competition_input_sample_filenames(sample_index)
    filepaths = {key: str(sample_dir / filename) for key, filename in filenames.items()}

    # Convert images from RGB to BGR
    image_left = ImageConverter.from_numpy_int_format(sample.image_left).image_in_opencv_format
    image_right = ImageConverter.from_numpy_int_format(sample.image_right).image_in_opencv_format

    cv2.imwrite(filepaths["image_left"], image_left)
    cv2.imwrite(filepaths["image_right"], image_right)
    cv2.imwrite(filepaths["depth_map"], sample.depth_map)

    if sample.confidence_map is not None:
        cv2.imwrite(filepaths["confidence_map"], sample.confidence_map)

    if sample.depth_image is not None:
        depth_image = ImageConverter.from_numpy_int_format(sample.depth_image).image_in_opencv_format
        cv2.imwrite(filepaths["depth_image"], depth_image)

    with open(filepaths["camera_intrinsics"], "w") as f:
        json.dump(
            CameraIntrinsics.from_matrix_and_resolution(sample.camera_intrinsics, sample.camera_resolution).model_dump(
                exclude_none=True
            ),
            f,
            indent=4,
        )

    with open(filepaths["camera_pose"], "w") as f:
        json.dump(
            Pose.from_homogeneous_matrix(sample.camera_pose).model_dump(exclude_none=True),
            f,
            indent=4,
        )

    pcd = point_cloud_to_open3d(sample.point_cloud)
    o3d.t.io.write_point_cloud(filepaths["point_cloud"], pcd)


def load_competition_input_sample(dataset_dir: str, sample_index: int) -> CompetitionInputSample:
    """Loads a competition input sample from a directory.

    Args:
        dataset_dir: The directory containing the sample directory.
        sample_index: The index of the sample, must be unique per dataset.

    Returns:
        A CompetitionInputSample instance.
    """
    sample_dir = Path(dataset_dir) / f"sample_{sample_index:06d}"
    filenames = competition_input_sample_filenames(sample_index)
    filepaths = {key: str(sample_dir / filename) for key, filename in filenames.items()}

    image_left = cv2.imread(filepaths["image_left"])
    image_right = cv2.imread(filepaths["image_right"])
    depth_map = cv2.imread(filepaths["depth_map"], cv2.IMREAD_ANYDEPTH)
    depth_image = cv2.imread(filepaths["depth_image"])
    confidence_map = cv2.imread(filepaths["confidence_map"], cv2.IMREAD_ANYDEPTH)

    with open(filepaths["camera_pose"], "r") as f:
        camera_pose = Pose.model_validate_json(f.read()).as_homogeneous_matrix()

    with open(filepaths["camera_intrinsics"], "r") as f:
        intrinsics_model = CameraIntrinsics.model_validate_json(f.read())
        camera_instrinsics = intrinsics_model.as_matrix()
        camera_resolution = intrinsics_model.image_resolution.as_tuple()

    # Convert images from BGR to RGB
    image_left = ImageConverter.from_opencv_format(image_left).image_in_numpy_int_format
    image_right = ImageConverter.from_opencv_format(image_right).image_in_numpy_int_format
    depth_image = ImageConverter.from_opencv_format(
        depth_image
    ).image_in_numpy_int_format  # in case it's not grayscale

    pcd = o3d.t.io.read_point_cloud(filepaths["point_cloud"])
    point_cloud = open3d_to_point_cloud(pcd)

    return CompetitionInputSample(
        image_left=image_left,
        image_right=image_right,
        depth_map=depth_map,
        point_cloud=point_cloud,
        depth_image=depth_image,
        confidence_map=confidence_map,
        camera_pose=camera_pose,
        camera_intrinsics=camera_instrinsics,
        camera_resolution=camera_resolution,
    )


def competition_input_sample_filenames(sample_index: int) -> dict[str, str]:
    """Returns a dictionary of filenames for a given grasp index. Useful when collecting additional data.
    The keys are the same as the fields of the corresponding dataclass and the values are the file names.
    The data that is different for each grasp is suffixed with the zero-padded grasp index.

    Args:
        sample_index: The index of the sample, must be unique per dataset.

    Returns:
        A dictionary of filenames.
    """
    sample_index_padded = f"{sample_index:06d}"

    return {
        "image_left": f"image_left_{sample_index_padded}.png",
        "image_right": f"image_right_{sample_index_padded}.png",
        "depth_map": f"depth_map_{sample_index_padded}.tiff",
        "point_cloud": f"point_cloud_{sample_index_padded}.ply",
        "depth_image": f"depth_image_{sample_index_padded}.png",
        "confidence_map": f"confidence_map_{sample_index_padded}.tiff",
        "camera_pose": "camera_pose.json",
        "camera_intrinsics": "camera_intrinsics.json",
        "camera_resolution": "camera_resolution.json",
    }
