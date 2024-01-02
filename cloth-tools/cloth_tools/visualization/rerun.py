import rerun as rr
from airo_camera_toolkit.interfaces import RGBCamera
from airo_typing import HomogeneousMatrixType


def rr_log_camera(camera: RGBCamera, camera_pose: HomogeneousMatrixType, entity_path: str = "world/camera") -> None:
    """Log the camera and its pose to Rerun.

    Args:
        camera: The camera.
        camera_pose: The pose of the camera in the world frame.
    """
    rr_pinhole = rr.Pinhole(image_from_camera=camera.intrinsics_matrix(), resolution=camera.resolution)
    rr_camera_pose = rr.Transform3D(translation=camera_pose[0:3, 3], mat3x3=camera_pose[0:3, 0:3])
    rr.log(entity_path, rr_pinhole)
    rr.log(entity_path, rr_camera_pose)
