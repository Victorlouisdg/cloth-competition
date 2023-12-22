from airo_camera_toolkit.cameras.zed2i import Zed2i
from airo_robots.manipulators.bimanual_position_manipulator import DualArmPositionManipulator
from airo_typing import CameraExtrinsicMatrixType


class DualArmStation:
    """A station is a collection of hardware, e.g. the robots, camera, but also configuration such camera extrinsics,
    or can even contain a simulation environment for motion planning etc.

    In this project we assume a dual arm station contains at least:
    * two robot arms (grouped into a DualArmPositionManipulator object)
    * a stereo RGBD camera (for now must be a ZED2i camera)
    * the pose of the camera in the world frame.

    The world frame is defined to be the base frame of the left robot.
    """

    def __init__(self, dual_arm: DualArmPositionManipulator, camera: Zed2i, camera_pose: CameraExtrinsicMatrixType):
        self.dual_arm = dual_arm
        self.camera = camera
        self.camera_pose = camera_pose
