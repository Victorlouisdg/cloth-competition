import numpy as np
from airo_camera_toolkit.cameras.zed2i import Zed2i
from cloth_tools.config import load_camera_pose_in_left_and_right, setup_dual_arm_ur5e
from cloth_tools.stations.dual_arm_station import DualArmStation
from loguru import logger


class CompetitionStation(DualArmStation):
    """
    This station specifically contains the hardware setup for the ICRA 2024 competition.
    It (currently) consists of two UR5e robots and a single ZED2i camera.
    The robots are mounted approximately 0.9 meter apart.
    """

    def __init__(self) -> None:
        # Setting up the camera
        # TODO start multiprocessed camera here and add video recorders etc.
        camera = Zed2i(resolution=Zed2i.RESOLUTION_2K, depth_mode=Zed2i.NEURAL_DEPTH_MODE, fps=15)
        camera_pose_in_left, camera_pose_in_right = load_camera_pose_in_left_and_right()
        camera_pose = camera_pose_in_left  # this must be consistent with the setup_dual_arm_ur5e call below

        # Setting up the robots and grippers
        dual_arm = setup_dual_arm_ur5e(camera_pose_in_left, camera_pose_in_right)
        super().__init__(dual_arm, camera, camera_pose)

        # Adding additional attributes
        self.home_joints_left = np.deg2rad([180, -135, 95, -50, -90, -90])
        self.home_joints_right = np.deg2rad([-180, -45, -95, -130, 90, 90])

        # TODO add attribute for access to the drake scene / motion planning here

        logger.info("CompetitionStation initialized.")


if __name__ == "__main__":
    # Check whether all hardware is connected
    station = CompetitionStation()
