from typing import List

import numpy as np
from airo_camera_toolkit.cameras.zed.zed2i import Zed2i
from airo_typing import HomogeneousMatrixType, JointConfigurationType
from cloth_tools.config import load_camera_pose_in_left_and_right, setup_dual_arm_ur5e_in_world
from cloth_tools.drake.building import add_meshcat_to_builder, finish_build
from cloth_tools.drake.scenes import add_dual_ur5e_and_table_to_builder
from cloth_tools.ompl.dual_arm_planner import DualArmOmplPlanner
from cloth_tools.planning.interfaces import DualArmMotionPlanner
from cloth_tools.stations.dual_arm_station import DualArmStation
from loguru import logger
from pydrake.math import RigidTransform, RollPitchYaw
from pydrake.planning import RobotDiagramBuilder, SceneGraphCollisionChecker
from ur_analytic_ik import ur5e

# TODO make these things not hardcoded, but maybe load from extrinsics?
# Should be consistent with the Drake plant used for collision checking
tcp_transform = np.identity(4)
tcp_transform[2, 3] = 0.175

# LCB stands for "What the left control box considers to be the coordinate frame"
y_distance = 0.45
X_W_L = RigidTransform(rpy=RollPitchYaw([0, 0, np.pi / 2]), p=[0, y_distance, 0]).GetAsMatrix4()
X_W_R = RigidTransform(rpy=RollPitchYaw([0, 0, np.pi / 2]), p=[0, -y_distance, 0]).GetAsMatrix4()
X_CB_B = RigidTransform(rpy=RollPitchYaw([0, 0, np.pi]), p=[0, 0, 0]).GetAsMatrix4()
X_LCB_W = X_CB_B @ np.linalg.inv(X_W_L)
X_RCB_W = X_CB_B @ np.linalg.inv(X_W_R)


def left_inverse_kinematics_fn(tcp_pose: HomogeneousMatrixType) -> List[JointConfigurationType]:
    X_W_TCP = tcp_pose
    solutions_1x6 = ur5e.inverse_kinematics_with_tcp(X_LCB_W @ X_W_TCP, tcp_transform)
    solutions = [solution.squeeze() for solution in solutions_1x6]
    return solutions


def right_inverse_kinematics_fn(tcp_pose: HomogeneousMatrixType) -> List[JointConfigurationType]:
    X_W_TCP = tcp_pose
    solutions_1x6 = ur5e.inverse_kinematics_with_tcp(X_RCB_W @ X_W_TCP, tcp_transform)
    solutions = [solution.squeeze() for solution in solutions_1x6]
    return solutions


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

        X_LCB_C = camera_pose_in_left  # Note: we could average between the extrinsics of the left and right camera
        X_W_C = np.linalg.inv(X_LCB_W) @ X_LCB_C
        camera_pose = X_W_C  # this must be consistent with the setup_dual_arm_ur5e call below

        # Setting up the robots and grippers
        X_W_LCB = np.linalg.inv(X_LCB_W)
        X_W_RCB = np.linalg.inv(X_RCB_W)
        dual_arm = setup_dual_arm_ur5e_in_world(X_W_LCB, X_W_RCB)
        super().__init__(dual_arm, camera, camera_pose)

        # Adding additional attributes
        self.home_joints_left = np.deg2rad([180, -135, 95, -50, -90, -90])
        self.home_joints_right = np.deg2rad([-180, -45, -95, -130, 90, 90])

        # Planner for the two arms without obstacles (only the table)
        self.planner: DualArmMotionPlanner = self._setup_planner()

        # This is purely for visualization, but read the robot joints and publish them to meshcat
        diagram = self._diagram
        context = self._context
        arm_indices = self._arm_indices
        home_joints_left = self.home_joints_left
        home_joints_right = self.home_joints_right
        plant = diagram.plant()
        plant_context = plant.GetMyContextFromRoot(context)
        arm_left_index, arm_right_index = arm_indices
        plant.SetPositions(plant_context, arm_left_index, home_joints_left)
        plant.SetPositions(plant_context, arm_right_index, home_joints_right)
        diagram.ForcedPublish(context)

        logger.info("CompetitionStation initialized.")

    def _setup_planner(self) -> DualArmOmplPlanner:
        # Creating the default scene
        robot_diagram_builder = RobotDiagramBuilder()
        meshcat = add_meshcat_to_builder(robot_diagram_builder)
        arm_indices, gripper_indices = add_dual_ur5e_and_table_to_builder(robot_diagram_builder)
        diagram, context = finish_build(robot_diagram_builder, meshcat)

        collision_checker = SceneGraphCollisionChecker(
            model=diagram,
            robot_model_instances=[*arm_indices, *gripper_indices],
            edge_step_size=0.125,  # Arbitrary value: we don't use the CheckEdgeCollisionFree
            env_collision_padding=0.005,
            self_collision_padding=0.005,
        )

        is_state_valid_fn = collision_checker.CheckConfigCollisionFree

        # expose these things for visualization
        self._diagram = diagram
        self._context = context
        self._collision_checker = collision_checker
        self._meshcat = meshcat
        self._arm_indices = arm_indices
        self._gripper_indices = gripper_indices

        planner = DualArmOmplPlanner(
            is_state_valid_fn,
            left_inverse_kinematics_fn,
            right_inverse_kinematics_fn,
        )
        return planner


if __name__ == "__main__":
    import cv2
    from airo_camera_toolkit.utils.image_converter import ImageConverter
    from cloth_tools.visualization.opencv import draw_pose

    # Check whether all hardware is connected
    station = CompetitionStation()
    camera = station.camera
    dual_arm = station.dual_arm

    X_W_C = station.camera_pose
    X_C_W = np.linalg.inv(X_W_C)
    X_C_LCB = X_C_W @ np.linalg.inv(X_LCB_W)
    X_C_RCB = X_C_W @ np.linalg.inv(X_RCB_W)

    window_name = "Competiton station"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    while True:
        image_rgb = camera.get_rgb_image_as_int()
        image_bgr = ImageConverter.from_numpy_int_format(image_rgb).image_in_opencv_format

        X_CB_LTCP = dual_arm.left_manipulator.get_tcp_pose()
        X_CB_RTCP = dual_arm.right_manipulator.get_tcp_pose()

        X_C_LTCP = X_C_LCB @ X_CB_LTCP
        X_C_RTCP = X_C_RCB @ X_CB_RTCP

        # visualize X_C_W, X_C_LCB. X_C_RCB, X_C_LTCP, X_C_RTCP
        draw_pose(image_bgr, X_C_W, camera.intrinsics_matrix(), np.identity(4), 0.25)
        draw_pose(image_bgr, X_C_LCB, camera.intrinsics_matrix(), np.identity(4))
        draw_pose(image_bgr, X_C_RCB, camera.intrinsics_matrix(), np.identity(4))
        draw_pose(image_bgr, X_C_LTCP, camera.intrinsics_matrix(), np.identity(4), 0.05)
        draw_pose(image_bgr, X_C_RTCP, camera.intrinsics_matrix(), np.identity(4), 0.05)

        cv2.imshow(window_name, image_bgr)
        key = cv2.waitKey(1)
        if key == ord("q"):
            break
