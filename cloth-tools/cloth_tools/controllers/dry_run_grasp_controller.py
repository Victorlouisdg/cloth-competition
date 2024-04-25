import os
import shutil
import sys
import time
from functools import partial
from pathlib import Path

import cv2
from airo_dataset_tools.data_parsers.pose import Pose
from airo_drake import animate_dual_joint_trajectory
from airo_planner import DualArmOmplPlanner
from airo_typing import HomogeneousMatrixType
from cloth_tools.controllers.controller import Controller
from cloth_tools.dataset.format import load_competition_observation
from cloth_tools.drake.scenes import make_drake_scene_from_observation, make_dual_arm_collision_checker
from cloth_tools.kinematics.constants import JOINT_BOUNDS, TCP_TRANSFORM
from cloth_tools.kinematics.inverse_kinematics import inverse_kinematics_in_world_fn
from cloth_tools.planning.grasp_planning import plan_pregrasp_and_grasp_trajectory
from cloth_tools.stations.competition_station import CompetitionStation
from cloth_tools.trajectory_execution import execute_dual_arm_drake_trajectory
from cloth_tools.visualization.opencv import draw_pose
from loguru import logger


def get_grasp_confirmation(
    grasp_pose: HomogeneousMatrixType,
    observation,
):
    image_copy = observation.image_left.copy()
    image_annotated = cv2.cvtColor(image_copy, cv2.COLOR_RGB2BGR)
    draw_pose(image_annotated, grasp_pose, observation.camera_intrinsics, observation.camera_pose_in_world)

    window_name = "Confirm grasp? Press 'y' for yes, 'n' for no, 'q' to quit."
    logger.info(f"{window_name} in OpenCV window.")
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1280, 720)

    while True:
        cv2.imshow(window_name, image_annotated)
        key = cv2.waitKey(0)
        if key == ord("y"):
            logger.success("Grasp pose confirmed.")
            return True
        elif key == ord("n"):
            logger.warning("Grasp pose rejected.")
            return False
        elif key == ord("q"):
            sys.exit(0)


class DryRunGraspController(Controller):
    """Controller that handles the execution of grasp uploaded to the server."""

    def __init__(self, station: CompetitionStation, sample_dir: str, grasps_dir: str):
        self.station = station

        self.sample_dir = sample_dir
        self.grasps_dir = grasps_dir

        observation_start_dir = Path(self.sample_dir) / "observation_start"
        observation_start = load_competition_observation(observation_start_dir)
        self.observation = observation_start

    def wait_for_plannable_grasp(self):
        self.station
        self.station.dual_arm
        observation = self.observation
        grasps_dir = self.grasps_dir

        # Creating the reusable planning components
        scene = make_drake_scene_from_observation(observation, include_cloth_obstacle=False)
        scene_with_cloth = make_drake_scene_from_observation(observation, include_cloth_obstacle=True)

        X_W_LCB = observation.arm_left_pose_in_world
        X_W_RCB = observation.arm_right_pose_in_world

        inverse_kinematics_left_fn = partial(
            inverse_kinematics_in_world_fn, X_W_CB=X_W_LCB, tcp_transform=TCP_TRANSFORM
        )
        inverse_kinematics_right_fn = partial(
            inverse_kinematics_in_world_fn, X_W_CB=X_W_RCB, tcp_transform=TCP_TRANSFORM
        )

        collision_checker_no_cloth = make_dual_arm_collision_checker(scene)
        collision_checker_with_cloth = make_dual_arm_collision_checker(scene_with_cloth)

        planner_pregrasp = DualArmOmplPlanner(
            is_state_valid_fn=collision_checker_with_cloth.CheckConfigCollisionFree,
            inverse_kinematics_left_fn=inverse_kinematics_left_fn,
            inverse_kinematics_right_fn=inverse_kinematics_right_fn,
            joint_bounds_left=JOINT_BOUNDS,
            joint_bounds_right=JOINT_BOUNDS,
        )

        # Trying the planning with different grasps

        logger.info(f"Waiting for grasp poses in: {grasps_dir}")

        # Move the arms to their home positions
        failed_files = set()

        while True:
            grasp_files = set(os.listdir(grasps_dir))
            files_to_consider = grasp_files - failed_files

            if not files_to_consider:
                logger.info(f"No grasp poses received yet, waiting... (len(failed_files)={len(failed_files)}")
                time.sleep(1.0)
                continue

            grasp_files = sorted(list(files_to_consider), reverse=True)
            latest_file = grasp_files[0]

            logger.info(f"Trying grasp pose from: {latest_file}")

            filepath = os.path.join(grasps_dir, latest_file)

            try:
                with open(filepath, "r") as f:
                    grasp_pose = Pose.model_validate_json(f.read()).as_homogeneous_matrix()

                trajectory_pregrasp_and_grasp = plan_pregrasp_and_grasp_trajectory(
                    planner_pregrasp,
                    grasp_pose,
                    observation.arm_left_joints,
                    observation.arm_right_joints,
                    inverse_kinematics_left_fn,
                    inverse_kinematics_right_fn,
                    collision_checker_no_cloth.CheckConfigCollisionFree,
                    scene.robot_diagram.plant(),
                    with_left=False,
                )

                animate_dual_joint_trajectory(
                    scene_with_cloth.meshcat,
                    scene_with_cloth.robot_diagram,
                    scene_with_cloth.arm_left_index,
                    scene_with_cloth.arm_right_index,
                    trajectory_pregrasp_and_grasp,
                )

                logger.success(f"You can see the trajectory animation at: {scene_with_cloth.meshcat.web_url()}")
                if get_grasp_confirmation(grasp_pose, observation):
                    logger.success(f"Executing grasp pose from: {latest_file}")

                    # TODO copy the successful grasp pose from grasps dir to the sample dir
                    grasp_dir = Path(self.sample_dir) / "grasp"
                    grasp_dir.mkdir(parents=True, exist_ok=True)
                    grasp_pose_file = grasp_dir / "grasp_pose.json"
                    # copy filepath to grasp_pose_file
                    shutil.copy2(filepath, grasp_pose_file)

                    # TODO also save image with grasp
                    image_copy = observation.image_left.copy()
                    image_annotated = cv2.cvtColor(image_copy, cv2.COLOR_RGB2BGR)
                    draw_pose(
                        image_annotated, grasp_pose, observation.camera_intrinsics, observation.camera_pose_in_world
                    )

                    grasp_image_file = grasp_dir / "frontal_image_grasp.jpg"
                    cv2.imwrite(str(grasp_image_file), image_annotated)

                    # Execute the grasp
                    dual_arm = self.station.dual_arm
                    execute_dual_arm_drake_trajectory(dual_arm, trajectory_pregrasp_and_grasp)
                    dual_arm.right_manipulator.gripper.close().wait()

                    return
                else:
                    logger.warning("Grasp pose rejected.")
                    failed_files.add(latest_file)
                    continue

            except Exception as e:
                failed_files.add(latest_file)
                logger.warning(f"Cannot load/plan/execute {latest_file}: {e}")
                time.sleep(0.0001)
                continue

    def execute(self) -> None:
        logger.info(f"{self.__class__.__name__} started.")
        self.wait_for_plannable_grasp()
        logger.info(f"{self.__class__.__name__} finished.")


if __name__ == "__main__":
    from cloth_tools.stations.competition_station import CompetitionStation

    station = CompetitionStation()

    # Hardcoded dirs for testing

    dataset_dir = Path("/home/victor/cloth-competition/notebooks/data/remote_dry_run_2024-04-26/dev_team")
    sample_id = "2024-04-23_10-20-07-968516"

    sample_dir = dataset_dir / f"sample_{sample_id}"
    grasps_dir = dataset_dir / f"grasps_{sample_id}"

    print(os.path.exists(sample_dir))
    print(os.path.exists(grasps_dir))

    controller = DryRunGraspController(station, sample_dir, grasps_dir)
    controller.execute()

    # TODO here: On my laptop: upload a few grasps
