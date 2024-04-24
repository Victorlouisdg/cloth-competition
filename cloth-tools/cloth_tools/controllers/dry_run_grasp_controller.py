import os
import time
from functools import partial

from airo_dataset_tools.data_parsers.pose import Pose
from airo_drake import animate_dual_joint_trajectory
from airo_planner import DualArmOmplPlanner
from cloth_tools.controllers.controller import Controller
from cloth_tools.dataset.collection import collect_observation
from cloth_tools.dataset.format import CompetitionObservation
from cloth_tools.drake.scenes import make_drake_scene_from_observation, make_dual_arm_collision_checker
from cloth_tools.kinematics.constants import JOINT_BOUNDS, TCP_TRANSFORM
from cloth_tools.kinematics.inverse_kinematics import inverse_kinematics_in_world_fn
from cloth_tools.planning.grasp_planning import plan_pregrasp_and_grasp_trajectory
from cloth_tools.stations.competition_station import CompetitionStation
from loguru import logger


class DryRunGraspController(Controller):
    """Controller that handles the execution of grasp uploaded to the server."""

    def __init__(self, station: CompetitionStation, observation_start: CompetitionObservation, grasps_dir: str):
        self.station = station
        self.observation = observation_start
        self.grasps_dir = grasps_dir

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
                logger.info("No grasp poses received yet, waiting...")
                time.sleep(1.0)
                continue

            grasp_files = sorted(list(files_to_consider), reverse=True)
            latest_file = grasp_files[0]

            logger.info(f"Trying grasp pose from: {latest_file}")

            filepath = os.path.join(grasps_dir, latest_file)

            try:
                with open(filepath, "r") as f:
                    grasp_pose = Pose.model_validate_json(f.read()).as_homogeneous_matrix()
                # TODO call plan_pregrasp_and_grasp_trajectory here

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

                print(f"You can see the trajectory animation at: {scene_with_cloth.meshcat.web_url()}")

            except Exception as e:
                failed_files.add(latest_file)
                logger.warning(f"Cannot load {latest_file}: {e}")
                time.sleep(0.0001)
                continue

            logger.success(f"Loaded grasp pose successfully from: {latest_file}")
            break

    def execute(self) -> None:
        logger.info(f"{self.__class__.__name__} started.")
        self.wait_for_plannable_grasp()
        logger.info(f"{self.__class__.__name__} finished.")


if __name__ == "__main__":
    from cloth_tools.stations.competition_station import CompetitionStation

    station = CompetitionStation()

    # Dummy observation and hardcoded grasps dir for testing
    observation = collect_observation(station)
    grasps_dir = "/home/victor/cloth-competition/notebooks/data/remote_dry_run_2024-04-26/dev_team/grasps_2024-04-23_10-20-07-968516"

    print(os.path.exists(grasps_dir))

    controller = DryRunGraspController(station, observation, grasps_dir)
    controller.execute()

    # TODO here: On my laptop: upload a few grasps
