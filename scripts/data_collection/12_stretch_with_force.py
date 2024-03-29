import sys
import time

import numpy as np
import rerun as rr
from cloth_tools.controllers.grasp_highest_controller import hang_in_the_air_tcp_pose
from cloth_tools.stations.competition_station import CompetitionStation
from cloth_tools.trajectory_execution import execute_dual_arm_trajectory, time_parametrize_toppra
from cloth_tools.wrench_smoother import WrenchSmoother
from linen.elemental.move_backwards import move_pose_backwards
from loguru import logger


def move_to_stretch_pose(station: CompetitionStation):
    dual_arm = station.dual_arm
    planner = station.planner

    hang_pose_left = hang_in_the_air_tcp_pose(left=True)
    hang_pose_right = hang_in_the_air_tcp_pose(left=False)

    tcp_distance = 0.1

    backwards_shift = tcp_distance / 2.0  # keep end distance same as distance at grasp
    stretch_pose_left = move_pose_backwards(hang_pose_left, backwards_shift)
    stretch_pose_right = move_pose_backwards(hang_pose_right, backwards_shift)

    # Move arms closer to the camera
    global_x_shift = -0.4
    stretch_pose_left[:3, 3] += np.array([global_x_shift, 0, 0])
    stretch_pose_right[:3, 3] += np.array([global_x_shift, 0, 0])

    start_joints_left = dual_arm.left_manipulator.get_joint_configuration()
    start_joints_right = dual_arm.right_manipulator.get_joint_configuration()

    path_to_stretch = planner.plan_to_tcp_pose(
        start_joints_left, start_joints_right, stretch_pose_left, stretch_pose_right
    )

    if path_to_stretch is None:
        logger.info("No path found to stretch pose")
        sys.exit(0)

    # Time parametrize the path
    # Move very slowly
    trajectory_to_stretch, time_trajectory_to_stretch = time_parametrize_toppra(
        path_to_stretch, station._diagram.plant(), joint_speed_limit=0.5, joint_acceleration_limit=0.5
    )

    # Execute the path to the stretch pose
    execute_dual_arm_trajectory(dual_arm, trajectory_to_stretch, time_trajectory_to_stretch)


if __name__ == "__main__":
    rr.init("Force measurement", spawn=True)

    station = CompetitionStation()
    dual_arm = station.dual_arm

    X_W_LCB = station.left_arm_pose
    X_W_RCB = station.right_arm_pose

    move_to_stretch_pose(station)

    time.sleep(5)  # let 5 seconds pass to settle the forces of the movement

    dual_arm.left_manipulator.rtde_control.zeroFtSensor()
    dual_arm.right_manipulator.rtde_control.zeroFtSensor()

    tension_threshold = 4.0
    tcp_distance_threshold = 0.9  # never move more that this distance apart

    has_servo_distance_decreased = False

    servo_time = 0.05  # seconds
    servo_speed_start = 0.05  # meters per second starting fast result in a shock in the F/T readings
    # servo_speed_untensioned = 0.25  # meters per second
    # servo_speed_tensioned = 0.05  # meters per second

    servo_distance_start = servo_speed_start * servo_time
    # servo_distance_untensioned = servo_speed_untensioned * servo_time
    # servo_distance_tensioned = servo_speed_tensioned * servo_time

    servo_distance = servo_distance_start

    servos_per_second = 1.0 / servo_time
    history = int(servos_per_second * 2)  # smooth over last 2 seconds
    history = max(history, 1)

    logger.info("Smoothing over a history of {} samples".format(history))

    wrench_smoother_left = WrenchSmoother(history=history)
    wrench_smoother_right = WrenchSmoother(history=history)

    servo_awaitable = None

    # adjusting servo speed/distance at runtime results in F/T measurement shocks
    # low_tension_counter = 0
    # low_tension_counter_max = 10

    while True:
        wrench_left = dual_arm.left_manipulator.rtde_receive.getActualTCPForce()
        wrench_right = dual_arm.right_manipulator.rtde_receive.getActualTCPForce()

        wrench_smoothed_left = wrench_smoother_left.add_wrench(wrench_left)
        wrench_smoothed_right = wrench_smoother_right.add_wrench(wrench_right)

        # log each scalar
        for i, label in zip(range(3), ["Fx", "Fy", "Fz"]):
            rr.log(f"/force/left/{label}", rr.Scalar(wrench_smoothed_left[i]))
            rr.log(f"/force/right/{label}", rr.Scalar(wrench_smoothed_right[i]))

        for i, label in zip(range(3, 6), ["Tx", "Ty", "Tz"]):
            rr.log(f"/torque/left/{label}", rr.Scalar(wrench_smoothed_left[i]))
            rr.log(f"/torque/right/{label}", rr.Scalar(wrench_smoothed_right[i]))

        tension = (wrench_smoothed_left[0] - wrench_smoothed_right[0]) / 2.0
        logger.info(f"Tension: {tension:.1f} N")
        rr.log("/force/tension", rr.Scalar(tension))

        if tension > tension_threshold:
            logger.info(f"Tension reached threshold {tension:.2f} N")
            dual_arm.left_manipulator.rtde_control.servoStop()
            dual_arm.right_manipulator.rtde_control.servoStop()
            break

        X_LCB_TCP = dual_arm.left_manipulator.get_tcp_pose()
        X_RCB_TCP = dual_arm.right_manipulator.get_tcp_pose()
        X_W_LTCP = X_W_LCB @ X_LCB_TCP
        X_W_RTCP = X_W_RCB @ X_RCB_TCP
        tcp_distance = np.linalg.norm(X_W_LTCP[:3, 3] - X_W_RTCP[:3, 3])

        if tcp_distance > tcp_distance_threshold:
            logger.info(f"TCP distance reached threshold {tcp_distance:.2f} m")
            dual_arm.left_manipulator.rtde_control.servoStop()
            dual_arm.right_manipulator.rtde_control.servoStop()
            break

        # if tension < 3.0:
        #     low_tension_counter += 1

        # if low_tension_counter > low_tension_counter_max and servo_distance < servo_distance_untensioned:
        #     logger.info(f"Increasing servo slowly distance from {servo_distance} to {servo_distance_untensioned}")
        #     servo_distance += (servo_distance_untensioned - servo_distance_tensioned) / 10.0
        #     if servo_distance >= servo_distance_untensioned:
        #         servo_distance = servo_distance_untensioned
        #         logger.info(f"Servo distance reached max: {servo_distance}")
        #     has_servo_distance_decreased = False
        #     low_tension_counter = 0

        # if tension > 3.0 and not has_servo_distance_decreased:
        #     logger.info(f"Decreasing servo distance from {servo_distance} to {servo_distance_tensioned}")
        #     servo_distance = servo_distance_tensioned
        #     has_servo_distance_decreased = True
        #     low_tension_counter = 0

        # X_servo_L = move_pose_backwards(X_W_LTCP, servo_distance)
        # X_servo_R = move_pose_backwards(X_W_RTCP, servo_distance)

        # if servo_awaitable is not None:
        #     servo_awaitable.wait()

        # servo_awaitable = dual_arm.servo_to_tcp_pose(X_servo_L, X_servo_R, time=servo_time)  # about 1 cm/s
