import time
from collections import deque

import rerun as rr
import scipy
from cloth_tools.stations.competition_station import CompetitionStation
from loguru import logger


def smooth(wrenches, sigma):
    return scipy.ndimage.gaussian_filter1d(wrenches, sigma=sigma)[-1]


if __name__ == "__main__":
    station = CompetitionStation()

    rr.init("Force measurement", spawn=True)

    history = 100
    sigma = 2 * history + 1  # make it odd

    forces_left_x = deque(maxlen=history)
    forces_left_y = deque(maxlen=history)
    forces_left_z = deque(maxlen=history)
    torques_left_x = deque(maxlen=history)
    torques_left_y = deque(maxlen=history)
    torques_left_z = deque(maxlen=history)
    forces_right_x = deque(maxlen=history)
    forces_right_y = deque(maxlen=history)
    forces_right_z = deque(maxlen=history)
    torques_right_x = deque(maxlen=history)
    torques_right_y = deque(maxlen=history)
    torques_right_z = deque(maxlen=history)

    wrenches_left = [forces_left_x, forces_left_y, forces_left_z, torques_left_x, torques_left_y, torques_left_z]
    wrenches_right = [
        forces_right_x,
        forces_right_y,
        forces_right_z,
        torques_right_x,
        torques_right_y,
        torques_right_z,
    ]

    while True:
        time_start = time.time()
        wrench_left = station.dual_arm.left_manipulator.rtde_receive.getActualTCPForce()
        wrench_right = station.dual_arm.right_manipulator.rtde_receive.getActualTCPForce()
        logger.info(f"Time to receive wrench: {time.time() - time_start:.5f}")

        for i, wrench in enumerate(wrenches_left):
            wrench.append(wrench_left[i])

        for i, wrench in enumerate(wrenches_right):
            wrench.append(wrench_right[i])

        wrench_smoothed_left = []

        for wrench in wrenches_left:
            wrench_smoothed_left.append(smooth(wrench, sigma))

        wrench_smoothed_right = []

        for wrench in wrenches_right:
            wrench_smoothed_right.append(smooth(wrench, sigma))

        logger.info(f"Time to smooth wrench: {time.time() - time_start:.4f}")

        # log each scalar
        for i, label in zip(range(3), ["Fx", "Fy", "Fz"]):
            rr.log(f"/force/left/{label}", rr.Scalar(wrench_smoothed_left[i]))
            rr.log(f"/force/right/{label}", rr.Scalar(wrench_smoothed_right[i]))

        for i, label in zip(range(3, 6), ["Tx", "Ty", "Tz"]):
            rr.log(f"/torque/left/{label}", rr.Scalar(wrench_smoothed_left[i]))
            rr.log(f"/torque/right/{label}", rr.Scalar(wrench_smoothed_right[i]))

        # Hardcode that tension is measure along the x-axis of the robot bases.
        tension = (wrench_smoothed_left[0] - wrench_smoothed_right[0]) / 2.0
        rr.log("/force/tension", rr.Scalar(tension))
