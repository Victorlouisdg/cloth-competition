from collections import deque

import scipy
from airo_typing import WrenchType


class WrenchSmoother:
    def __init__(self, history: int):
        self.history = history
        self.sigma = 2 * history + 1

        self.forces_x = deque(maxlen=history)
        self.forces_y = deque(maxlen=history)
        self.forces_z = deque(maxlen=history)
        self.torques_x = deque(maxlen=history)
        self.torques_y = deque(maxlen=history)
        self.torques_z = deque(maxlen=history)

        self.wrench_deques = [
            self.forces_x,
            self.forces_y,
            self.forces_z,
            self.torques_x,
            self.torques_y,
            self.torques_z,
        ]

    def smooth(self, wrench_deque: deque) -> float:
        return scipy.ndimage.gaussian_filter1d(wrench_deque, sigma=self.sigma)[-1]

    def smooth_wrench(self) -> WrenchType:
        wrench = []

        for wrench_deque in self.wrench_deques:
            wrench.append(self.smooth(wrench_deque))

        return wrench

    def add_wrench(self, wrench: WrenchType) -> WrenchType:
        for i, component in enumerate(wrench):
            self.wrench_deques[i].append(component)

        return self.smooth_wrench()
