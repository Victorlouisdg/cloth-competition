import numpy as np

TCP_TRANSFORM = np.identity(4)
TCP_TRANSFORM[2, 3] = 0.175

JOINT_BOUNDS_LOWER = np.deg2rad([-360, -195, -160, -360, -360, -360])
JOINT_BOUNDS_UPPER = np.deg2rad([360, 15, 160, 360, 360, 360])
JOINT_BOUNDS = JOINT_BOUNDS_LOWER, JOINT_BOUNDS_UPPER
