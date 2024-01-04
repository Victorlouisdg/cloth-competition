from typing import Callable, List

import numpy as np
from ompl import base as ob
from ompl import geometric as og


def ompl_state_to_numpy(state: ob.State, n: int) -> np.ndarray:
    return np.array([state[i] for i in range(n)])


def numpy_to_ompl_state(state_numpy: np.ndarray, space: ob.StateSpace) -> ob.State:
    state = ob.State(space)
    for i in range(len(state_numpy)):
        state()[i] = state_numpy[i]
    return state


def ompl_path_to_numpy(path: og.PathGeometric, n: int) -> List[np.ndarray]:
    return [ompl_state_to_numpy(path.getState(i), n) for i in range(path.getStateCount())]


def function_numpy_to_ompl(function: Callable, n: int):
    """Take a function that has a single numpy array as input,
    and wrap it so that it takes an OMPL state as input instead."""

    def wrapper(ompl_state: ob.State):
        numpy_state = ompl_state_to_numpy(ompl_state, n)
        return function(numpy_state)

    return wrapper


def _joints_state_space(n: int) -> ob.RealVectorStateSpace:
    space = ob.RealVectorStateSpace(n)
    bounds = ob.RealVectorBounds(n)
    bounds.setLow(-2 * np.pi)
    bounds.setHigh(2 * np.pi)
    space.setBounds(bounds)
    return space


def single_arm_state_space() -> ob.RealVectorStateSpace:
    return _joints_state_space(6)


def dual_arm_state_space() -> ob.RealVectorStateSpace:
    return _joints_state_space(12)
