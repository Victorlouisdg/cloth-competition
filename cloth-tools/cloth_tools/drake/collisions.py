from typing import Sequence

import numpy as np
from pydrake.multibody.tree import ModelInstanceIndex
from pydrake.planning import SceneGraphCollisionChecker
from pydrake.systems.framework import Diagram


# Should be investigate purpose of this function further
def _configuration_distance(q1: np.ndarray, q2: np.ndarray) -> float:
    return np.linalg.norm(q1 - q2)


def get_collision_checker(
    diagram: Diagram, robot_model_instances: Sequence[ModelInstanceIndex]
) -> SceneGraphCollisionChecker:
    collision_checker = SceneGraphCollisionChecker(
        model=diagram,
        robot_model_instances=robot_model_instances,
        configuration_distance_function=_configuration_distance,
        edge_step_size=0.125,
    )
    return collision_checker
