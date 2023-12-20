from typing import Tuple

from airo_typing import BoundingBox3DType


def bounding_box_between_robots(
    distance_between_robots: float = 0.9,
    width: float = 0.5,
    length: float = 0.5,
    height_bottom: float = 0.0,
    height_top: float = 0.5,
) -> BoundingBox3DType:
    """Bounding box between two robots.
    Assumes the two robots are mounted on the x-axis with one at the origin and one in the positive x-direction.

    Args:
        distance_between_robots: distance between the two robot's bases in meters
        width: size of the bbox in the x-direction
        length: size of the bbox in the y-direction
        height_bottom: height of the bottom of the bbox relative to the robot's base
        height_top: height of the top of the bbox relative to the robot's base

    Returns:
        the bounding box with the requested dimensions
    """
    x_middle = distance_between_robots / 2
    x_min = x_middle - width / 2
    x_max = x_middle + width / 2
    y_min = -length / 2
    y_max = length / 2
    z_min = height_bottom
    z_max = height_top
    return ((x_min, y_min, z_min), (x_max, y_max, z_max))


BBOX_CLOTH_ON_TABLE = bounding_box_between_robots(0.9, 0.6, 1.1, 0.02, 0.2)
BBOX_CLOTH_IN_THE_AIR = bounding_box_between_robots(0.9, 0.4, 0.4, 0.20, 0.95)


def bbox_to_mins_and_sizes(
    bounding_box: BoundingBox3DType,
) -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
    """Convert a bounding box specified as the mix and max corners to the alternative mins and sizes format.
    This is the format used by Rerun.

    Args:
        bounding_box: bounding box specified as the mix and max corners

    Returns:
        bounding box specified as a tuple: the mins and sizes
    """
    mins, maxs = bounding_box
    sizes = (maxs[0] - mins[0], maxs[1] - mins[1], maxs[2] - mins[2])
    return (mins, sizes)
