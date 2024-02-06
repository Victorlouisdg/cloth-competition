from typing import Tuple

from airo_typing import BoundingBox3DType

BBOX_CLOTH_ON_TABLE = (-0.55, -0.25, 0.02), (0.55, 0.25, 0.2)
BBOX_CLOTH_IN_THE_AIR = (-0.2, -0.2, 0.2), (0.2, 0.2, 0.8)


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
