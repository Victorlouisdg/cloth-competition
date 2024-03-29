"""
Detect live whether there is cloth in the air/on the table or none at all.
Works by counting the amount of points in two 3D bounding boxes.
Also shows the highest point in the table bbox and the lowest point in the air bbox.
"""

from typing import Dict, Tuple

import cv2
import numpy as np
import rerun as rr
from airo_camera_toolkit.cameras.multiprocess.multiprocess_stereo_rgbd_camera import (
    MultiprocessStereoRGBDPublisher,
    MultiprocessStereoRGBDReceiver,
)
from airo_camera_toolkit.cameras.zed.zed2i import Zed2i
from airo_camera_toolkit.pinhole_operations.projection import project_points_to_image_plane
from airo_camera_toolkit.point_clouds.operations import crop_point_cloud

# from airo_camera_toolkit.reprojection import project_frame_to_image_plane
from airo_camera_toolkit.utils.image_converter import ImageConverter
from airo_spatial_algebra import transform_points
from airo_typing import BoundingBox3DType, NumpyIntImageType, PointCloud, Vector3DType
from cloth_tools.bounding_boxes import BBOX_CLOTH_IN_THE_AIR, BBOX_CLOTH_ON_TABLE, bbox_to_mins_and_sizes
from cloth_tools.config import load_camera_pose_in_left_and_right
from cloth_tools.point_clouds.camera import get_image_and_filtered_point_cloud
from cloth_tools.point_clouds.operations import highest_point, lowest_point
from cloth_tools.stations.coordinate_frames import create_egocentric_world_frame
from cloth_tools.visualization.rerun import rr_log_camera
from loguru import logger


def log_and_draw_point(
    image: NumpyIntImageType, position: Vector3DType, color: Tuple[int, int, int], point_name: str, bbox_name: str
) -> None:
    rr_point = rr.Points3D(positions=position, colors=color, radii=0.02)
    rr.log(f"world/{point_name}", rr_point)
    rr.log(f"{bbox_name}/{point_name}", rr_point)

    X_C_W = np.linalg.inv(X_W_C)
    point_2d = project_points_to_image_plane(transform_points(X_C_W, position), camera.intrinsics_matrix())
    # point_2d = project_frame_to_image_plane(position, camera.intrinsics_matrix(), np.linalg.inv(X_W_C))
    point_2d = point_2d.squeeze()
    point_2d_int = np.rint(point_2d).astype(int)
    cv2.circle(image, tuple(point_2d_int), 10, bbox_colors[bbox_name], thickness=2)


def process_bboxes(
    point_cloud_filtered: PointCloud,
    image: NumpyIntImageType,
    bboxes: Dict[str, BoundingBox3DType],
    bbox_colors: Dict[str, Tuple[int, int, int]],
) -> None:
    n_points_dict = {}
    for bbox_name, bbox in bboxes.items():
        point_cloud_cropped = crop_point_cloud(point_cloud_filtered, bbox)

        rr_point_cloud_cropped = rr.Points3D(positions=point_cloud_cropped.points, colors=point_cloud_cropped.colors)
        rr.log(f"{bbox_name}/point_cloud", rr_point_cloud_cropped)

        n_points = len(point_cloud_cropped.points)
        n_points_dict[bbox_name] = n_points
        rr.log(f"plot/{bbox_name}", rr.TimeSeriesScalar(n_points, color=bbox_colors[bbox_name]))

        if bbox_name == "bbox_table" and n_points != 0:
            highest = highest_point(point_cloud_cropped.points)
            log_and_draw_point(image, highest, bbox_colors[bbox_name], "highest", bbox_name)

        if bbox_name == "bbox_air" and n_points != 0:
            lowest = lowest_point(point_cloud_cropped.points)
            log_and_draw_point(image, lowest, bbox_colors[bbox_name], "lowest", bbox_name)

    logger.info(f"Points in bbox_table: {n_points_dict['bbox_table']:7d}, bbox_air: {n_points_dict['bbox_air']:7d}")


def rr_log_bboxes(bboxes: Dict[str, BoundingBox3DType], bbox_colors: Dict[str, Tuple[int, int, int]]) -> None:
    for bbox_name, bbox in bboxes.items():
        bbox_mins, bbox_sizes = bbox_to_mins_and_sizes(bbox)
        bbox_color = bbox_colors[bbox_name]
        rr_bbox = rr.Boxes3D(mins=bbox_mins, sizes=bbox_sizes, colors=bbox_color)
        rr.log(f"world/{bbox_name}", rr_bbox)
        rr.log(f"{bbox_name}/{bbox_name}", rr_bbox)


if __name__ == "__main__":

    camera_kwargs = {
        "resolution": Zed2i.RESOLUTION_2K,
        "depth_mode": Zed2i.NEURAL_DEPTH_MODE,
        "fps": 15,
    }
    multiprocess = True

    if multiprocess:

        import multiprocessing

        multiprocessing.set_start_method("spawn")

        camera_publisher = MultiprocessStereoRGBDPublisher(Zed2i, camera_kwargs=camera_kwargs)
        camera_publisher.start()
        camera = MultiprocessStereoRGBDReceiver("camera")
    else:
        camera = Zed2i(**camera_kwargs)

    window_name = "Cloth BBoxes"

    X_LCB_C, X_RCB_C = load_camera_pose_in_left_and_right()
    X_W_C, _, _ = create_egocentric_world_frame(X_LCB_C, X_RCB_C)

    bbox_table = BBOX_CLOTH_ON_TABLE
    bbox_air = BBOX_CLOTH_IN_THE_AIR

    yellow = (255, 231, 122)
    blue = (122, 173, 255)
    bboxes = {"bbox_table": bbox_table, "bbox_air": bbox_air}
    bbox_colors = {"bbox_table": yellow, "bbox_air": blue}

    # Setting up rerun
    rr.init(window_name, spawn=True)
    rr_log_camera(camera, X_W_C)
    rr_log_bboxes(bboxes, bbox_colors)

    rr.log("world/identity_pose", rr.Transform3D(scale=0.5))

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    while True:
        image_rgb, _, point_cloud_filtered = get_image_and_filtered_point_cloud(camera, X_W_C)
        confidence_map = camera._retrieve_confidence_map()
        rr.log("confidence_map", rr.Image(confidence_map))

        rr_point_cloud = rr.Points3D(positions=point_cloud_filtered.points, colors=point_cloud_filtered.colors)
        rr.log("world/point_cloud", rr_point_cloud)

        process_bboxes(point_cloud_filtered, image_rgb, bboxes, bbox_colors)

        image_bgr = ImageConverter.from_numpy_int_format(image_rgb).image_in_opencv_format

        cv2.imshow(window_name, image_bgr)
        key = cv2.waitKey(1)
        if key == ord("q"):
            break

    if multiprocess:
        camera._close_shared_memory()
        camera_publisher.stop()
        camera_publisher.join()
