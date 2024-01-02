"""
Detect live whether there is cloth in the air/on the table or none at all.
Works by counting the amount of points in two 3D bounding boxes.
Also shows the highest point in the table bbox and the lowest point in the air bbox.
"""
import cv2
import numpy as np
import rerun as rr
from airo_camera_toolkit.cameras.zed2i import Zed2i
from airo_camera_toolkit.point_clouds.operations import crop_point_cloud
from airo_camera_toolkit.reprojection import project_frame_to_image_plane
from airo_camera_toolkit.utils import ImageConverter
from cloth_tools.bounding_boxes import BBOX_CLOTH_IN_THE_AIR, BBOX_CLOTH_ON_TABLE, bbox_to_mins_and_sizes
from cloth_tools.config import load_camera_pose_in_left_and_right
from cloth_tools.point_clouds.camera import get_image_and_filtered_point_cloud
from cloth_tools.point_clouds.operations import highest_point, lowest_point
from cloth_tools.visualization.rerun import rr_log_camera
from loguru import logger


def log_and_draw_point(image, position, color, point_name, bbox_name) -> None:
    rr_point = rr.Points3D(positions=position, colors=color, radii=0.02)
    rr.log(f"world/{point_name}", rr_point)
    rr.log(f"{bbox_name}/{point_name}", rr_point)

    point_2d = project_frame_to_image_plane(position, camera.intrinsics_matrix(), np.linalg.inv(camera_pose_in_left))
    point_2d = point_2d.squeeze()
    point_2d_int = np.rint(point_2d).astype(int)
    cv2.circle(image, tuple(point_2d_int), 10, bbox_colors[bbox_name], thickness=2)


def process_bboxes(point_cloud_filtered, image_rgb, bboxes, bbox_colors) -> None:
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
            log_and_draw_point(image_rgb, highest, bbox_colors[bbox_name], "highest", bbox_name)

        if bbox_name == "bbox_air" and n_points != 0:
            lowest = lowest_point(point_cloud_cropped.points)
            log_and_draw_point(image_rgb, lowest, bbox_colors[bbox_name], "lowest", bbox_name)

    logger.info(f"Points in bbox_table: {n_points_dict['bbox_table']:7d}, bbox_air: {n_points_dict['bbox_air']:7d}")


def rr_log_bboxes(bboxes, bbox_colors) -> None:
    for bbox_name, bbox in bboxes.items():
        bbox_mins, bbox_sizes = bbox_to_mins_and_sizes(bbox)
        bbox_color = bbox_colors[bbox_name]
        rr_bbox = rr.Boxes3D(mins=bbox_mins, sizes=bbox_sizes, colors=bbox_color)
        rr.log(f"world/{bbox_name}", rr_bbox)
        rr.log(f"{bbox_name}/{bbox_name}", rr_bbox)


if __name__ == "__main__":
    camera = Zed2i(resolution=Zed2i.RESOLUTION_2K, depth_mode=Zed2i.NEURAL_DEPTH_MODE, fps=15)

    window_name = "Cloth BBoxes"

    camera_pose_in_left, camera_pose_in_right = load_camera_pose_in_left_and_right()
    bbox_table = BBOX_CLOTH_ON_TABLE
    bbox_air = BBOX_CLOTH_IN_THE_AIR

    yellow = (255, 231, 122)
    cyan = (122, 173, 255)
    bboxes = {"bbox_table": bbox_table, "bbox_air": bbox_air}
    bbox_colors = {"bbox_table": yellow, "bbox_air": cyan}

    # Setting up rerun
    rr.init(window_name, spawn=True)
    rr_log_camera(camera, camera_pose_in_left)
    rr_log_bboxes(bboxes, bbox_colors)

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    while True:
        image_rgb, point_cloud_filtered = get_image_and_filtered_point_cloud(camera, camera_pose_in_left)

        rr_point_cloud = rr.Points3D(positions=point_cloud_filtered.points, colors=point_cloud_filtered.colors)
        rr.log("world/point_cloud", rr_point_cloud)

        process_bboxes(point_cloud_filtered, image_rgb, bboxes, bbox_colors)

        image_bgr = ImageConverter.from_numpy_int_format(image_rgb).image_in_opencv_format

        cv2.imshow(window_name, image_bgr)
        key = cv2.waitKey(1)
        if key == ord("q"):
            break
