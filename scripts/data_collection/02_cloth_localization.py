"""Detect live whether there is cloth in the air/on the table or none at all.
Works by counting the amount of points in two 3D bounding boxes."""
import cv2
import rerun as rr
from airo_camera_toolkit.cameras.zed2i import Zed2i
from airo_camera_toolkit.point_clouds.conversions import open3d_to_point_cloud, point_cloud_to_open3d
from airo_camera_toolkit.point_clouds.operations import crop_point_cloud, filter_point_cloud
from airo_camera_toolkit.utils import ImageConverter
from cloth_tools.bounding_boxes import BBOX_CLOTH_IN_THE_AIR, BBOX_CLOTH_ON_TABLE, bbox_to_mins_and_sizes
from cloth_tools.config import load_camera_pose_in_left_and_right
from loguru import logger

if __name__ == "__main__":
    camera = Zed2i(resolution=Zed2i.RESOLUTION_2K, depth_mode=Zed2i.NEURAL_DEPTH_MODE, fps=15)

    window_name = "Cloth localization"

    camera_pose_in_left, camera_pose_in_right = load_camera_pose_in_left_and_right()
    bbox_table = BBOX_CLOTH_ON_TABLE
    bbox_air = BBOX_CLOTH_IN_THE_AIR

    yellow = (255, 231, 122)
    cyan = (122, 173, 255)
    bboxes = {"bbox_table": bbox_table, "bbox_air": bbox_air}
    bbox_colors = {"bbox_table": yellow, "bbox_air": cyan}

    # Setting up rerun
    rr.init(window_name, spawn=True)
    rr.log("world/camera", rr.Pinhole(image_from_camera=camera.intrinsics_matrix(), resolution=camera.resolution))
    rr.log(
        "world/camera", rr.Transform3D(translation=camera_pose_in_left[0:3, 3], mat3x3=camera_pose_in_left[0:3, 0:3])
    )

    for bbox_name, bbox in bboxes.items():
        bbox_mins, bbox_sizes = bbox_to_mins_and_sizes(bbox)
        bbox_color = bbox_colors[bbox_name]
        rr.log(f"world/{bbox_name}", rr.Boxes3D(mins=bbox_mins, sizes=bbox_sizes, colors=bbox_color))
        rr.log(f"{bbox_name}/{bbox_name}", rr.Boxes3D(mins=bbox_mins, sizes=bbox_sizes, colors=bbox_color))

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    while True:
        image_rgb = camera.get_rgb_image_as_int()
        point_cloud_in_camera = camera._retrieve_colored_point_cloud()
        confidence_map = camera._retrieve_confidence_map()

        # Transform the point cloud to the world frame
        pcd_in_camera = point_cloud_to_open3d(point_cloud_in_camera)
        pcd = pcd_in_camera.transform(camera_pose_in_left)  # transform to world frame (= base frame of left robot)

        # Filter the low confidence points
        confidence_mask = (confidence_map <= 1.0).reshape(-1)  # Threshold and flatten

        point_cloud = open3d_to_point_cloud(pcd)
        point_cloud_filtered = filter_point_cloud(point_cloud, confidence_mask)

        rr.log(
            "world/point_cloud", rr.Points3D(positions=point_cloud_filtered.points, colors=point_cloud_filtered.colors)
        )

        n_points_dict = {}
        for bbox_name, bbox in bboxes.items():
            bbox_mins, bbox_sizes = bbox_to_mins_and_sizes(bbox)
            point_cloud_cropped = crop_point_cloud(point_cloud_filtered, bbox)
            rr.log(
                f"{bbox_name}/point_cloud",
                rr.Points3D(positions=point_cloud_cropped.points, colors=point_cloud_cropped.colors),
            )

            n_points = len(point_cloud_cropped.points)
            rr.log(f"plot/{bbox_name}", rr.TimeSeriesScalar(n_points, color=bbox_colors[bbox_name]))

            n_points_dict[bbox_name] = n_points

        logger.info(
            f"Points in bbox_table: {n_points_dict['bbox_table']:7d}, bbox_air: {n_points_dict['bbox_air']:7d}"
        )

        image_bgr = ImageConverter.from_numpy_int_format(image_rgb).image_in_opencv_format

        cv2.imshow(window_name, image_bgr)
        key = cv2.waitKey(1)
        if key == ord("q"):
            break
