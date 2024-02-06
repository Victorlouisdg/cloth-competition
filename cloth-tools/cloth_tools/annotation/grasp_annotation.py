from typing import Tuple

import cv2
import numpy as np
from airo_camera_toolkit.pinhole_operations.projection import project_points_to_image_plane
from airo_camera_toolkit.pinhole_operations.unprojection import unproject_onto_depth_values
from airo_camera_toolkit.point_clouds.conversions import point_cloud_to_open3d
from airo_camera_toolkit.utils.image_converter import ImageConverter
from airo_spatial_algebra import transform_points
from airo_typing import (
    CameraIntrinsicsMatrixType,
    HomogeneousMatrixType,
    NumpyDepthMapType,
    NumpyIntImageType,
    OpenCVIntImageType,
    PointCloud,
    Vector3DType,
)
from cloth_tools.visualization.opencv import draw_pose
from loguru import logger


def top_down_camera_pose(height: float = 1.5) -> np.ndarray:
    Z = np.array([0, 0, -1])
    X = np.array([0, -1, 0])  # X is the rightwards direction for the camera
    Y = np.cross(Z, X)
    orientation = np.column_stack([X, Y, Z])
    position = np.array([0, 0, height])

    pose = np.identity(4)
    pose[:3, :3] = orientation
    pose[:3, 3] = position

    return pose


def grasp_hanging_cloth_pose(position: Vector3DType, approach_direction: Vector3DType) -> HomogeneousMatrixType:
    """Create a pose for grasping a hanging cloth.

    Args:
        position: The position of the grasp.
        approach_direction: The direction the gripper should approach the grasp from.

    Returns:
        pose: The pose for the grasp.
    """
    Z = approach_direction

    # Pointing gripper Y up or down leads to the gripper opening horizontally
    # I chose Y-down here because it's closer the other poses used in the controllers
    Y = np.array([0, 0, -1])
    X = np.cross(Y, Z)
    orientation = np.column_stack([X, Y, Z])
    orientation = np.column_stack([X, Y, Z])

    pose = np.identity(4)
    pose[:3, :3] = orientation
    pose[:3, 3] = position

    return pose


def project_point_cloud_to_image(
    point_cloud: PointCloud,
    camera_pose: HomogeneousMatrixType,
    intrinsics: np.ndarray,
    resolution: Tuple[int, int],
    background_color: Tuple[int, int, int] = (0, 0, 0),
    blur_image: bool = False,
) -> OpenCVIntImageType:
    pcd = point_cloud_to_open3d(point_cloud)
    rgbd_image = pcd.project_to_rgbd_image(*resolution, intrinsics, extrinsics=np.linalg.inv(camera_pose))

    image_rgb_float = np.asarray(rgbd_image.color)
    depth_map_float = np.asarray(rgbd_image.depth).squeeze()

    # Convert to uint8
    image = ImageConverter.from_numpy_format(image_rgb_float).image_in_opencv_format

    # Turn pixels with zero depth to gray
    image[depth_map_float == 0.0] = background_color

    if blur_image:
        image = cv2.blur(image, (3, 3))
        image = cv2.medianBlur(image, 3)

    return image


def calculate_approach_direction_from_clicked_points(
    frontal_clicked_in_world: Vector3DType,
    topdown_clicked_in_image: Tuple[int, int],
    camera_topdown_pose: HomogeneousMatrixType,
    intrinsics: CameraIntrinsicsMatrixType,
):
    p_W = frontal_clicked_in_world
    X_W_VC = camera_topdown_pose
    topdown_clicked_in_image = np.array(topdown_clicked_in_image)

    X_VC_W = np.linalg.inv(X_W_VC)
    frontal_clicked_in_image = project_points_to_image_plane(transform_points(X_VC_W, p_W), intrinsics).squeeze()

    # Image space vector from topdown to frontal clicked points
    vector_in_image = frontal_clicked_in_image - topdown_clicked_in_image

    # Convert to camera space 3D vector
    vector_in_camera = np.array([vector_in_image[0], vector_in_image[1], 0])  # Image space to camera space vector
    vector_in_camera = vector_in_camera / np.linalg.norm(vector_in_camera)  # Normalize to unit vector

    # Transform to world space vector
    v_VC = vector_in_camera
    R_W_VC = X_W_VC[:3, :3]
    v_W = R_W_VC @ v_VC

    approach_direction = v_W
    return approach_direction


def calculate_grasp_pose_from_clicked_points(
    frontal_clicked_in_image: Tuple[int, int],
    topdown_clicked_in_image: Tuple[int, int] | None,
    depth_map: NumpyDepthMapType,
    camera_pose: HomogeneousMatrixType,
    camera_topdown_pose: HomogeneousMatrixType,
    intrinsics: CameraIntrinsicsMatrixType,
) -> HomogeneousMatrixType:
    X_W_C = camera_pose
    X_W_VC = camera_topdown_pose

    # Transforming the point clicked in the frontal view to the world frame
    u, v = frontal_clicked_in_image
    image_coordinates = np.array([u, v]).reshape((1, 2))
    depth_values = np.array([depth_map[v, u]])
    p_C = unproject_onto_depth_values(image_coordinates, depth_values, intrinsics)
    p_W = transform_points(X_W_C, p_C).squeeze()

    # Calculate the approach direction
    Z = np.array([1, 0, 0])

    if topdown_clicked_in_image is not None:
        Z = calculate_approach_direction_from_clicked_points(p_W, topdown_clicked_in_image, X_W_VC, intrinsics)

    grasp_pose = grasp_hanging_cloth_pose(p_W, Z)
    return grasp_pose


def get_manual_grasp_annotation(  # noqa: C901
    image: NumpyIntImageType,
    depth_map: NumpyDepthMapType,
    point_cloud: PointCloud,
    camera_pose: HomogeneousMatrixType,
    intrinsics: CameraIntrinsicsMatrixType,
    log_to_rerun: bool = False,
) -> HomogeneousMatrixType | None:
    """Manually annotate a grasp pose on a hanging piece of cloth.

    Args:
        image: A (frontal) image.
        depth_map: The corresponding depth map.
        point_cloud: The (filtered) point cloud in the world frame.
        camera_pose: The pose of the camera in the world frame.
        intrinsics: The camera intrinsics.
    """
    cyan = (255, 255, 0)

    window_frontal = "Grasp Annotation - Frontal view"
    window_topdown = "Grasp Annotation - Topdown view"

    logger.info("[Grasp Annotation] - Usage:")
    logger.info("Click in the front view to set grasp location.")
    logger.info("Click in the topdown view to set the approach direction.")
    logger.info("Press 'y' to confirm the grasp pose.")
    logger.info("Press 'b' to toggle blur in the topdown view.")
    logger.info("Press 'q' to quit.")

    h = 600
    w = int(1.5 * h)
    cv2.namedWindow(window_frontal, cv2.WINDOW_NORMAL)
    cv2.namedWindow(window_topdown, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_frontal, w, h)
    cv2.resizeWindow(window_topdown, w, h)
    cv2.moveWindow(window_frontal, 0, 0)
    cv2.moveWindow(window_topdown, w + 80, 0)

    clicked_point = {window_frontal: None, window_topdown: None}

    def callback_frontal(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            logger.info(f"Frontal view: click at ({x}, {y}), depth = {depth_map[y, x]:.3f} meter")
            clicked_point[window_frontal] = (x, y)

        if event == cv2.EVENT_RBUTTONDOWN:
            logger.info("Frontal view: removed click.")
            clicked_point[window_frontal] = None

    def callback_topdown(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            logger.info(f"Topdown view: click at ({x}, {y})")
            clicked_point[window_topdown] = (x, y)
        if event == cv2.EVENT_RBUTTONDOWN:
            logger.info("Topdown view: removed click.")
            clicked_point[window_topdown] = None

    cv2.setMouseCallback(window_frontal, callback_frontal)
    cv2.setMouseCallback(window_topdown, callback_topdown)

    if log_to_rerun:
        import rerun as rr

        rr.init("Grasp Annotation - Point cloud", spawn=True)
        rr.log("world/point_cloud", rr.Points3D(positions=point_cloud.points, colors=point_cloud.colors))
        rr.log("world/identity", rr.Transform3D(scale=0.5))

    X_W_C = camera_pose
    X_W_VC = top_down_camera_pose(height=1.5)

    resolution = (image.shape[1], image.shape[0])

    image_frontal = ImageConverter.from_numpy_int_format(image).image_in_opencv_format
    image_topdown_sharp = project_point_cloud_to_image(
        point_cloud, X_W_VC, intrinsics, resolution, background_color=(90, 90, 90)
    )
    image_topdown_blurred = project_point_cloud_to_image(
        point_cloud, X_W_VC, intrinsics, resolution, background_color=(90, 90, 90), blur_image=True
    )

    image_topdown = image_topdown_sharp
    blur_image = False
    grasp_pose = None

    while True:
        image_frontal_annotated = image_frontal.copy()
        image_topdown_annotated = image_topdown.copy()

        if clicked_point[window_frontal] is not None:
            grasp_pose = calculate_grasp_pose_from_clicked_points(
                clicked_point[window_frontal], clicked_point[window_topdown], depth_map, X_W_C, X_W_VC, intrinsics
            )

            draw_pose(image_frontal_annotated, grasp_pose, intrinsics, X_W_C)
            draw_pose(image_topdown_annotated, grasp_pose, intrinsics, X_W_VC)

            if clicked_point[window_topdown] is not None:
                cv2.circle(image_topdown_annotated, clicked_point[window_topdown], 5, cyan, 3, cv2.LINE_AA)

            if log_to_rerun:
                rr.log("world/grasp_pose", rr.Transform3D(translation=grasp_pose[:3, 3], mat3x3=grasp_pose[:3, :3]))

        cv2.imshow(window_frontal, image_frontal_annotated)
        cv2.imshow(window_topdown, image_topdown_annotated)
        key = cv2.waitKey(10)
        if key == ord("q"):
            cv2.destroyAllWindows()
            return None
        if key == ord("y"):
            cv2.destroyAllWindows()
            return grasp_pose
        if key == ord("b"):
            blur_image = not blur_image
            image_topdown = image_topdown_blurred if blur_image else image_topdown_sharp
