from airo_camera_toolkit.point_clouds.operations import crop_point_cloud
from cloth_tools.point_clouds.camera import get_image_and_filtered_point_cloud


def sufficient_points_in_bbox(station, bbox, minimum_amount_of_points=1000):
    camera = station.camera
    camera_pose = station.camera_pose

    _, _, point_cloud = get_image_and_filtered_point_cloud(camera, camera_pose)
    point_cloud_cropped = crop_point_cloud(point_cloud, bbox)

    return len(point_cloud_cropped.points) > minimum_amount_of_points
