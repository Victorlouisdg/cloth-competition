import time

from airo_camera_toolkit.cameras.multiprocess.multiprocess_rgbd_camera import (
    MultiprocessRGBDPublisher,
    MultiprocessRGBDReceiver,
)
from airo_camera_toolkit.cameras.multiprocess.multiprocess_video_recorder import MultiprocessVideoRecorder
from airo_camera_toolkit.cameras.zed.zed2i import Zed2i
from airo_camera_toolkit.utils.image_converter import ImageConverter

# from cloth_tools.stations.competition_station import CompetitionStation

if __name__ == "__main__":
    import multiprocessing

    multiprocessing.set_start_method("spawn")
    # station = CompetitionStation()

    camera_publisher = MultiprocessRGBDPublisher(
        Zed2i,
        camera_kwargs={
            "resolution": Zed2i.RESOLUTION_2K,
            "fps": 15,
            "depth_mode": Zed2i.NEURAL_DEPTH_MODE,
            # "serial_number": "SN30209878",
        },
    )

    camera_publisher.start()
    camera = MultiprocessRGBDReceiver("camera")

    # camera = Zed2i(resolution=Zed2i.RESOLUTION_2K, fps=15, depth_mode=Zed2i.NEURAL_DEPTH_MODE)

    image_rgb = camera.get_rgb_image_as_int()
    point_cloud = camera.get_colored_point_cloud()

    import rerun as rr

    rr.init("test", spawn=True)
    rr.log("world/point_cloud", rr.Points3D(positions=point_cloud.points, colors=point_cloud.colors))

    import cv2

    image = ImageConverter.from_numpy_int_format(image_rgb).image_in_opencv_format

    window_name = "Multiprocess Image"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1280, 720)
    cv2.imshow(window_name, image)
    cv2.waitKey(1)

    recorder = MultiprocessVideoRecorder("camera")
    recorder.start()
    time.sleep(5)
    recorder.stop()

    camera_publisher.stop()
    camera_publisher.join()
