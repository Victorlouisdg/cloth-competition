import time

from airo_camera_toolkit.cameras.multiprocess.multiprocess_rgbd_camera import (
    MultiprocessRGBDPublisher,
    MultiprocessRGBDReceiver,
)
from airo_camera_toolkit.cameras.multiprocess.multiprocess_video_recorder import MultiprocessVideoRecorder
from airo_camera_toolkit.cameras.zed.zed2i import Zed2i
from airo_camera_toolkit.utils.image_converter import ImageConverter

# from cloth_tools.stations.competition_station import CompetitionStation
# from loguru import logger

# station = CompetitionStation()

camera_publisher = MultiprocessRGBDPublisher(
    Zed2i,
    camera_kwargs={
        "resolution": Zed2i.RESOLUTION_2K,
        "fps": 15,
        "depth_mode": Zed2i.NEURAL_DEPTH_MODE,
    },
)

camera_publisher.start()


camera = MultiprocessRGBDReceiver("camera")
image_rgb = camera.get_rgb_image_as_int()

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
