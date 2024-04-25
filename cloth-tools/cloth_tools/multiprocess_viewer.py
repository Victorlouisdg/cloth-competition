import multiprocessing
import os
import time
from multiprocessing import Process
from typing import Optional

import cv2
from airo_camera_toolkit.cameras.multiprocess.multiprocess_rgb_camera import MultiprocessRGBReceiver
from airo_camera_toolkit.image_transforms.image_transform import ImageTransform
from cloth_tools.dataset.bookkeeping import datetime_for_filename
from loguru import logger


class MultiprocessViewer(Process):
    def __init__(
        self,
        shared_memory_namespace: str,
        image_transform: Optional[ImageTransform] = None,
        save_images_dir: Optional[str] = None,
    ):
        super().__init__(daemon=True)
        self._shared_memory_namespace = shared_memory_namespace
        self._image_transform = image_transform
        self.started_event = multiprocessing.Event()
        self.finished_event = multiprocessing.Event()
        self.shutdown_event = multiprocessing.Event()

        self.save_images_dir = save_images_dir
        if self.save_images_dir is not None:
            self.save_images_dir = str(self.save_images_dir)
            os.makedirs(self.save_images_dir, exist_ok=True)

    def start(self) -> None:
        super().start()
        # Block until the recording has started
        self.started_event.wait()

    def run(self) -> None:
        """main loop of the process, runs until the process is terminated"""
        receiver = MultiprocessRGBReceiver(self._shared_memory_namespace)
        logger.info("Viewer started.")

        window_name = f"MultiprocessViewer: {self._shared_memory_namespace}"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.moveWindow(window_name, 0, 0)
        cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

        # cv2.resizeWindow(window_name, 1280, 720)

        self.started_event.set()

        while not self.shutdown_event.is_set():
            image_rgb = receiver.get_rgb_image_as_int()
            image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

            # TODO add options to save jpgs to a directory? -> if video recording would fail
            if self.save_images_dir is not None:
                image_filename = os.path.join(self.save_images_dir, f"frame_{datetime_for_filename()}.jpg")
                cv2.imwrite(image_filename, image_bgr)

            cv2.imshow(window_name, image_bgr)
            key = cv2.waitKey(1)
            if key == ord("q"):
                break

        receiver._close_shared_memory()
        self.finished_event.set()

    def stop(self) -> None:
        self.shutdown_event.set()
        self.finished_event.wait(timeout=2)


if __name__ == "__main__":
    from pathlib import Path

    save_images_dir = Path("output") / f"frames_{datetime_for_filename()}"
    save_images_dir = None

    viewer = MultiprocessViewer("camera", save_images_dir=save_images_dir)

    viewer.start()
    time.sleep(20)
    viewer.stop()
