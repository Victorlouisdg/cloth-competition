import time

import cv2
import numpy as np
import scipy.ndimage
from airo_camera_toolkit.image_transforms.image_transform import ImageTransform
from airo_camera_toolkit.interfaces import RGBCamera
from airo_typing import OpenCVIntImageType
from loguru import logger


def calculate_variance_of_laplacian(image: OpenCVIntImageType):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    variance = laplacian.var()
    return variance


class MotionBlurDetector:
    def __init__(
        self,
        camera: RGBCamera,
        image_transform: ImageTransform,
        visualize=True,
        log_to_rerun: bool = True,
        threshold: float = 1.0,  # Ranges from > 200 to almost 0, 1 is almost motionless
    ):
        self.camera = camera
        self.image_transform = image_transform
        self.visualize = visualize
        self.log_to_rerun = log_to_rerun
        self.window_name = "Motion blur detection"

        self.variances = []
        self.variances_smoothed = []
        self.variances_of_variances_smoothed = []
        self.threshold = threshold

    def wait_for_blur_to_stabilize(self, warmup: float = 2.0, timeout: float = 10.0) -> bool:
        """Wait for the motion blur to stabilize.

        Args:
            timeout: the maximum time to wait for the motion blur to stabilize

        Returns:
            True if the motion blur stabilized within the timeout, False otherwise
        """
        time_start = time.time()
        time_last_log = 0.0

        if self.visualize:
            cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)

        fps = self.camera.fps

        history = int(2 * fps)
        sigma = history + 1  # make it odd

        while True:
            image_rgb = self.camera.get_rgb_image_as_int()
            image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
            image_transformed = self.image_transform.transform_image(image_bgr)

            variance = calculate_variance_of_laplacian(image_transformed)
            self.variances.append(variance)
            variance_smoothed = scipy.ndimage.gaussian_filter1d(self.variances, sigma=sigma)[-1]
            self.variances_smoothed.append(variance_smoothed)
            variance_of_variances_smoothed = np.var(self.variances_smoothed[-history:])
            self.variances_of_variances_smoothed.append(variance_of_variances_smoothed)

            if time.time() - time_last_log > 1.0:
                logger.info(
                    f"Variance of the smoothed variance of the Laplacian: {variance_of_variances_smoothed:.2f}"
                )
                time_last_log = time.time()

            if self.log_to_rerun:
                import rerun as rr

                rr.log("laplacian_variance/plot", rr.Scalar(variance_smoothed))
                rr.log("variance_of_variance/plot", rr.Scalar(variance_of_variances_smoothed))

            if variance_of_variances_smoothed < self.threshold and time.time() - time_start > warmup:
                logger.info("Motion blur stabilized")
                cv2.destroyWindow(self.window_name)
                return True

            if time.time() - time_start > timeout:
                logger.warning(
                    f"Motion blur did not stabilize within the timeout. Var: {variance_of_variances_smoothed:.2f}"
                )
                cv2.destroyWindow(self.window_name)
                return False

            if self.visualize:
                cv2.imshow(self.window_name, image_transformed)
                key = cv2.waitKey(1)
                if key == ord("q"):
                    cv2.destroyWindow(self.window_name)
                    return False


# if __name__ == "__main__":
#     camera_kwargs = {
#         "resolution": Zed2i.RESOLUTION_2K,
#         "depth_mode": Zed2i.NEURAL_DEPTH_MODE,
#         "fps": 15,
#     }

#     camera =Zed2i(**camera_kwargs)
#     image_transform = ImageTransform()
#     motion_blur_detector = MotionBlurDetector(camera, image_transform)
#     motion_blur_detector.wait_for_blur_to_stabilize()
