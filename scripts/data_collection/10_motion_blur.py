import cv2
import numpy as np
import rerun as rr
import scipy
from cloth_tools.stations.competition_station import CompetitionStation
from loguru import logger


def detect_blur_fft(image, size=60, thresh=10, vis=False):
    # grab the dimensions of the image and use the dimensions to
    # derive the center (x, y)-coordinates
    (h, w) = image.shape
    (cX, cY) = (int(w / 2.0), int(h / 2.0))

    # compute the FFT to find the frequency transform, then shift
    # the zero frequency component (i.e., DC component located at
    # the top-left corner) to the center where it will be more
    # easy to analyze
    fft = np.fft.fft2(image)
    fftShift = np.fft.fftshift(fft)

    # compute the magnitude spectrum of the transform
    # magnitude = 20 * np.log(np.abs(fftShift))

    # zero-out the center of the FFT shift (i.e., remove low
    # frequencies), apply the inverse shift such that the DC
    # component once again becomes the top-left, and then apply
    # the inverse FFT
    fftShift[cY - size : cY + size, cX - size : cX + size] = 0
    fftShift = np.fft.ifftshift(fftShift)
    recon = np.fft.ifft2(fftShift)

    magnitude = 20 * np.log(np.abs(recon))
    mean = np.mean(magnitude)
    # the image will be considered "blurry" if the mean value of the
    # magnitudes is less than the threshold value
    return (mean, mean <= thresh)


if __name__ == "__main__":
    station = CompetitionStation()
    camera = station.camera

    window_name = "Motion blur detection"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1280, 720)

    rr.init(window_name, spawn=True)

    variances = []
    means = []

    # Detect motion blur per image and log
    while True:
        image_rgb = camera.get_rgb_image_as_int()
        image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

        # Calculate the variance of the Laplacian of the image
        variance = cv2.Laplacian(image_bgr, cv2.CV_64F).var()

        mean, blurry = detect_blur_fft(cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY))

        variances.append(variance)
        means.append(mean)

        logger.info(f"Blurry: {blurry} - FFT blur: {mean:.2f} - Variance of the Laplacian: {variance:.2f}")

        variance_smoothed = scipy.ndimage.gaussian_filter1d(variances, sigma=7)[-1]
        mean_smoothed = scipy.ndimage.gaussian_filter1d(means, sigma=5)[-1]

        logger.info(f"Variance of the Laplacian smoothed: {variance_smoothed:.2f}")
        logger.info(f"FFT blur smoothed: {mean_smoothed:.2f}")

        # rr.log
        rr.log("laplacian_variance/plot", rr.Scalar(variance))
        rr.log("laplacian_variance/plot_smoothed", rr.Scalar(variance_smoothed))
        rr.log("fft_blur/plot", rr.Scalar(mean))
        rr.log("fft_blur/plot_smoothed", rr.Scalar(mean_smoothed))

        cv2.imshow(window_name, image_bgr)
        key = cv2.waitKey(1)
        if key == ord("q"):
            break
