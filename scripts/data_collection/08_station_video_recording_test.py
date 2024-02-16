import time

from airo_camera_toolkit.cameras.multiprocess.multiprocess_video_recorder import MultiprocessVideoRecorder
from cloth_tools.stations.competition_station import CompetitionStation

if __name__ == "__main__":
    station = CompetitionStation()

    recorder = MultiprocessVideoRecorder("camera", log_fps=True)
    recorder.start()
    time.sleep(10)
    recorder.stop()
    time.sleep(5)  # Give video recorder time to shut down
