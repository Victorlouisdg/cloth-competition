import time

from airo_camera_toolkit.cameras.multiprocess.multiprocess_video_recorder import MultiprocessVideoRecorder
from cloth_tools.stations.competition_station import CompetitionStation

if __name__ == "__main__":
    station = CompetitionStation()

    recorder = MultiprocessVideoRecorder("camera")
    recorder.start()
    time.sleep(10)
    recorder.stop()
