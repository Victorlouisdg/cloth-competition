from cloth_tools.bounding_boxes import BBOX_CLOTH_IN_THE_AIR, BBOX_CLOTH_ON_TABLE
from cloth_tools.controllers.grasp_hanging_controller import GraspHangingController
from cloth_tools.controllers.grasp_highest_controller import GraspHighestController
from cloth_tools.controllers.grasp_lowest_controller import GraspLowestController
from cloth_tools.controllers.home_controller import HomeController
from cloth_tools.controllers.stretch_controller import StretchController
from cloth_tools.motion_blur_detector import MotionBlurDetector
from cloth_tools.stations.competition_station import CompetitionStation

if __name__ == "__main__":
    station = CompetitionStation()

    dual_arm = station.dual_arm

    while True:
        # Move the arms to their home positions
        home_controller = HomeController(station)
        home_controller.execute(interactive=False)

        grasp_highest_controller = GraspHighestController(station, BBOX_CLOTH_ON_TABLE)
        grasp_highest_controller.execute(interactive=True)

        motion_blur_detector = MotionBlurDetector(station.camera, station.hanging_cloth_crop)
        motion_blur_detector.wait_for_blur_to_stabilize()

        grasp_lowest_controller = GraspLowestController(station, BBOX_CLOTH_IN_THE_AIR)
        grasp_lowest_controller.execute(interactive=True)

        motion_blur_detector = MotionBlurDetector(station.camera, station.hanging_cloth_crop)
        motion_blur_detector.wait_for_blur_to_stabilize(timeout=15)

        # Save competition input data here
        grasp_hanging_controller = GraspHangingController(station, BBOX_CLOTH_IN_THE_AIR)
        grasp_hanging_controller.execute(interactive=True)

        stretch_controller = StretchController(station)
        stretch_controller.execute(interactive=True)

        # Save results here
        motion_blur_detector = MotionBlurDetector(station.camera, station.hanging_cloth_crop)
        motion_blur_detector.wait_for_blur_to_stabilize(timeout=15)
