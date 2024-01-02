from cloth_tools.controllers.controller import Controller
from cloth_tools.stations.competition_station import CompetitionStation
from cloth_tools.stations.dual_arm_station import DualArmStation

# @dataclass
# class GraspHighestControllerParams:
#     graph_depth: float = 0.05


class GraspHighestController(Controller):
    def __init__(self, station: DualArmStation):
        self.station = station

    # def plan(self, ):
    # TODO how to check which grasp depths are reachable?

    def execute(self) -> None:
        dual_arm = self.station.dual_arm

        dual_arm.left_manipulator.gripper.open()  # type: ignore
        dual_arm.right_manipulator.gripper.open()  # type: ignore


if __name__ == "__main__":
    station = CompetitionStation()
    grasp_highest_controller = GraspHighestController(station)
    grasp_highest_controller.execute()