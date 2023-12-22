from abc import ABC


class Controller(ABC):
    """A controller is responsible for sending commands to a (dual arm) robot to achieve a certain goal.
    Calling execute() passes full control of the controller.  The controller is responsible for retrieving the information
    it needs from the robot, cameras and other sensors. The controller is allowed to block its thread, open windows for
    visualization, save data to disk, etc.

    Controllers are meant to be composable. For example, an UnfoldingController may have a GraspHighestPointController
    to first lift a piece of cloth and then call a GraspLowestPointController to grasp the lowest point of the cloth in
    the air. These controller can themselves have other controllers as components e.g. a GraspController.
    """

    def execute(self) -> None:
        raise NotImplementedError
