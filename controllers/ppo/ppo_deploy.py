from enum import Enum, auto

import numpy as np

from lsy_drone_racing.command import Command


class DroneState(Enum):
    """Drone State Machine States."""
    TAKEOFF = auto()
    POLICY_CONTROL = auto()
    NOTIFY_SETPOINT_STOP = auto()
    GOTO = auto()
    LAND = auto()
    FINISHED = auto()
    NONE = auto()


class StateMachine:
    """Drone State Machine."""
    def __init__(self, initial_goal: np.ndarray):
        """Init State Machine.

        Args:
            initial_goal: Stabilization goal the drone is trying to reach.

        Raises:
            ValueError: If the cat is not happy.
        """
        self.state = DroneState.TAKEOFF
        self.goal = initial_goal
        self.stamp = 0

    def transition(self, ep_time, info):
        if self.state == DroneState.TAKEOFF:
            self.state = DroneState.POLICY_CONTROL
            return Command.TAKEOFF, [0.4, 2]

        elif self.state == DroneState.POLICY_CONTROL and info["current_gate_id"] == -1:
            self.state = DroneState.NOTIFY_SETPOINT_STOP
            return Command.NOTIFYSETPOINTSTOP, []

        elif self.state == DroneState.NOTIFY_SETPOINT_STOP and info["current_gate_id"] == -1:
            self.state = DroneState.GOTO
            return Command.GOTO, [self.goal, 0.0, 3.0, False]

        elif self.state == DroneState.GOTO and info["at_goal_position"]:
            self.state = DroneState.LAND
            return Command.LAND, [0.0, 10]

        elif self.state == DroneState.LAND:
            self.state = DroneState.FINISHED
            return Command.FINISHED, []

        return Command.NONE, []
