"""Command type module that translates actions to crazyflie commands."""

from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pycrazyswarm import Crazyflie
    from safe_control_gym.envs.firmware_wrapper import FirmwareWrapper


class Command(Enum):
    """Command types that can be used with pycffirmware."""

    FINISHED = -1  # Args: None (exits the control loop)
    NONE = 0  # Args: None (do nothing)

    FULLSTATE = 1  # Args: [pos, vel, acc, yaw, rpy_rate]
    # https://crazyswarm.readthedocs.io/en/latest/api.html#pycrazyswarm.crazyflie.Crazyflie.cmdFullState
    TAKEOFF = 2  # Args: [height, duration]
    # https://crazyswarm.readthedocs.io/en/latest/api.html#pycrazyswarm.crazyflie.Crazyflie.takeoff
    LAND = 3  # Args: [height, duration]
    # https://crazyswarm.readthedocs.io/en/latest/api.html#pycrazyswarm.crazyflie.Crazyflie.land
    STOP = 4  # Args: None
    # https://crazyswarm.readthedocs.io/en/latest/api.html#pycrazyswarm.crazyflie.Crazyflie.stop
    GOTO = 5  # Args: [pos, yaw, duration, relative (bool)]
    # https://crazyswarm.readthedocs.io/en/latest/api.html#pycrazyswarm.crazyflie.Crazyflie.goTo

    NOTIFYSETPOINTSTOP = 6  # Args: None
    # Must be called to transfer drone state from low level control (cmdFullState) to high level control (takeoff, land, goto)
    # https://crazyswarm.readthedocs.io/en/latest/api.html#pycrazyswarm.crazyflie.Crazyflie.notifySetpointsStop


def apply_sim_command(wrapped_env: FirmwareWrapper, command_type: Command, args: Any):
    """Apply the command to the simulation environment.

    Args:
        wrapped_env: The firmware wrapper.
        command_type: The command type to apply.
        args: Additional arguments as potentially required by `command_type`.
    """
    if command_type == Command.FULLSTATE:
        wrapped_env.sendFullStateCmd(*args)
    elif command_type == Command.TAKEOFF:
        wrapped_env.sendTakeoffCmd(*args)
    elif command_type == Command.LAND:
        wrapped_env.sendLandCmd(*args)
    elif command_type == Command.STOP:
        wrapped_env.sendStopCmd()
    elif command_type == Command.GOTO:
        wrapped_env.sendGotoCmd(*args)
    elif command_type == Command.NOTIFYSETPOINTSTOP:
        wrapped_env.notifySetpointStop()
    elif command_type == Command.NONE:
        pass
    elif command_type == Command.FINISHED:
        pass
    else:
        raise ValueError("[ERROR] Invalid command_type.")


def apply_command(cf: Crazyflie, command_type: Command, args: Any):
    """Apply the command to the drone controller.

    Args:
        cf: The Crazyflie interface class.
        command_type: The command type to apply.
        args: Additional arguments as potentially required by `command_type`.
    """
    if command_type == Command.FULLSTATE:
        # Sim version takes an additional 'ep_time' args that needs to be removed when deployed
        args = args[:5]
        cf.cmdFullState(*args)
    elif command_type == Command.TAKEOFF:
        cf.takeoff(*args)
    elif command_type == Command.LAND:
        cf.land(*args)
    elif command_type == Command.STOP:
        cf.stop()
    elif command_type == Command.GOTO:
        cf.goTo(*args)
    elif command_type == Command.NOTIFYSETPOINTSTOP:
        cf.notifySetpointsStop()
    elif command_type == Command.NONE:
        pass
    elif command_type == Command.FINISHED:
        pass
    else:
        raise ValueError("[ERROR] Invalid command_type.")
