"""Constants for the drone racing project."""

from dataclasses import dataclass

FIRMWARE_FREQ = 500
Z_LOW = 0.3
Z_HIGH = 0.775
CTRL_TIMESTEP = 1 / 30
CTRL_FREQ = 30
QUADROTOR_KF = 3.16e-10
QUADROTOR_KM = 7.94e-12
SENSOR_RANGE = 0.45


@dataclass
class GateDesc:
    """Gate description."""

    height: float
    shape: str = "square"
    edge: float = 0.45


@dataclass
class LowGateDesc(GateDesc):
    """Low gate description."""

    height: float = 0.525


@dataclass
class HighGateDesc(GateDesc):
    """High gate description."""

    height: float = 1.0


@dataclass
class QuadrotorPhysicParams:
    """Physical parameters of the quadrotor."""

    quadrotor_mass: float = 0.03454
    quadrotor_ixx_inertia: float = 1.4e-05
    quadrotor_iyy_inertia: float = 1.4e-05
    quadrotor_izz_inertia: float = 2.17e-05


@dataclass
class ObstacleDesc:
    """Obstacle description."""

    shape: str = "cylinder"
    height: float = 1.05
    radius: float = 0.05
