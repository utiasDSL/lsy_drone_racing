from functools import partial
from pathlib import Path

import pytest
from safe_control_gym.utils.registration import make

from lsy_drone_racing.constants import FIRMWARE_FREQ
from lsy_drone_racing.utils import load_config
from lsy_drone_racing.wrapper import DroneRacingObservationWrapper, DroneRacingWrapper


@pytest.fixture(scope="session")
def base_env():
    """Create the drone racing environment."""
    # Load configuration and check if firmare should be used.
    config_path = Path(__file__).resolve().parents[1] / "config/getting_started.yaml"
    config = load_config(config_path)
    # Overwrite config options
    config.quadrotor_config.gui = False
    CTRL_FREQ = config.quadrotor_config["ctrl_freq"]
    # Create environment
    assert config.use_firmware, "Firmware must be used for the competition."
    pyb_freq = config.quadrotor_config["pyb_freq"]
    assert pyb_freq % FIRMWARE_FREQ == 0, "pyb_freq must be a multiple of firmware freq"
    config.quadrotor_config["ctrl_freq"] = FIRMWARE_FREQ
    env_factory = partial(make, "quadrotor", **config.quadrotor_config)
    yield make("firmware", env_factory, FIRMWARE_FREQ, CTRL_FREQ)


@pytest.mark.parametrize("terminate_on_lap", [True, False])
def test_drone_racing_wrapper(base_env, terminate_on_lap: bool):
    """Test the DroneRacingWrapper."""
    env = DroneRacingWrapper(base_env, terminate_on_lap=terminate_on_lap)


def test_drone_racing_obs_wrapper(base_env):
    """Test the DroneRacingObservationWrapper."""
    env = DroneRacingObservationWrapper(base_env)
