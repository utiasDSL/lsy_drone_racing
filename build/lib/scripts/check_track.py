"""Script for checking the positioning of gates and obstacles."""

import logging
from pathlib import Path

import fire

from lsy_drone_racing.ros.ros_utils import check_race_track
from lsy_drone_racing.utils import load_config

logger = logging.getLogger("rosout." + __name__)


def main(config: str = "level2.toml"):
    """Check if the real race track conforms to the race configuration.

    Args:
        config: Path to the race configuration. Assumes the file is in `config/`.
    """
    config = load_config(Path(__file__).resolve().parents[1] / "config" / config)
    check_race_track(config.env.track, config.env.randomizations)
    logger.info("Race track check passed")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    fire.Fire(main)
