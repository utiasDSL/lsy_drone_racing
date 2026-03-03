"""Deployment script for the multi-drone racing host.

The host is responsible for:
- Track validation and drone connection
- Coordinating client synchronization via Zenoh
- Supervising the race and handling completion

Usage:

python deploy_host.py --config multi_level0.toml

"""

from __future__ import annotations

import logging
from pathlib import Path

import fire
import rclpy

from lsy_drone_racing.envs.real_race_host import CrazyFlieRealRaceHost
from lsy_drone_racing.utils import load_config

logger = logging.getLogger(__name__)


def main(config: str = "multi_level2.toml"):
    """Deploy and run the race host.

    Args:
        config: Path to the competition configuration. Assumes the file is in `config/`.
    """
    # Setup logging
    logging.basicConfig(level=logging.DEBUG)
    logging.getLogger("jax").setLevel(logging.ERROR)
    logging.getLogger("cflib").setLevel(logging.WARNING)
    logger.setLevel(logging.INFO)
    logging.getLogger("lsy_drone_racing").setLevel(logging.DEBUG)
    
    # Initialize ROS2
    rclpy.init()
    
    host = None
    try:
        # Load configuration
        config_obj = load_config(Path(__file__).parents[1] / "config" / config)
        # Create host
        host = CrazyFlieRealRaceHost(config_obj)
        # Connect to drones
        logger.info("Host created, connecting to drones...")
        host.connect_drones()
        # Start main loop
        logger.info("Drones connected, starting main loop...")
        host.host_main_loop()
    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt, shutting down...")
    except Exception as e:
        logger.error(f"Host encountered an error: {e}", exc_info=True)
    finally:
        if host:
            host.close()
        rclpy.shutdown()
        logger.info("Host shutdown complete")


if __name__ == "__main__":
    fire.Fire(main)
