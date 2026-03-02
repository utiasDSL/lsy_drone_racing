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


def main(config: str = "multi_level0.toml"):
    """Deploy and run the race host.

    Args:
        config: Path to the competition configuration. Assumes the file is in `config/`.
    """
    # Setup logging
    logging.basicConfig(level=logging.WARNING)
    logging.getLogger("jax").setLevel(logging.ERROR)
    logger.setLevel(logging.INFO)
    logging.getLogger("lsy_drone_racing").setLevel(logging.INFO)
    
    # Initialize ROS2
    rclpy.init()
    
    try:
        # Load configuration
        config_obj = load_config(Path(__file__).parents[1] / "config" / config)
        
        # Create and run host
        host = CrazyFlieRealRaceHost(config_obj)
        logger.info("Host created, starting main loop...")
        host.host_main_loop()
        
    except KeyboardInterrupt:
        logger.info("Host interrupted by user")
    except Exception as e:
        logger.error(f"Host encountered an error: {e}", exc_info=True)
    finally:
        try:
            host._cleanup()
        except Exception:
            logger.exception("Error during host cleanup")
        rclpy.shutdown()
        logger.info("Host shutdown complete")


if __name__ == "__main__":
    fire.Fire(main)
