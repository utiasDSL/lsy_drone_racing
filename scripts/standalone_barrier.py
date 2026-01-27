"""Standalone takeoff barrier script.

This script sets up and manages a shared start barrier for coordinated multi-drone takeoff.
All drone processes should connect to this barrier before starting their episodes.

Usage:

python standalone_barrier.py --config multi_level0.toml
python standalone_barrier.py --config multi_level0.toml --num_drones 3

Configuration is read from the TOML config file's [deploy.takeoff_barrier] section.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

# Add parent directory to path to import from lsy_drone_racing
sys.path.insert(0, str(Path(__file__).parents[1]))

from lsy_drone_racing.utils import load_config
from lsy_drone_racing.utils.takeoff_barrier import TakeOffBarrier

logger = logging.getLogger(__name__)


def main(config: str = "multi_level0.toml", num_drones: int | None = None, verbose: bool = False):
    """Start a standalone takeoff barrier server.

    Args:
        config: Path to the competition configuration. Assumes the file is in `config/`.
        num_drones: Number of drones expected to synchronize on this barrier. If None, inferred
            from the config file.
        verbose: Enable verbose logging (default: False).
    """
    if verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)
    logger.setLevel(logging.INFO)

    # Load configuration
    config_obj = load_config(Path(__file__).parents[1] / "config" / config)

    # Determine number of drones
    if num_drones is None:
        if hasattr(config_obj, "deploy") and hasattr(config_obj.deploy, "drones"):
            num_drones = len(config_obj.deploy.drones)
        else:
            raise ValueError(
                "num_drones must be specified or config must have deploy.drones section"
            )

    logger.info("Starting takeoff barrier for %d drones", num_drones)

    # Extract barrier configuration from config file
    barrier_cfg = config_obj.deploy.takeoff_barrier
    barrier_config = {
        "authkey": barrier_cfg["authkey"].encode()
        if isinstance(barrier_cfg["authkey"], str)
        else barrier_cfg["authkey"],
        "timeout_s": barrier_cfg["timeout_s"],
        "filename": barrier_cfg["filename"],
        "port": barrier_cfg["port"],
        "bind_host": barrier_cfg.get("bind_host", "0.0.0.0"),
        "public_host": barrier_cfg.get("public_host"),
    }

    # Create barrier instance
    barrier = TakeOffBarrier(**barrier_config)

    try:
        # Create the barrier manager without participating
        logger.info("Setting up barrier manager for %d drone(s)...", num_drones)
        barrier.wait(rank=0, parties=num_drones, participate=False)
        logger.info("Barrier manager is running and ready for %d drones to connect.", num_drones)
        logger.info("Press Ctrl+C to shutdown after all drones have passed.")
        
        # Keep the manager alive indefinitely
        import signal
        import threading
        
        shutdown_event = threading.Event()
        
        def signal_handler(sig, frame):
            logger.info("Shutdown signal received. Closing barrier manager...")
            shutdown_event.set()
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        
        shutdown_event.wait()
    except Exception as exc:
        logger.error("Barrier encountered an error: %s", exc)
        sys.exit(1)
    finally:
        barrier.close()


if __name__ == "__main__":
    import fire

    fire.Fire(main)
