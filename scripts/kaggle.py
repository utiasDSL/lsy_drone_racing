"""Kaggle competition auto-submission script.

Note:
    Please do not alter this script or ask the course supervisors first!
"""

import logging
from pathlib import Path

import pandas as pd
from sim import simulate

from lsy_drone_racing.utils import load_config

logger = logging.getLogger(__name__)


def main():
    """Run the simulation N times and save the results as 'submission.csv'."""
    n_runs = 10
    config_file = "level2.toml"
    config = load_config(Path(__file__).parents[1] / "config" / config_file)
    ep_times = simulate(
        config=config_file, controller=config.controller.file, n_runs=n_runs, gui=False
    )

    # Log the number of failed runs if any
    if n_failed := len([x for x in ep_times if x is None]):
        logger.warning(f"{n_failed} run{'' if n_failed == 1 else 's'} failed out of {n_runs}!")
    else:
        logger.info("All runs completed successfully!")

    # Abort if more than half of the runs failed
    if n_failed > n_runs / 2:
        logger.error("More than 50% of all runs failed! Aborting submission.")
        raise RuntimeError("Too many runs failed!")

    ep_times = [x for x in ep_times if x is not None]
    data = {"ID": [i for i in range(len(ep_times))], "submission_time": ep_times}
    pd.DataFrame(data).to_csv("submission.csv", index=False)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
