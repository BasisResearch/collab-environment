import os
from loguru import logger

from collab_env.data.file_utils import expand_path
from tests.sim_test_util import sim_check_files


def test_sim_1_with_visualizer():
    """
    Test everything that sim_test_1 tests but with the visualizer.
    This should not be run in remote environment, since the visualizer window
    will likely not be able to be created in the remote setup.

    Returns:

    """
    logger.info("called")
    remote_test = "CI" in os.environ
    print(f"remote test is {remote_test}")
    if not remote_test:
        sim_check_files(
            config_file="tests/sim_test_1_wth_vis_config.yaml",
            sim_runs_path=expand_path("sim-output/tests-sim-runs-1-visualizer"),
        )
