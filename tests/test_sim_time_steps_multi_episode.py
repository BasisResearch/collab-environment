import os
from glob import glob

# import pandas as pd
from loguru import logger

# import pyarrow as pa
from collab_env.data.file_utils import get_project_root, expand_path

from tests.sim_test_util import check_parquet_file, remove_run_folder, create_run_folder


def test_sim_multi_episode_output_consistency():
    """
    This will test the following:
    1 - the agents locations and velocity at a time step are consistent with the agent locations
    at the next time step

    2 - the distances reported to the targets and the scene are consistent with the locations reported
    for the agents and the closest points reported for the targets and the scene.

    Returns:

    """
    # make sure these match the config file
    num_episodes = 2
    num_frames = 30
    num_agents = 20
    num_targets = 1

    sim_runs_path = expand_path(
        "sim-output/tests-sim-runs-multi-episode", get_project_root()
    )

    remote_test = "CI" in os.environ
    if not remote_test:
        # Clear the sim-runs folder from previous tests -- a little dicey
        # If this is a remote test, these files shouldn't be here, so don't bother. They
        # get cleaned up at the end
        remove_run_folder(sim_runs_path)
    create_run_folder(sim_runs_path)

    program_path = expand_path(
        "collab_env/sim/boids/run_boids_simulator.py", get_project_root()
    )

    # Test to see that the run_boids_simulator runs successfully with test config file
    config_file_name = "tests/sim_test_time_steps_multi_episode_config.yaml"
    logger.info(f"python {program_path} -cf {config_file_name}")
    result = os.system(f"python {program_path} -cf {config_file_name}")
    assert result == 0

    # Get the output folders in the run path for this test
    folder_list = glob(f"{sim_runs_path}/boids_sim_run_time_steps_multi_episode*")

    check_parquet_file(
        folder_list[0], num_episodes, num_frames, num_agents, num_targets
    )
    logger.debug("check parquet files complete")

    if remote_test:
        remove_run_folder(sim_runs_path)
