# import pandas as pd

# import pyarrow as pa
from collab_env.data.file_utils import get_project_root, expand_path

from tests.sim_test_util import sim_check_files


def test_sim_multi_episode_output_consistency():
    """
    This will test the following:
    1 - the agents locations and velocity at a time step are consistent with the agent locations
    at the next time step

    2 - the distances reported to the targets and the scene are consistent with the locations reported
    for the agents and the closest points reported for the targets and the scene.

    Returns:

    """

    sim_runs_path = expand_path(
        "sim-output/tests-sim-runs-multi-episode", get_project_root()
    )

    # make sure these match the config file
    sim_check_files(
        sim_runs_path=sim_runs_path,
        config_file="tests/sim_test_time_steps_multi_episode_config.yaml",
        num_episodes=2,
        num_time_steps=30,
        num_agents=20,
        num_targets=1,
        num_log_files=1,
        prefix="boids_sim_run_time_steps_multi_episode",
        video=False,
    )
