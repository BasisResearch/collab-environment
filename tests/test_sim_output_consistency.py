# import pandas as pd

# import pyarrow as pa
from collab_env.data.file_utils import get_project_root, expand_path

from tests.sim_test_util import sim_check_files


def test_sim_output_consistency():
    """
    This will test the following:
    1 - the agents locations and velocity at a time step are consistent with the agent locations
    at the next time step

    2 - the distances reported to the targets and the scene are consistent with the locations reported
    for the agents and the closest points reported for the targets and the scene.

    Returns:

    """

    sim_runs_path = expand_path(
        "sim-output/tests-sim-runs-consistency", get_project_root()
    )
    sim_check_files(
        sim_runs_path=sim_runs_path,
        prefix="boids_sim_run_consistent",
        config_file="tests/sim_test_consistent_config.yaml",
        num_episodes=1,
        video=False,
        num_log_files=1,
        num_time_steps=20,
        num_agents=20,
        num_targets=1,
    )
