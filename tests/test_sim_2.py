import os
from glob import glob

# import pandas as pd
import pyarrow.parquet as pq

# import pyarrow as pa
from collab_env.data.file_utils import get_project_root, expand_path
from collab_env.sim.boids.run_boids_simulator import run_simulator

from tests.sim_test_util import remove_run_folder, create_run_folder


def test_sim_files_no_video_no_vis():
    """

    Returns:

    """
    remote_test = "CI" in os.environ

    if not remote_test:
        num_episodes = 1
        config_file = "tests/sim_test_2_config.yaml"

        # clear the sim-runs folder from previous tests -- a little dicey
        sim_runs_path = expand_path("sim-output/tests-sim-runs-2", get_project_root())
        remove_run_folder(sim_runs_path=sim_runs_path)
        create_run_folder(sim_runs_path=sim_runs_path)

        # program_path = expand_path(
        #     "collab_env/sim/boids/run_boids_simulator.py", get_project_root()
        # )
        #
        # # Test to see that the run_boids_simulator runs successfully with test config file
        # result = os.system(f"python {program_path} -cf {config_file}")
        # assert result == 0, f"program result {result}"

        run_simulator(expand_path(config_file, get_project_root()))

        # Test to see that output folder was created. There should be exactly 1 of these.
        folder_list = glob(f"{sim_runs_path}/boids_sim_run_2*")
        assert len(folder_list) == 1, f"folder list {folder_list}"

        # Test to see the proper number of episodes were recorded.
        parquet_file_list = glob(f"{folder_list[0]}/*.parquet")
        assert len(parquet_file_list) == num_episodes, (
            f"parquet file list {parquet_file_list}"
        )

        # Test to see that a video is stored in the run folder for each episode
        video_file_list = glob(f"{folder_list[0]}/*.mp4")
        assert len(video_file_list) == 0, f"video file list {video_file_list}"

        # Check to see that the log file was not created since logging is turned off in the config file
        log_file_list = glob(f"{folder_list[0]}/*.log")
        assert len(log_file_list) == 0, f"log file list {log_file_list}"

        for episode in range(num_episodes):
            # Test to see the proper number of frames were recorded for the episode.
            # Should be 5 frames.
            df = pq.read_pandas(
                parquet_file_list[episode], columns=["time", "type"]
            ).to_pandas()
            assert df.max(axis=0)["time"] == 5, "number of time steps"

            # Test to see the proper number of agents were recorded for the first episode.
            # Should be 20 agents over 6 time steps (including 0), so agent should appear 120 times.
            assert df["type"].value_counts().get("agent", 0) == 6 * 20, (
                "number of agents"
            )
