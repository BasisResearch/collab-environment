import os
import shutil
from glob import glob

# import pandas as pd
import pyarrow.parquet as pq
from loguru import logger

# import pyarrow as pa
from collab_env.data.file_utils import get_project_root, expand_path


def test_sim_files_visualizer():
    config_file = "tests/sim_test_2_config.yaml"

    # clear the sim-runs folder from previous tests -- a little dicey
    sim_runs_path = expand_path("tests/sim-runs", get_project_root())
    logger.info(f"removing path {sim_runs_path}")
    shutil.rmtree(sim_runs_path)
    logger.info(f"making directory {sim_runs_path}")
    os.mkdir(f"{sim_runs_path}")

    program_path = expand_path(
        "collab_env/sim/boids/run_boids_simulator.py", get_project_root()
    )

    # Test to see that the run_boids_simulator runs successfully with test config file
    result = os.system(f"python {program_path} -cf {config_file}")
    assert result == 0

    # Test to see that output folder was created
    folder_list = glob(f"{sim_runs_path}/*")
    assert len(folder_list) > 0

    # Test to see the proper number of episodes were recorded. Should be 3 episodes.
    result_file_list = glob(f"{folder_list[0]}/*.parquet")
    assert len(result_file_list) == 1

    # Test to see the proper number of frames were recorded for the first episode. Should be 5 frames.
    df = pq.read_pandas(result_file_list[0], columns=["time", "type"]).to_pandas()
    assert df.max(axis=0)["time"] == 5

    # Test to see the proper number of agents were recorded for the first episode.
    # Should be 20 agents over 5 agents, so agent should appear 50 times.
    assert df["type"].value_counts().get("agent", 0) == 5 * 20

    # Test to see that a video is stored in the run folder for each episode
    result_file_list = glob(f"{folder_list[0]}/*.mp4")
    assert len(result_file_list) == 1

    # Check the size of the video to make sure the file is not empty
    # This number may need to change. This number is based on the
    # number of frames being 5 and the particular image size
    # when I was running this initially. The number is not fixed. It
    # apparently is different for different runs, so just check
    # that it is not empty, so let's just go with 75000. For lower
    # resolution, this needs to be smaller, so just go with 1K When the
    # frames are not written at all, the file it is about 257 bytes.
    assert os.path.getsize(result_file_list[0]) > 1000

    # Check to see that the log file was not created since logging
    # set to false in config.
    result_file_list = glob(f"{folder_list[0]}/*.log")
    assert len(result_file_list) == 0
