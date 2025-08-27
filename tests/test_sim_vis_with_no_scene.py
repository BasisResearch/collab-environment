import os
from glob import glob

# import pandas as pd
import pyarrow.parquet as pq

# import pyarrow as pa
from collab_env.data.file_utils import get_project_root, expand_path

from tests.sim_test_util import remove_run_folder, create_run_folder


def test_sim_files_visualizer_no_scene_mesh():
    remote_test = "CI" in os.environ

    #
    # Only do this test if it is not remote since we can't create a visualizer
    # window in a remote test.
    #
    if not remote_test:
        config_file = "tests/sim_test_vis_no_scene_config.yaml"

        # clear the sim-runs folder from previous tests -- a little dicey
        sim_runs_path = expand_path(
            "sim-output/tests-sim-runs-vis-no-scene", get_project_root()
        )
        if not remote_test:
            # if this is not a remote test, we should clean up the files
            # clear the sim-runs folder from previous tests
            remove_run_folder(sim_runs_path)

        create_run_folder(sim_runs_path)

        program_path = expand_path(
            "collab_env/sim/boids/run_boids_simulator.py", get_project_root()
        )

        # Test to see that the run_boids_simulator runs successfully with test config file
        result = os.system(f"python {program_path} -cf {config_file}")
        assert result == 0

        # Test to see that output folder was created
        folder_list = glob(f"{sim_runs_path}/boids_sim_run_3*")
        assert len(folder_list) == 1, "folder_list is {}".format(folder_list)

        # Test to see the proper number of episodes were recorded. Should be 3 episodes.
        result_file_list = glob(f"{folder_list[0]}/*.parquet")
        print(f"parquet folder list for {folder_list[0]} = {result_file_list}")
        assert len(result_file_list) == 1

        # Test to see the proper number of frames were recorded for the first episode. Should be 20 frames.
        df = pq.read_pandas(result_file_list[0], columns=["time", "type"]).to_pandas()
        assert df.max(axis=0)["time"] == 20

        # Test to see the proper number of agents were recorded for the first episode.
        # Should be 20 agents over 20 time steps (including 0), so agent should appear 420 times.
        assert df["type"].value_counts().get("agent", 0) == 21 * 20

        # Test to see that a video is stored in the run folder for each episode
        result_file_list = glob(f"{folder_list[0]}/*.mp4")
        assert len(result_file_list) == 1, "video file existence failure"

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

        #
        # if the test is remote, don't leave the files sitting around.
        #
        if remote_test:
            remove_run_folder(sim_runs_path)
