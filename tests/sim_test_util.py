import os
import shutil
from glob import glob

import numpy as np
import pyarrow.parquet as pq
from loguru import logger

from collab_env.data.file_utils import get_project_root, expand_path


def remove_run_folder(sim_runs_path=None):
    if sim_runs_path is not None:
        if os.path.isdir(sim_runs_path):
            logger.info(f"removing path {sim_runs_path}")
            shutil.rmtree(sim_runs_path)


def create_run_folder(sim_runs_path=None):
    if sim_runs_path is not None:
        logger.info(f"making directory {sim_runs_path}")
        os.mkdir(f"{sim_runs_path}")


def check_parquet_file(
    folder, num_episodes=1, num_frames=5, num_agents=20, num_targets=1
):
    """
    Args:
        folder:
        num_episodes:
        num_frames:
        num_agents:
        num_targets:

    Tests the following:
    1. max id of an agent in the dataframe matches the number of agents
    2. max id of a target in the dataframe matches the number of targets
    3. min time is 0 for both targets and agents
    4. max time for both targets and agents is the number of frames in the episode
    5. location of an agent at time t+1 if the location at time t + velocity at time t+1
    6. the norm of location - mesh closest point is within tolerance of distance to mesh closest point in dataframe
    7. the norm of location - target mesh closest point is within tolerance of distance to target mesh closest point in dataframe

    Returns:

    """
    # get the parquet file in the first folder
    # parquet_file_list = glob(f"{run_folder}/*.parquet")
    # Test to see the proper number of episodes were recorded. Should be 3 episodes.
    parquet_file_list = glob(f"{folder}/*.parquet")
    assert len(parquet_file_list) == num_episodes, (
        f"parquet file list {parquet_file_list}"
    )
    logger.info(f"parquet folder list = {parquet_file_list}")

    for episode in range(num_episodes):
        logger.debug(f"episode = {episode}")
        # read the dataframe from the parquet file
        logger.debug(f"reading parquet file {parquet_file_list[episode]}")
        episode_df = pq.read_pandas(parquet_file_list[episode]).to_pandas()
        logger.debug(f"data frame read\n {episode_df}")
        # Test to see the proper number of agents were recorded for the first episode.
        # Should be 20 agents over 6 time steps (including 0), so agent should appear 120 times.
        assert (
            episode_df["type"].value_counts().get("agent", 0)
            == (num_frames + 1) * num_agents
        ), f"num agents record = {episode_df['type'].value_counts().get('agent', 0)}"

        # get all the target rows
        targets_df = episode_df.loc[episode_df["type"] == "env"]

        # get all the agent rows
        agents_df = episode_df.loc[episode_df["type"] == "agent"]

        assert num_agents == agents_df["id"].max(), (
            f"num agents mismatch {agents_df['id'].max()}"
        )
        assert num_targets == targets_df["id"].max(), (
            f"num targets mismatch {targets_df['id'].max()}"
        )

        min_time_agents = agents_df["time"].min()
        min_time_targets = targets_df["time"].min()
        assert min_time_agents == min_time_targets, (
            "min time for agents and targets mismatch"
        )
        assert min_time_agents == 0, "min time for agents not 0"

        max_time_agents = agents_df["time"].max()
        max_time_targets = targets_df["time"].max()
        max_agents_row = agents_df.loc[agents_df["time"].idxmax()]
        logger.info(f"row with max time in agents df \n {max_agents_row}")
        assert max_time_agents == max_time_targets, (
            "max time for agents and targets mismatch"
        )
        assert max_time_agents == num_frames, "max time not equal to number of frames"

        # Test to see the proper number of agents were recorded for this episode.
        # Should be (num_frames+1) * num_agents since we record 0-th frame as initial
        # positions.
        assert (
            episode_df["type"].value_counts().get("agent", 0)
            == (num_frames + 1) * num_agents
        ), f"num agents record = {episode_df['type'].value_counts().get('agent', 0)}"

        for i in range(num_agents):
            # get the rows for agent i
            agent = agents_df.loc[episode_df["id"] == i]

            # get the list of locations and velocities for agent i (one for each time step)
            location = agent[["x", "y", "z"]].to_numpy()
            velocity = agent[["v_x", "v_y", "v_z"]].to_numpy()

            # get the mesh scene distances and closest points for agent i
            mesh_scene_closest_point = agent["mesh_scene_closest_point"].to_numpy()
            mesh_scene_distance = agent["mesh_scene_distance"].to_numpy()

            # check each time step (should be able to do this with numpy instead of a for loop
            for time in range(len(location) - 1):
                logger.debug(f"time = {time}")
                if np.any(location[time + 1] - (location[time] + velocity[time + 1])):
                    logger.debug("failed velocity change of position test ")
                    logger.debug(location[time])
                    logger.debug(velocity[time + 1])
                    logger.debug(location[time + 1])
                    logger.debug(location[time] + velocity[time + 1])
                    logger.debug(
                        location[time + 1] - (location[time] + velocity[time + 1])
                    )
                    assert False

                diff = (
                    np.linalg.norm(location[time] - mesh_scene_closest_point[time])
                    - mesh_scene_distance[time]
                    < 0.001
                )
                if not diff.all():
                    logger.debug("failed mesh scene distance test ")
                    logger.debug(f"location\n {location[time]}")
                    logger.debug(f"{mesh_scene_closest_point[time]}")
                    logger.debug(f"{mesh_scene_distance[time]}")
                    logger.debug(
                        f"{np.linalg.norm(location[time] - mesh_scene_closest_point[time])}"
                    )
                    logger.debug(
                        f"{np.linalg.norm(location[time] - mesh_scene_closest_point[time]) - mesh_scene_distance[time]}"
                    )
                    assert False, "failed mesh scene distance test"

                for target in range(1, num_targets + 1):
                    target_mesh_closest_point = agent[
                        f"target_mesh_closest_point_{target}"
                    ].to_numpy()
                    distance_to_target_mesh_closest_point = agent[
                        f"distance_to_target_mesh_closest_point_{target}"
                    ].to_numpy()
                    diff = (
                        np.linalg.norm(location[time] - target_mesh_closest_point[time])
                        - distance_to_target_mesh_closest_point[time]
                        < 0.001
                    )
                    # logger.info(f'target mesh diff = {np.linalg.norm(location[time] - target_mesh_closest_point[time]) - distance_to_target_mesh_closest_point[time]}')
                    if not diff.all():
                        logger.debug("failed target mesh  distance test ")
                        logger.debug(f"location\n {location[time]}")
                        logger.debug(f"{target_mesh_closest_point[time]}")
                        logger.debug(f"{distance_to_target_mesh_closest_point[time]}")
                        logger.debug(
                            f"{np.linalg.norm(location[time] - distance_to_target_mesh_closest_point[time])}"
                        )
                        logger.debug(
                            f"{np.linalg.norm(location[time] - mesh_scene_closest_point[time]) - mesh_scene_distance[time]}"
                        )
                        assert False, "failed target mesh distance test"


def sim_check_files(
    config_file=None,
    sim_runs_path=None,
    num_episodes=3,
    video=True,
    num_log_files=1,
    num_time_steps=5,
    num_agents=20,
    num_targets=5,
):
    """
    Tests the following scenarios:
    1. run_boids_simulator successfully runs with command line specified config file.
    2. output folder is created.
    3. three episodes of parquet files are created.
    4. three videos are created, one for each episode.
    5. a single log file is created for all episodes
    6. parquet file includes correct number of frames for each episode.
    7. parquet file includes correct number of agents for each episode.
    8. the size of each video indicates that it is not empty. Empty videos seem to be 257 bytes.

    If the test is remote, the files need to be removed after the run. If not remote, we
    leave the files so we can look at them to make sure the tests are working correctly, but
    we remove the folder on start so we don't end up with a bunch of stuff in the test folder.

    Returns:

    """
    if sim_runs_path is None:
        sim_runs_path = expand_path("sim-output/tests-sim-runs-1", get_project_root())

    remote_test = "CI" in os.environ
    if not remote_test:
        # Clear the sim-runs folder from previous tests -- a little dicey
        # If this is a remote test, these files shouldn't be here, so don't bother. They
        # get cleaned up at the end
        remove_run_folder(sim_runs_path)

    create_run_folder(sim_runs_path)

    print(f"\nsim runs path {sim_runs_path}\n")

    if config_file is None:
        config_file = "tests/sim_test_1_config.yaml"

    program_path = expand_path(
        "collab_env/sim/boids/run_boids_simulator.py", get_project_root()
    )

    # Test to see that the run_boids_simulator runs successfully with test config file
    result = os.system(f"python {program_path} -cf {config_file}")
    assert result == 0, f"failed -- python {program_path} -cf {config_file}"

    # Test to see that output folder was created. There should be exactly 1 of these.
    folder_list = glob(f"{sim_runs_path}/boids_sim_run_1*")
    assert len(folder_list) == 1, f"folder list = {folder_list}"

    # Test to see the proper number of episodes were recorded. Should be 3 episodes.
    parquet_file_list = glob(f"{folder_list[0]}/*.parquet")
    assert len(parquet_file_list) == num_episodes, (
        f"parquet file list {parquet_file_list}"
    )

    # Test to see that a video is stored in the run folder for each episode
    video_file_list = glob(f"{folder_list[0]}/*.mp4")
    print(f"video file list = {video_file_list}")
    assert len(video_file_list) == num_episodes if video else 0, (
        f"video file list {video_file_list}"
    )

    # Check to see that the log file was created correctly
    log_file_list = glob(f"{folder_list[0]}/*.log")
    assert len(log_file_list) == num_log_files, f"log file list {log_file_list}"

    check_parquet_file(
        folder_list[0],
        num_episodes=num_episodes,
        num_frames=num_time_steps,
        num_agents=num_agents,
        num_targets=num_targets,
    )

    for episode in range(num_episodes):
        # Test to see the proper number of frames were recorded for the episode.
        # Should be 5 frames.
        df = pq.read_pandas(
            parquet_file_list[episode], columns=["time", "type"]
        ).to_pandas()
        assert df.max(axis=0)["time"] == num_time_steps, (
            f"max time steps {df.max(axis=0)['time']}"
        )

        # Test to see the proper number of agents were recorded for the first episode.
        # Should be 20 agents over 6 time steps (including 0), so agent should appear 120 times.
        assert (
            df["type"].value_counts().get("agent", 0)
            == (num_time_steps + 1) * num_agents
        ), f"num agents record = {df['type'].value_counts().get('agent', 0)}"

        # Check the size of the video to make sure the file is not empty
        # This number may need to change. This number is based on the
        # number of frames being 5 and the particular image size
        # when I was running this initially. The number is not fixed. It
        # apparently is different for different runs, so just check
        # that it is not empty, so let's just go with 75000. When the
        # frames are not written at all, the file it is about 257 bytes.
        if video:
            assert os.path.getsize(video_file_list[episode]) > 1000, (
                f"video size {os.path.getsize(video_file_list[episode])}"
            )

    if remote_test:
        # if this is not a remote test, we should clean up the files
        # clear the sim-runs folder from previous tests
        remove_run_folder(sim_runs_path)
