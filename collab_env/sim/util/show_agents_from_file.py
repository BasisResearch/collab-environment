"""
TOC -- 080625 2:51PM

This program will read the agent positions from a file and display them in the viewer and video through time.

"""

import argparse
import os

from datetime import datetime

import numpy as np

from tqdm import tqdm  # Progress bar
import gymnasium as gym
import yaml
from loguru import logger

import pandas as pd
import pyarrow.parquet as pq
import pyarrow as pa
import shutil


from collab_env.sim.boids.boidsAgents import BoidsWorldAgent
import collab_env.sim.gymnasium_env as gymnasium_env  # noqa: F401
from collab_env.data.file_utils import get_project_root, expand_path
from collab_env.sim.boids.sim_utils import add_obs_to_df, function_filter


def get_config():
    parser = argparse.ArgumentParser(
        prog="run_boids_simulator",
        description="Simulates boids in a 3D environment",
        epilog="---",
    )
    parser.add_argument("-cf", "--config_file")
    args = parser.parse_args()
    if args.config_file:
        config_filename = expand_path(args.config_file, get_project_root())
    else:
        config_filename = expand_path(
            "collab_env/sim/boids/config.yaml", get_project_root()
        )

    config = yaml.safe_load(open(config_filename))
    return config, config_filename


def setup_logging(config, run_folder):
    if not config["logging"]["logging"]:
        logger.disable("")
    else:
        # TOC -- 080325 11:19AM
        # Remove the existing handlers and add a new one attached to the
        # log file in the run folder and with the prefix specified in the config
        # file.
        logger.remove()
        if len(config["logging"]["log_functions"]) > 0:
            logger.add(
                expand_path(f"{config['logging']['logfile_prefix']}.log", run_folder),
                level=config["logging"]["log_level"],
                filter=function_filter(
                    function_list=config["logging"]["log_functions"]
                ),
            )
        else:
            logger.add(
                expand_path(f"{config['logging']['logfile_prefix']}.log", run_folder),
                level=config["logging"]["log_level"],
            )


def get_pandas_columns():
    distance_columns = [f"distance_target_{t}" for t in range(1, num_targets + 1)]
    pandas_columns = [
        "id",
        "type",
        "time",
        "x",
        "y",
        "z",
        "v_x",
        "v_y",
        "v_z",
    ] + distance_columns
    return pandas_columns


if __name__ == "__main__":
    #
    # Get the config file name if specified on the command line
    #
    # TOC -- 080625 2:52PM
    # Seens like this should go in some sort of util function since we need it for multiple programs -- unless we
    # combine all of this into one simulator program -- which might be the right idea actually.

    config, config_filename = get_config()

    # TOC -- 080225 9:15AM
    # Create the output folder
    new_folder_name = f"{config['simulator']['run_main_folder']}/{config['simulator']['run_sub_folder_prefix']}-started-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    new_run_folder = expand_path(new_folder_name, get_project_root())
    os.mkdir(new_run_folder)

    #
    # Set up the logger
    #
    setup_logging(config, new_run_folder)

    # TOC -- 080225 9:54AM
    # Copy the config file into the run folder to record configuration for the run.
    # There may be a better way to do this to make sure we get all parameters stored
    # in case there are still hardcoded values in the code -- which should be removed
    # at some point.
    copied_config_file_path = expand_path("config.yaml", new_run_folder)
    shutil.copy(config_filename, copied_config_file_path)

    # TOC -- 080225
    # Find the part for the video in the run folder.
    video_file_path = expand_path(
        f"video.{config['visuals']['video_file_extension']}", new_run_folder
    )
    logger.debug(f"video path {video_file_path}")

    #
    # Determine render mode for environment
    #
    if config["visuals"]["show_visualizer"]:
        render_mode = "human"
    else:
        render_mode = ""

    #
    # Stored this since I need to pass it to both the environment and the agent and
    # want it to be easier to keep in sync if the config format changes.
    #
    target_creation_time = config["simulator"]["target_creation_time"]

    #
    # Create environment and agent
    #
    env = gym.make(
        "gymnasium_env/BoidsWorldSimple-v0",
        render_mode=render_mode,
        num_agents=config["simulator"]["num_agents"],
        num_targets=config["simulator"]["num_targets"],
        num_ground_targets=config["simulator"]["num_ground_targets"],
        walking=config["simulator"]["walking"],
        show_box=config["simulator"]["show_box"],
        store_video=config["visuals"]["store_video"],
        show_visualizer=config["visuals"]["show_visualizer"],
        vis_width=config["visuals"]["width"],
        vis_height=config["visuals"]["height"],
        video_file_path=video_file_path,
        video_codec=config["visuals"]["video_codec"],
        video_fps=config["visuals"]["video_fps"],
        agent_shape=config["visuals"]["agent_shape"],
        agent_color=config["visuals"]["agent_color"],
        agent_scale=config["visuals"]["agent_scale"],
        target_scale=config["visuals"]["target_scale"],
        agent_mean_init_velocity=config["agent"]["mean_init_velocity"],
        agent_variance_init_velocity=config["agent"]["variance_init_velocity"],
        box_size=config["environment"]["box_size"],
        scene_scale=config["environment"]["scene_scale"],
        scene_filename=config["meshes"]["mesh_scene"],
        scene_position=config["environment"]["scene_position"],
        scene_angle=np.pi * np.array(config["meshes"]["scene_angle"]) / 180.0,
        target_creation_time=target_creation_time,
    )

    agent = BoidsWorldAgent(
        env=env,
        num_agents=config["simulator"]["num_agents"],
        num_targets=config["simulator"]["num_targets"],
        walking=config["simulator"]["walking"],
        has_mesh_scene=(config["meshes"]["mesh_scene"] != ""),
        min_ground_separation=config["agent"]["min_ground_separation"],
        min_separation=config["agent"]["min_separation"],
        neighborhood_dist=config["agent"]["neighborhood_dist"],
        ground_weight=config["agent"]["ground_weight"],
        separation_weight=config["agent"]["separation_weight"],
        alignment_weight=config["agent"]["alignment_weight"],
        cohesion_weight=config["agent"]["cohesion_weight"],
        target_weight=[0.0]
        * config["simulator"][
            "num_targets"
        ],  # start at all 0's and add weights when created -- not a great design.
        max_speed=config["agent"]["max_speed"],
        min_speed=config["agent"]["min_speed"],
        max_force=config["agent"]["max_force"],
    )

    #
    # Run the episodes
    #
    num_targets = config["simulator"]["num_targets"]
    pandas_columns = get_pandas_columns()

    # Reset the environment
    obs, info = env.reset()

    # TOC -- 080225 8:58AM
    # create the dataframe for the simulation output
    df = pd.DataFrame(columns=pandas_columns)

    done = False

    """
    TOC -- 073125 -- 8:46AM
    Need to decide on how this is going to end. 

    TOC -- 073124 2:21PM 
    We are going with a limited number of frames.
    """
    # logger.debug('starting main loop ')

    df = pd.read_parquet()
    trajectory_file = expand_path(
        config["trajectory_folder"]["trajectory_filename"], get_project_root()
    )
    # while not done:
    for time_step in tqdm(range(config["simulator"]["num_frames"])):
        # Agent chooses action, which is the velocity
        action = agent.get_action(obs)

        # Take the action in the environment and observe the result
        next_obs, reward, terminated, truncated, info = env.step(action)

        # TOC -- 080225 8:58AM
        # Record the observation
        df = add_obs_to_df(df, next_obs, time_step=(time_step + 1))
        # Observe the next state
        obs = next_obs

        # ignore terminated for now since we are just running for a specified number of frames
        # done = terminated or truncated
        # done = True

    logger.info(f"df columns = {df.columns}")
    logger.info(f"positions:\n{df[['x', 'y', 'z']]}")
    logger.info(f"velocities:\n{df[['v_x', 'v_y', 'v_z']]}")
    logger.info(f"distances:\n{df[['distance_target_1']]}")

    table = pa.Table.from_pandas(df)
    logger.debug(f"table \n {table}")

    file_path = expand_path(
        f"episode-completed-{datetime.now().strftime('%Y%m%d-%H%M%S')}.parquet",
        # f"episode-{episode}.parquet",
        new_run_folder,
    )
    logger.info(f"writing output to {file_path}")
    pq.write_table(table, file_path)

    if config["visuals"]["store_video"]:
        # change the name of the video file to include the episode
        episode_video_file_path = expand_path(
            f"video.{config['visuals']['video_file_extension']}", new_run_folder
        )
        logger.debug(f"episode video path {episode_video_file_path}")
        shutil.move(video_file_path, episode_video_file_path)
