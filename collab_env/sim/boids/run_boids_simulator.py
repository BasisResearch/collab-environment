"""
TOC
- 7/21/25 12:29 -- they seem to move together with all weights at 1. Need to bound them inside a cube and play with neighborhood sizes as some of
them seem to stop -- not sure why that is happening.
- 7/21/25 -- runs with 5 agents and a moving target but the agents scatter.
- works as a 2D grid
- 7/3/25 23:42 the open3D visualizer opens and runs while the environment is running.
"""

import argparse
import os

from datetime import datetime

import numpy as np

from tqdm import tqdm  # Progress bar
import gymnasium as gym
import yaml
from loguru import logger

import pyarrow.parquet as pq
import pyarrow as pa
import shutil


from collab_env.sim.boids.boidsAgents import BoidsWorldAgent
import collab_env.sim.gymnasium_env as gymnasium_env  # noqa: F401
from collab_env.data.file_utils import get_project_root, expand_path
from collab_env.sim.boids.sim_utils import (
    add_obs_to_df,
    function_filter,
    plot_trajectories,
)

# NUM_AGENTS = 40
# WALKING = False


"""
TOC -- 080825 10:10AM
This needs to be done much more efficiently.  
"""


if __name__ == "__main__":
    #
    # Get the config file name if specified on the command line
    #
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

    if config["visuals"]["show_visualizer"]:
        render_mode = "human"
    else:
        render_mode = ""

    # TOC -- 080225 9:15AM
    # Create the output folder
    """
    # TOC -- 080425 1:49PM
    # Using the time in the folder name seems to be causing a problem for the pytest runs. Furthermore, we could have
    # multiple runs happening at the same time, so let's try using the process and thread ids to distinguish.  
    """
    new_folder_name = f"{config['simulator']['run_main_folder']}/{config['simulator']['run_sub_folder_prefix']}-started-{datetime.now().strftime('%Y%m%d-%H%M%S')}"

    new_run_folder = expand_path(new_folder_name, get_project_root())
    os.mkdir(new_run_folder)

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
                expand_path(
                    f"{config['logging']['logfile_prefix']}.log", new_run_folder
                ),
                level=config["logging"]["log_level"],
                filter=function_filter(
                    function_list=config["logging"]["log_functions"]
                ),
            )
        else:
            logger.add(
                expand_path(
                    f"{config['logging']['logfile_prefix']}.log", new_run_folder
                ),
                level=config["logging"]["log_level"],
            )

    # TOC -- 080225 9:54AM
    # Copy the config file into the run folder to record configuration for the run.
    # There may be a better way to do this to make sure we get all parameters stored
    # in case there are still hardcoded values in the code -- which should be removed
    # at some point.
    copied_config_file_path = expand_path("config.yaml", new_run_folder)
    shutil.copy(config_filename, copied_config_file_path)

    # TOC -- 080225
    # Find the path for the video in the run folder.
    video_file_path = expand_path(
        f"video.{config['visuals']['video_file_extension']}", new_run_folder
    )
    logger.debug(f"video path {video_file_path}")

    target_creation_time = config["simulator"]["target_creation_time"]
    """ 
    TOC -- 080825 7:15PM
    If no fixed target positions were specified, we should pass None to the environment
    """
    fixed_target_position = config["environment"]["target_position"]
    if len(fixed_target_position) == 0:
        fixed_target_position = None
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
        agent_init_range_low=config["agent"]["init_range_low"],
        agent_init_range_high=config["agent"]["init_range_high"],
        agent_height_range_low=config["agent"]["height_range_low"],
        agent_height_range_high=config["agent"]["height_range_high"],
        agent_height_init_min=config["agent"]["height_init_min"],
        agent_height_init_max=config["agent"]["height_init_max"],
        target_init_range_low=config["environment"]["init_range_low"],
        target_init_range_high=config["environment"]["init_range_high"],
        target_height_init_max=config["environment"]["height_init_max"],
        target_mesh_file=config["meshes"]["sub_mesh_target"]
        if config["simulator"]["submesh_target"]
        else None,
        target_mesh_init_color=config["visuals"]["target_mesh_init_color"],
        target_mesh_color=config["visuals"]["target_mesh_color"],
        box_size=config["environment"]["box_size"],
        scene_scale=config["environment"]["scene_scale"],
        scene_filename=config["meshes"]["mesh_scene"],
        scene_position=config["environment"]["scene_position"],
        scene_angle=np.pi * np.array(config["meshes"]["scene_angle"]) / 180.0,
        target_creation_time=target_creation_time,
        target_positions=fixed_target_position,
        color_tracks_by_time=config["tracks"]["color_by_time"],
        number_track_color_groups=config["tracks"]["number_of_color_groups"],
        track_color_rate=config["tracks"]["track_color_rate"],
        saved_image_path=new_run_folder,
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
        random_walk=config["agent"]["random_walk"],
    )

    num_targets = config["simulator"]["num_targets"]
    # distance_columns = [f"distance_target_center_{t}" for t in range(1, num_targets + 1)]
    # distance_target_mesh = [f"distances_to_target_mesh_closest_point_{t}" for t in range(1, num_targets + 1)]
    # closest_point_columns = [
    #     f"closest_point_target_{t}" for t in range(1, num_targets + 1)
    # ]
    # pandas_columns = (
    #     [
    #         "id",
    #         "type",
    #         "time",
    #         "x",
    #         "y",
    #         "z",
    #         "v_x",
    #         "v_y",
    #         "v_z",
    #     ]
    #     + distance_columns
    #     + closest_point_columns
    # )

    #
    # There should be one seed for each episode
    #
    seed_list = config["simulator"]["seed"]
    #
    # Run the episodes
    #
    for episode in tqdm(range(config["simulator"]["num_episodes"])):
        # Start a new episode

        logger.debug(f"main(): starting episode {episode}")

        # Reset the environment
        obs, info = env.reset(seed=seed_list[episode])

        # TOC -- 080225 8:58AM
        # create the dataframe for the simulation output
        # df = pd.DataFrame(columns=pandas_columns)

        # TOC -- 080725 10:45PM
        # Add the initial positions to the dataframe
        df = add_obs_to_df(None, obs, time_step=0)
        done = False

        #
        # MAIN LOOP
        #

        for time_step in tqdm(range(config["simulator"]["num_frames"])):
            if time_step == target_creation_time:
                agent.set_target_weight(config["agent"]["target_weight"][0], 0)
                """
                TOC -- 080425 2:40PM
                I can't call this method since the environment is in a wrapper. 
                I need to understand wrappers better.
                """
                # env.create_target()

            # Agent chooses action
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
            if terminated or truncated:
                break  # I hate breaks, why does Python make me do it?

        env.close()

        logger.info(f"episode {episode}: df columns = {df.columns}")
        logger.info(f"positions:\n{df[['x', 'y', 'z']]}")
        logger.info(f"velocities:\n{df[['v_x', 'v_y', 'v_z']]}")
        # logger.info(f"distances:\n{df[['distance_target_1']]}")

        #
        # Dump data to output file
        #
        table = pa.Table.from_pandas(df)
        logger.debug(f"table \n {table}")

        file_path = expand_path(
            f"episode-{episode}-completed-{datetime.now().strftime('%Y%m%d-%H%M%S')}.parquet",
            # f"episode-{episode}.parquet",
            new_run_folder,
        )
        logger.info(f"writing output to {file_path}")
        pq.write_table(table, file_path)

        """
        TOC -- 080825
        plot the trajectories for the paper figures. This need to be redesigned
        so that plotting trajectories is in a separate program that is run on 
        the parquet file rather than with the main simulator. Needs to be able
        to display the agents in the visualizer without storing video and with the 
        ability to snap pictures based on keyboard presses so that users can 
        adjust the camera view and zoom on the visualizer to get the figures they 
        want.  
        """
        if config["simulator"]["show_trajectories"]:
            plot_trajectories(df, env)

        """
        TOC -- 081125 3:37PM
        How is this working? It looks like I am moving the file while the rendering is 
        still writing to it. 
        """
        if config["visuals"]["store_video"]:
            # change the name of the video file to include the episode
            episode_video_file_path = expand_path(
                f"episode-{episode}-video.{config['visuals']['video_file_extension']}",
                new_run_folder,
            )
            logger.debug(f"episode video path {episode_video_file_path}")
            shutil.move(video_file_path, episode_video_file_path)

    logger.info("all episodes complete")
