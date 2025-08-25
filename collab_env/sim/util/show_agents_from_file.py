"""
TOC -- 080625 2:51PM

This program will read the agent positions from a file and display them in the viewer and video through time.

"""

import argparse
import os
import time
import open3d as o3d

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


import collab_env.sim.gymnasium_env as gymnasium_env  # noqa: F401
from collab_env.data.file_utils import get_project_root, expand_path
from collab_env.sim.boids.sim_utils import function_filter


def get_config(config_file):
    if config_file is not None:
        config_filename = expand_path(config_file, get_project_root())
    else:
        config_filename = expand_path(
            "collab_env/sim/boids/trajectory_config.yaml", get_project_root()
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


def plot_trajectory(points):
    # Create a LineSet
    lines = [[i, i + 1] for i in range(len(points) - 1)]
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(points)
    line_set.lines = o3d.utility.Vector2iVector(lines)

    # Optionally, set colors for the lines
    line_set.paint_uniform_color([0, 1, 0])  # Red color

    # Visualize
    o3d.visualization.draw_geometries([line_set])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="run_boids_simulator",
        description="Simulates boids in a 3D environment",
        epilog="---",
    )
    parser.add_argument("-cf", "--config_file")
    args = parser.parse_args()

    #
    # Get the config file name if specified on the command line
    #
    # TOC -- 080625 2:52PM
    # Seems like this should go in some sort of util function since we need it for multiple programs -- unless we
    # combine all of this into one simulator program -- which might be the right idea actually.

    trajectory_config, trajectory_config_filename = get_config(args.config_file)
    print("trajectory config filename: ", trajectory_config_filename)

    # TOC -- 080225 9:15AM
    # Create the output folder
    new_folder_name = f"{trajectory_config['simulator']['run_main_folder']}/{trajectory_config['simulator']['run_sub_folder_prefix']}-started-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    new_run_folder = expand_path(new_folder_name, get_project_root())
    os.mkdir(new_run_folder)

    #
    # Set up the logger
    #
    setup_logging(trajectory_config, new_run_folder)

    # TOC -- 080225 9:54AM
    # Copy the config file into the run folder to record configuration for the run.
    # There may be a better way to do this to make sure we get all parameters stored
    # in case there are still hardcoded values in the code -- which should be removed
    # at some point.
    trajectory_copied_config_file_path = expand_path(
        "trajectory_config.yaml", new_run_folder
    )
    shutil.copy(trajectory_config_filename, trajectory_copied_config_file_path)

    # TOC -- 080225
    # Find the part for the video in the run folder.
    video_file_path = expand_path(
        f"video.{trajectory_config['visuals']['video_file_extension']}", new_run_folder
    )
    logger.debug(f"video path {video_file_path}")

    #
    # Determine render mode for environment
    #
    if trajectory_config["visuals"]["show_visualizer"]:
        render_mode = "human"
    else:
        render_mode = ""

    #
    # Stored this since I need to pass it to both the environment and the agent and
    # want it to be easier to keep in sync if the config format changes.
    #

    trajectory_input_folder = expand_path(
        trajectory_config["files"]["trajectory_folder"], get_project_root()
    )
    trajectory_file = expand_path(
        trajectory_config["files"]["trajectory_file"],
        trajectory_input_folder,
    )

    #
    # Get the configuration file from the simulation run. This might not be ideal since I need to do this for
    # the GNN. And not sure I need it. Actually, I do because I have to recreate the environment as it existed
    # when the trajectories where created.
    #
    sim_config, sim_config_filename = get_config(
        expand_path("config.yaml", trajectory_input_folder)
    )

    df = pd.read_parquet(trajectory_file)
    print("df:\n", df[["id", "type", "time", "x", "y", "z"]])
    input("hit enter")
    num_time_steps = df["time"].max()
    print("num time ", num_time_steps)
    num_agents = len(df.loc[(df["time"] == 0) & (df["type"] == "agent")])
    # assert num_time_steps == sim_config["simulator"]["num_frames"]
    #

    # get the
    agent_trajectories_list = []  # = np.zeros((num_time_steps+1, num_agents, 3))
    for t in range(num_time_steps):
        temp = df.loc[(df["type"] == "agent") & (df["time"] == t) & (df["id"] == 1)]
        print("temp ", temp[["x", "y", "z"]].to_numpy())
        agent_trajectories_list.append(temp[["x", "y", "z"]].to_numpy()[0])

        # agent_trajectories.append(temp[['x', 'y', 'z']])
    # agent_trajectories = [
    #
    #         ]
    #         for t in range(1, num_time_steps + 1)
    #     ]
    agent_trajectories = np.array(agent_trajectories_list)
    print("agent loc:\n", agent_trajectories)
    plot_trajectory(agent_trajectories)

    assert False
    target_trajectories = [
        df.loc[(df["type"] == "env") & (df["time"] == t)][["x", "y", "z"]].to_numpy()
        for t in range(1, num_time_steps + 1)
    ]
    print("target loc:\n", target_trajectories)
    assert False

    target_creation_time = sim_config["simulator"]["target_creation_time"]

    #
    # Create environment.
    #
    env = gym.make(
        "gymnasium_env/BoidsWorldSimple-v0",
        render_mode=render_mode,
        num_agents=sim_config["simulator"]["num_agents"],
        num_targets=sim_config["simulator"]["num_targets"],
        num_ground_targets=sim_config["simulator"]["num_ground_targets"],
        walking=sim_config["simulator"]["walking"],
        show_box=sim_config["simulator"]["show_box"],
        store_video=sim_config["visuals"]["store_video"],
        show_visualizer=sim_config["visuals"]["show_visualizer"],
        vis_width=sim_config["visuals"]["width"],
        vis_height=sim_config["visuals"]["height"],
        video_file_path=video_file_path,
        video_codec=sim_config["visuals"]["video_codec"],
        video_fps=sim_config["visuals"]["video_fps"],
        agent_shape=sim_config["visuals"]["agent_shape"],
        agent_color=sim_config["visuals"]["agent_color"],
        agent_scale=sim_config["visuals"]["agent_scale"],
        target_scale=sim_config["visuals"]["target_scale"],
        agent_mean_init_velocity=sim_config["agent"]["mean_init_velocity"],
        agent_variance_init_velocity=sim_config["agent"]["variance_init_velocity"],
        box_size=sim_config["environment"]["box_size"],
        scene_scale=sim_config["environment"]["scene_scale"],
        scene_filename=sim_config["meshes"]["mesh_scene"],
        scene_position=sim_config["environment"]["scene_position"],
        scene_angle=np.pi * np.array(sim_config["meshes"]["scene_angle"]) / 180.0,
        target_creation_time=target_creation_time,
        run_trajectories=True,
        agent_trajectories=agent_trajectories,
        target_trajectories=target_trajectories,
    )

    """ 
    TOC -- 080705 1:28PM
    Need to be able to set agent locations in the rendering, which requires that the rendering be separated 
    from the environment. This is a bit of a chore. 
    
    1. set initial agent locations 
    2. for each time step 
    3.    set the agent locations to the locations at the next time step
    4. done -- piece of cake, just need the rendering separate. 
    
    I could do this with the current environment by passing the agent locations to the step function. I don't totally
    trust subtracting off previous locations to get the velocity, so maybe a setting in the environment to take 
    trajectories for now just to get the figure done and then move everything when I get some time.   
    """

    # Reset the environment
    obs, info = env.reset()

    done = False

    """
    TOC -- 073125 -- 8:46AM
    Need to decide on how this is going to end. 

    TOC -- 073124 2:21PM 
    We are going with a limited number of frames.
    """
    # logger.debug('starting main loop ')

    time.sleep(2000)
    # while not done:
    for time_step in tqdm(range(num_time_steps)):
        # step() will ignore the action because we have run_trajectories set to true,
        # but I am putting it in for now so keep Gymnasium from complaining about the
        # action space.
        next_obs, reward, terminated, truncated, info = env.step(
            agent_trajectories[time_step]
        )
        time.sleep(30)

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

    if sim_config["visuals"]["store_video"]:
        # change the name of the video file to include the episode
        episode_video_file_path = expand_path(
            f"video.{sim_config['visuals']['video_file_extension']}", new_run_folder
        )
        logger.debug(f"episode video path {episode_video_file_path}")
        shutil.move(video_file_path, episode_video_file_path)
