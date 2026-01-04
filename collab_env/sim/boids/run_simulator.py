"""
--
- 7/21/25 12:29 -- they seem to move together with all weights at 1. Need to bound them inside a cube and play with neighborhood sizes as some of
them seem to stop -- not sure why that is happening.
- 7/21/25 -- runs with 5 agents and a moving target but the agents scatter.
- works as a 2D grid
- 7/3/25 23:42 the open3D visualizer opens and runs while the environment is running.
"""

import argparse
from importlib.metadata import distributions

from datetime import datetime
from pathlib import Path

import numpy as np

from tqdm import tqdm  # Progress bar
import gymnasium as gym
import yaml
from loguru import logger

import pyarrow.parquet as pq
import pyarrow as pa
import shutil

from collab_env.gnn.gnn_agent import GNN_Agents
from collab_env.sim.boids.boid_agents import BoidAgents
import collab_env.sim.gymnasium_env as gymnasium_env  # noqa: F401
from collab_env.data.file_utils import get_project_root, expand_path
from collab_env.sim.boids.sim_utils import (
    function_filter,
    plot_trajectories,
    add_obs_to_df,
)


def setup_logging(config: dict, new_run_folder: Path):
    if not config["logging"]["logging"]:
        logger.disable("")
    else:
        # -- 080325 11:19AM
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


def write_package_list(new_run_folder: Path):
    package_list_file_path = expand_path("package_list.txt", new_run_folder)
    with open(package_list_file_path, "w") as f:
        pairs = sorted(
            (
                (dist.metadata.get("Name") or dist.name).lower(),
                f"{dist.metadata.get('Name') or dist.name}=={dist.version}",
            )
            for dist in distributions()
        )
        result = ""
        for _, line in pairs:
            result += line + "\n"
        f.write(result)


def setup_seed_list(config):
    #
    # There should be one seed for each episode
    #
    seed_list = config["simulator"]["seed"]
    """
    -- 122325 
    if we don't have enough seeds, bail out and give the user a list of seeds to add
    to the config.yaml file. 
    """
    if len(seed_list) < config["simulator"]["num_episodes"]:
        print(
            f"Number of episodes ({config['simulator']['num_episodes']}) greater than number of seeds ({len(seed_list)}). Aborting."
        )
        rng = np.random.default_rng(seed=0)
        unique_ints = rng.choice(
            np.arange(1, 1000000),
            size=config["simulator"]["num_episodes"],
            replace=False,
        )
        print(
            "Here is a list of random seeds for each episode. Put them in the config file and rerun.\n",
            unique_ints.tolist(),
        )
        assert False, (
            f"Not enough random seeds ({len(seed_list)}) in config file. See message above."
        )

    return seed_list


def agent_factory(agent_type: int, config: dict, env: gym.Env):
    if agent_type == 0:
        agents = BoidAgents(config, env)
    elif agent_type == 1:
        agents = GNN_Agents(config, env)
    else:
        agents = None
    return agents


"""
-- 080825 10:10AM
This needs to be done much more efficiently.  
"""


def run_simulator(config_filename):
    config = yaml.safe_load(open(config_filename))
    if config["visuals"]["show_visualizer"]:
        render_mode = "human"
    else:
        render_mode = ""

    #
    # Set up the random seeds for each episode. ** This will abort if there aren't enough seeds in the config file. **
    #
    seed_list = setup_seed_list(config)

    # -- 080225 9:15AM
    # Create the output folder
    """
    # -- 080425 1:49PM
    # Using the time in the folder name seems to be causing a problem for the pytest runs. Furthermore, we could have
    # multiple runs happening at the same time, so let's try using the process and thread ids to distinguish.  
    """
    new_folder_name = f"{config['simulator']['run_main_folder']}/{config['simulator']['run_sub_folder_prefix']}-started-{datetime.now().strftime('%Y%m%d-%H%M%S')}"

    new_run_folder = expand_path(new_folder_name, get_project_root())
    new_run_folder.mkdir(parents=True, exist_ok=True)
    # os.mkdir(new_run_folder)

    setup_logging(config, new_run_folder)

    # -- 080225 9:54AM
    # Copy the config file into the run folder to record configuration for the run.
    # There may be a better way to do this to make sure we get all parameters stored
    # in case there are still hardcoded values in the code -- which should be removed
    # at some point.
    copied_config_file_path = expand_path("config.yaml", new_run_folder)
    shutil.copy(config_filename, copied_config_file_path)

    # write out the package list to help with reproducibility (this seems to be especially important
    # for random number generators, which change based on package level.
    write_package_list(new_run_folder)

    # -- 080225
    # Find the path for the video in the run folder.
    video_file_path = expand_path(
        f"video.{config['visuals']['video_file_extension']}", new_run_folder
    )
    logger.debug(f"video path {video_file_path}")

    target_creation_time = config["simulator"]["target_creation_time"]
    """ 
    -- 080825 7:15PM
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
        run_trajectories=config["simulator"]["run_trajectories"]
        if "run_trajectories" in config["simulator"]
        else None,
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

    agents = agent_factory(agent_type=1, config=config, env=env)

    #
    # Run the episodes
    #
    for episode in tqdm(range(config["simulator"]["num_episodes"]), leave=True):
        # Start a new episode

        logger.debug(f"main(): starting episode {episode}")

        # Reset the environment
        obs, info = env.reset(
            seed=seed_list[episode], options=agents.get_reset_options()
        )

        # -- 080225 8:58AM
        # create the dataframe for the simulation output
        # df = pd.DataFrame(columns=pandas_columns)

        # -- 080725 10:45PM
        # Add the initial positions to the dataframe
        # df = add_obs_to_df(None, obs, time_step=0)
        variant_index_list, variant_type_list = agents.get_variant_types()
        df = add_obs_to_df(
            None,
            obs,
            time_step=0,
            variant_index_list=variant_index_list,
            variant_type_list=variant_type_list,
        )
        # done = False

        #
        # MAIN LOOP
        #

        for time_step in tqdm(range(config["simulator"]["num_frames"]), leave=False):
            agents.update(time_step=time_step)
            action = agents.get_action_list(obs)

            # Take the action in the environment and observe the result
            print("action\n", action)
            next_obs, reward, terminated, truncated, info = env.step(action)

            # -- 080225 8:58AM
            # Record the observation
            # df = add_obs_to_df(df, next_obs, time_step=(time_step + 1))
            variant_index_list, variant_type_list = agents.get_variant_types()
            df = add_obs_to_df(
                df,
                next_obs,
                time_step=(time_step + 1),
                variant_index_list=variant_index_list,
                variant_type_list=variant_type_list,
            )

            # Observe the next state
            obs = next_obs

            # ignore terminated for now since we are just running for a specified number of frames
            # done = terminated or truncated
            # done = True
            if terminated or truncated:
                break  # I hate breaks, why does Python make me do it?

        env.close()

        logger.info(f"episode {episode}: df columns = {df.columns}")
        # logger.info(f"positions:\n{df[['x', 'y', 'z']]}")
        # logger.info(f"velocities:\n{df[['v_x', 'v_y', 'v_z']]}")
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
        # print(f"writing output to {file_path}")
        pq.write_table(table, file_path)

        """
        -- 080825
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
        -- 081125 3:37PM
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
    print(f"output written to {new_run_folder}")


if __name__ == "__main__":
    #
    # Get the config file name if specified on the command line
    #
    parser = argparse.ArgumentParser(
        prog="run_simulator",
        description="Simulates agents in a 3D environment",
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

    run_simulator(config_filename=config_filename)

    print(f"run simulator completed at {datetime.now().strftime('%Y%m%d-%H%M%S')}")
