"""
TOC
- 7/21/25 12:29 -- they seem to move together with all weights at 1. Need to bound them inside a cube and play with neighborhood sizes as some of
them seem to stop -- not sure why that is happening.
- 7/21/25 -- runs with 5 agents and a moving target but the agents scatter.
- works as a 2D grid
- 7/3/25 23:42 the open3D visualizer opens and runs while the environment is running.
"""

import argparse
from tqdm import tqdm  # Progress bar
import gymnasium as gym
import yaml
from loguru import logger

from collab_env.sim.boids.boidsAgents import BoidsWorldAgent
import collab_env.sim.gymnasium_env as gymnasium_env  # noqa: F401
from collab_env.data.file_utils import get_project_root, expand_path


# NUM_AGENTS = 40
# WALKING = False
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
    if config["simulator"]["show_visualizer"]:
        render_mode = "human"
    else:
        render_mode = None
    #
    # Create environment and agent
    #
    env = gym.make(
        "gymnasium_env/BoidsWorldSimple-v0",
        render_mode=render_mode,
        num_agents=config["simulator"]["num_agents"],
        walking=config["simulator"]["walking"],
        show_box=config["simulator"]["show_box"],
        box_size=config["environment"]["box_size"],
        scene_scale=config["environment"]["scene_scale"],
        scene_filename=config["files"]["mesh_scene"],
    )

    agent = BoidsWorldAgent(
        env=env,
        num_agents=config["simulator"]["num_agents"],
        walking=config["simulator"]["walking"],
        min_ground_separation=config["agent"]["min_ground_separation"],
        min_separation=config["agent"]["min_separation"],
        neighborhood_dist=config["agent"]["neighborhood_dist"],
        ground_weight=config["agent"]["ground_weight"],
        separation_weight=config["agent"]["separation_weight"],
        alignment_weight=config["agent"]["alignment_weight"],
        cohesion_weight=config["agent"]["cohesion_weight"],
        target_weight=config["agent"]["target_weight"],
        max_speed=config["agent"]["max_speed"],
        max_force=config["agent"]["max_force"],
    )

    #
    # Run the episodes
    #
    for episode in tqdm(range(config["simulator"]["num_episodes"])):
        # Start a new episode
        # print('main(): starting episode ' + str(episode))
        obs, info = env.reset()
        # print('main(): obs = ' + str(obs))
        done = False

        """
        TOC -- 073125 -- 8:46AM
        Need to decide on how this is going to end. 

        TOC -- 073124 2:21PM 
        We are going with a limited number of frames.
        """
        # logger.debug('starting main loop ')

        # while not done:
        for _ in tqdm(range(config["simulator"]["num_frames"])):
            # Agent chooses action
            action = agent.get_action(obs)

            # Take the action in the environment and observe the result
            next_obs, reward, terminated, truncated, info = env.step(action)

            # Observe the next state
            obs = next_obs

            # ignore terminated for now since we are just running for a soecified number of frames
            # done = terminated or truncated
            # done = True

    logger.info("all episodes complete")
