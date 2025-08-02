"""
TOC
- 7/21/25 12:29 -- they seem to move together with all weights at 1. Need to bound them inside a cube and play with neighborhood sizes as some of
them seem to stop -- not sure why that is happening.
- 7/21/25 -- runs with 5 agents and a moving target but the agents scatter.
- works as a 2D grid
- 7/3/25 23:42 the open3D visualizer opens and runs while the environment is running.
"""

from tqdm import tqdm  # Progress bar
import gymnasium as gym
import yaml
from loguru import logger

from collab_env.sim.boids.boidsAgents import BoidsWorldAgent
import collab_env.sim.gymnasium_env as gymnasium_env  # noqa: F401
from collab_env.data.file_utils import get_project_root, expand_path



#NUM_AGENTS = 40
#WALKING = False
if __name__ == "__main__":
    config_filename = expand_path(
        "collab_env/sim/boids/config.yaml", get_project_root()
    )
    config = yaml.safe_load(open(config_filename))

    """
    TOC -- 073125 3:00PM
    Need to include command line arguments 
    """


    # Create environment and agent
    env = gym.make(
        "gymnasium_env/BoidsWorldSimple-v0",
        render_mode="human",
        num_agents=config['simulator']['num_agents'],
        walking=config['simulator']['walking'],
        box_size=config['environment']['box_size'],
        scene_scale=config['environment']['scene_scale'],
        scene_filename=config['files']['mesh_scene'],
    )

    agent = BoidsWorldAgent(
        env=env,
        num_agents=config['simulator']['num_agents'],
        walking=config['simulator']['walking'],
        min_ground_separation=config['agent']['min_ground_separation'],
        min_separation=config['agent']['min_separation'],
        neighborhood_dist=config['agent']['neighborhood_dist'],
        ground_weight=config['agent']['ground_weight'],
        separation_weight=config['agent']['separation_weight'],
        alignment_weight=config['agent']['alignment_weight'],
        cohesion_weight=config['agent']['cohesion_weight'],
        target_weight=config['agent']['target_weight'],
        max_speed=config['agent']['max_speed'],
        max_force=config['agent']['max_force'],
    )

    for episode in tqdm(range(config['simulator']['num_episodes'])):
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
        for _ in tqdm(range(config['simulator']['num_frames'])):
            # Agent chooses action (initially random, gradually more intelligent)
            action = agent.get_action(obs)
            # print('action = ' + str(action))
            # Take action and observe result
            next_obs, reward, terminated, truncated, info = env.step(action)
            # logger.debug('terminated = ' + str(terminated))

            # Learn from this experience
            # agent.update(obs, action, reward, terminated, next_obs)

            # Move to next state
            # done = terminated or truncated
            obs = next_obs
            # done = True

        # Reduce exploration rate (agent becomes less random over time)
        # agent.decay_epsilon()

    logger.info("all episodes complete")
