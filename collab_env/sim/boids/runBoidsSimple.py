"""
TOC
- 7/21/25 12:29 -- they seem to move together with all weights at 1. Need to bound them inside a cube and play with neighborhood sizes as some of
them seem to stop -- not sure why that is happening.
- 7/21/25 -- runs with 5 agents and a moving target but the agents scatter.
- works as a 2D grid
- 7/3/25 23:42 the open3D visualizer opens and runs while the environment is running.
"""

import gymnasium as gym
from boidsAgents import BoidsWorldAgent

from tqdm import tqdm  # Progress bar

# from logger.debug import logger.debug


NUM_AGENTS = 40
WALKING = False
if __name__ == "__main__":
    """
    TOC -- 073125 3:00PM
    Need to include command line arguments 
    """
    # Training hyperparameters
    learning_rate = (
        0.01  # How fast to learn (higher = faster but less stable) -- not used
    )
    n_episodes = 1  # Number of boids runs.
    start_epsilon = 0.0  # Start with 100% random actions -- not used
    epsilon_decay = start_epsilon / (
        n_episodes / 2
    )  # Reduce exploration over time == not sued
    final_epsilon = 0.0  # Always keep some exploration -- not used

    speed = 0.01

    # Create environment and agent
    env = gym.make(
        "gymnasium_env/BoidsWorldSimple-v0",
        render_mode="human",
        num_agents=NUM_AGENTS,
        walking=WALKING,
    )

    agent = BoidsWorldAgent(
        env=env,
        action_to_direction=None,
        learning_rate=learning_rate,
        initial_epsilon=start_epsilon,
        epsilon_decay=epsilon_decay,
        final_epsilon=final_epsilon,
        num_agents=NUM_AGENTS,
        walking=WALKING,
    )

    for episode in tqdm(range(n_episodes)):
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
        for _ in tqdm(range(25000)):
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

    print("all episodes complete")
