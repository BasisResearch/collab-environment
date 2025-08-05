from collections import defaultdict
from loguru import logger

import gymnasium as gym
import numpy as np


class BoidsWorldAgent:
    """ """

    """
    TOC -- 073125 
    Need to figure out what I am doing with speed. Are these going at max speed and then limited by max_force?  
    """

    def __init__(
        self,
        env: gym.Env,
        num_agents=1,
        min_ground_separation=3.0,
        min_separation=4.0,
        neighborhood_dist=20.0,
        ground_weight=1.0,
        separation_weight=3.0,
        alignment_weight=0.5,
        cohesion_weight=0.1,
        target_weight=0.0,
        max_speed=0.8,
        min_speed=0.1,
        max_force=0.1,  # maximum steering force
        walking=True,
        has_mesh_scene=True,
    ):
        """Initialize a Q-Learning agent.

        Args:
            env: The training environment
            learning_rate: How quickly to update Q-values (0-1)
            initial_epsilon: Starting exploration rate (usually 1.0)
            epsilon_decay: How much to reduce epsilon each episode
            final_epsilon: Minimum exploration rate (usually 0.1)
            discount_factor: How much to value future rewards (0-1)
        """
        self.env = env
        self.num_agents = num_agents
        self.min_ground_separation = min_ground_separation
        self.min_separation = min_separation
        self.neighborhood_dist = neighborhood_dist
        self.ground_weight = ground_weight
        self.separation_weight = separation_weight
        self.alignment_weight = alignment_weight
        self.cohesion_weight = cohesion_weight
        self.target_weight = target_weight
        self.max_speed = max_speed
        self.min_speed=min_speed
        self.max_force = max_force
        self.walking = walking
        self.env_has_mesh_scene = has_mesh_scene

        # Q-table: maps (state, action) to expected reward
        # default dict automatically creates entries with zeros for new states
        self.q_values = [
            defaultdict(lambda: np.zeros(env.action_space.n))  # type:ignore
            for _ in range(num_agents)
        ]

        # Track learning progress
        self.training_error = []  # type:ignore

    def set_target_weight(self, new_weight):
        self.target_weight = new_weight

    """
    Based on Dan Shiffman's Nature of Code. This needs to be optimized for use with torch.
    
    Assumes the obs['agent_loc'] contains pairs of locations and velocities. May need to adjust this
    to have two different entries in the observation dictionary, which probably makes more sense. In 
    that case, it would probably be better to get each list out of the dictionary to start instead
    of doing this as for each loops since we need them multiple times. Also need to figure out if all 
    of this can be done with matrix multiplications in torch. 
    
    Might be interesting to see what happens if we do this asynchronous DP style. Would that be more
    realistic of less or would it have so little effect as to be insignificant? Actually, looks like it is 
    better for performance to do this ADP style because we can reuse the velocity array.   
    
    Other pages of interest:
    https://vanhunteradams.com/Pico/Animal_Movement/Boids-algorithm.html 
    
    This function is way too long. 
    """

    def simpleBoidsAction(self, obs):
        #logger.debug(f"called with obs: {obs}")
        velocity = np.array(obs["agent_vel"])  # using ADP style
        location = np.array(obs["agent_loc"])
        for i in range(self.num_agents):
            """
            TOC -- 072325 -- 03:29PM 
            If we get too close to the ground, reverse direction. This will likely be too abrupt 
            and needs to be fixed.
            
            TOC -- 072925 -- 11:56AM 
            Add a turning factor to make this less abrupt instead of just turning directly around (vanhunteradams uses this)
            
            TOC -- 080425 -- 9:44PM
            Only do the ground separation if there is a scene to hit the ground on. This grond thins 
            needs a more intelligent solution.   
            """
            if (not self.walking) and self.env_has_mesh_scene and (
                obs["mesh_distance"][i] < self.min_ground_separation
            ):
                # velocity[i] = -velocity[i] + np.random.normal(0, 0.01, 3)# turn around abruptly but add some noise
                # velocity[i] = velocity[i] + np.array([0.0, -velocity[i][1], 0.0])
                velocity[i] = velocity[i] + np.random.normal(1, 0.01, 3) * 0.1
            else:
                # velocity[i] = vel # not sure which is faster, getting the whole thing before or walking through
                sum_separation_vector = np.zeros(3)
                sum_align_vector = np.zeros(3)
                sum_cohesion_vector = np.zeros(3)
                num_close = 0
                num_neighbors = 0
                ground_force = np.zeros(3)
                # I wonder if we could do this with j starting at i+1
                for other in range(self.num_agents):
                    if i != other:
                        # separate
                        dist = np.linalg.norm(location[i] - location[other])
                        if dist < self.min_separation:
                            diff_vector = location[i] - location[other]
                            sum_separation_vector += diff_vector / (dist**2)
                            num_close += 1

                        # align and cohesion
                        # TODO: Do this with numpy array condition and np.sum instead
                        # logger.debug(f"neighborhood dist: {self.neighborhood_dist}")
                        if dist < self.neighborhood_dist:
                            # need velocities in the observation
                            sum_align_vector += velocity[other]
                            sum_cohesion_vector += location[other]
                            num_neighbors += 1

                if num_close > 0:
                    # logger.debug(
                    #     f"simpleBoidsAction(): num close to {i} is {num_close}"
                    # )
                    avg_sep_vector = sum_separation_vector / num_close
                    # avg_sep_vector = avg_sep_vector / np.linalg.norm(avg_sep_vector) * self.max_speed
                else:
                    avg_sep_vector = np.zeros(3)

                """
                TOC -- 073125 -- 10:20AM 
                Interesting that I am using max_speed here but nowhere else. Perhaps I should set everyone to max_speed
                like Shiffman does. That didn't work, so I ditched it. Caused everyone to push to the walls -- not 
                entirely sure why.  
                """
                if num_neighbors > 0:
                    avg_align_vector = sum_align_vector / num_neighbors
                    # avg_align_vector = avg_align_vector / np.linalg.norm(avg_align_vector) * self.max_speed
                    avg_cohesion_vector = sum_cohesion_vector / num_neighbors
                    # avg_cohesion_vector = avg_cohesion_vector / np.linalg.norm(avg_cohesion_vector) * self.max_speed

                else:
                    avg_align_vector = np.zeros(3)
                    avg_cohesion_vector = np.zeros(3)

                # Reynold's steering formula is to subtract the current velocity from the desired
                # and limit it to some maximum force.

                total_force = np.zeros(3)
                align_force = np.zeros(3)
                separation_force = np.zeros(3)
                cohesion_force = np.zeros(3)
                target_force = np.zeros(3)

                # alignment

                """
                TOC -- 072925 -- 10:14AM 
                                    
                Take out the normalization, max_force will be applied to accumulated force later.  
                """
                if (num_neighbors > 0) and (self.alignment_weight > 0.0):
                    steer = avg_align_vector - velocity[i]
                    # align_force = steer / np.linalg.norm(steer) * self.max_force
                    align_force = steer
                    logger.debug(f"alignment force {align_force}")

                # cohesion
                if (num_neighbors > 0) and (self.cohesion_weight > 0.0):
                    """
                    TOC -- 072225 -- 12:54PM -- This was supposed to be location not velocity. 
                    
                    TOC -- 072925 -- 10:14AM 
                    Take out the normalization, max_force will be applied to accumulated force later. 
                    """
                    steer = avg_cohesion_vector - obs["agent_loc"][i]
                    # cohesion_force = steer / np.linalg.norm(steer) * self.max_force
                    cohesion_force = steer
                    logger.debug(f"cohesion force {cohesion_force}")

                # separation
                if (num_close > 0) and (self.separation_weight > 0.0):
                    separation_force = self.separation_weight * avg_sep_vector
                    logger.debug(f"separation {avg_sep_vector}")

                # target
                if self.target_weight > 0.0:
                    steer = obs["target_loc"] - obs["agent_loc"][i]
                    logger.debug("target_loc = " + str(obs["target_loc"]))
                    logger.debug("velocity " + str(velocity[i]))
                    logger.debug("steer " + str(steer))

                    """
                    TOC -- 072925 10:10AM
                    Change this to just use the full force computed to steer rather than pushing it 
                    to max_force, which doesn't make sense. We will limit to max_force when we 
                    accumulate all of the forces. This is different from the way Shiffman does it -- 
                    not sure if he is doing it that way for pedagogical reasons. 
                    
                    TOC -- 073125 8:58AM
                    I think Shiffman uses max_speed. We should probably treat the forces consistently and 
                    make everyone move at max_speed and then limit the total steering force. This seem to 
                    cause problems with everyone going to the walls, so took it out. 
                    """
                    # target_force = steer / np.linalg.norm(steer) * self.max_speed
                    target_force = steer
                    logger.debug(f"target force {target_force}")

                # ground
                if (
                    not self.walking
                    and obs["mesh_distance"][i] < self.min_ground_separation
                ):
                    logger.debug(f"mesh distance {obs['mesh_distance'][i]}")

                    ground_diff_vector = location[i] - obs["mesh_closest_points"][i]
                    # if np.linalg.norm(ground_diff_vector) < 0.00001:
                    #    ground_force = ground_diff_vector * self.max_force
                    # else:
                    """
                    TOC -- 072925 -- 10:06AM
                    Not sure why I was making all of these go at maximum force. Shiffman goes at max_speed 
                    but limits the total steering force. Try this without max_speed and do the maximum force
                    below. No, no, no, this is dividing by the square of the norm. The closer we get to the 
                    ground the faster we are trying to move away. 
                    """
                    # ground_force = ground_diff_vector / (np.linalg.norm(ground_diff_vector) ** 2) * self.max_force
                    ground_force = ground_diff_vector / (
                        np.linalg.norm(ground_diff_vector) ** 2
                    )
                    logger.debug(f"ground_diff_vector {ground_diff_vector}")
                    logger.debug(f"ground force {ground_force}")

                    # sum_separation_vector += diff_vector/(dist**2)

                total_force += (
                    self.alignment_weight * align_force
                    + self.cohesion_weight * cohesion_force
                    + self.target_weight * target_force
                    + self.separation_weight * separation_force
                    + self.ground_weight * ground_force
                )
                logger.debug(f'total force: {total_force}')

                """
                TOC -- 073125 9:00AM
                I am second guessing applying max force limit at the end rather than for each force individually. It
                may be that we want some of the forces not to be limited and so having an option for finer granularity 
                may be important. 
                """
                norm_total_force = np.linalg.norm(total_force)
                if norm_total_force > self.max_force:
                    total_force = total_force / norm_total_force * self.max_force
                    logger.debug(f'adjusted total force: {total_force}')


                # apply force
                if np.linalg.norm(total_force) > 0:
                    """
                    TOC -- 072225 10:29AM -- Why was this subtraction? Oops. That fixed a lot of problems.   
                    """
                    velocity[i] = total_force + velocity[i]

                    """
                    TOC -- 072226 10:41AM -- Adding this code caused the boids to freeze and convulse with separation 
                    and alignment in play. Increasing the max_speed from 0.1 to 0.5 seems to have fixed it. I couldn't 
                    tell if any boids flew off the screen, because the simulation moved too fast. This may have fixed 
                    the problem of flying off the screen at warp speed. No, flying off the screen at warp speed was
                    due to the fact that I was multiply the acceleration by 2 every time it touched a wall. 
                    """
                    # if np.linalg.norm(velocity[i]) > self.max_speed:
                    #    velocity[i] = velocity[i] / np.linalg.norm(velocity[i]) * self.max_speed

                norm_velocity = np.linalg.norm(velocity[i])
                if norm_velocity > self.max_speed:
                    velocity[i] = velocity[i] / norm_velocity * self.max_speed
                elif norm_velocity < self.min_speed:
                    velocity[i] = velocity[i] / norm_velocity * self.min_speed

        logger.debug("returning velocity: " + str(velocity))
        return velocity

    """
    Returns the velocity for each agent. 
    """

    def get_action(self, obs):
        return self.simpleBoidsAction(obs)

    def update(self, obs, action: int, reward: float, terminated: bool, next_obs):
        pass
        """
        This shouldn't need to be called for the simple model with no learning. If we add learning, 
        then this needs to be written. 
        :param obs: 
        :param action: 
        :param reward: 
        :param terminated: 
        :param next_obs: 
        :return: 
        """

        """Update Q-value based on experience.
        
        This is the heart of Q-learning: learn from (state, action, reward, next_state)
        """
        # What's the best we could do from the next state?
        # (Zero if episode terminated - no future rewards possible)
        # future_q_value = (not terminated) * np.max(self.q_distances(next_obs))

        # What should the Q-value be? (Bellman equation)
        # target = reward + self.discount_factor * future_q_value
        # logger.debug('target: ' + str(target))
        # How wrong was our current estimate?
        # temporal_difference = target - self.q_distances(obs, action)

        """
        TOC 
        Skipping this since are q_distances just reflect the true distance to the target. 
        """
        # Update our estimate in the direction of the error
        # Learning rate controls how big steps we take
        #  self.q_values[obs][action] = (
        #     self.q_values[obs][action] + self.lr * temporal_difference
        # )

        # Track learning progress (useful for debugging)
        # self.training_error.append(temporal_difference)
