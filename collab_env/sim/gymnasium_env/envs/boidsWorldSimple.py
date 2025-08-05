import gymnasium as gym
import torch
from gymnasium import spaces
import numpy as np
import open3d
import cv2

# from Boids.sim_utils import calc_angles
from loguru import logger

from collab_env.data.file_utils import get_project_root, expand_path

HIT_DISTANCE = 0.01
SPEED = 0.1


class BoidsWorldSimpleEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(
        self,
        render_mode=None,
        size=5,
        num_agents=1,
        num_targets=1,
        num_ground_targets=1,
        moving_targets=False,
        walking=False,
        target_scale=1.0,
        agent_shape="CONE",
        agent_scale=2.0,
        agent_color=[1, 0, 0],
        agent_mean_init_velocity=0.0,
        agent_variance_init_velocity=0.2,
        box_size=40,
        show_box=False,
        scene_scale=100.0,
        scene_filename="meshes/Open3dTSDFfusion_mesh.ply",
        scene_position=[20, 20, 20],
        scene_angle=[np.pi / 2.0, 0, 0],
        show_visualizer=True,
        store_video=False,
        video_file_path="video.mp4",
        video_codec="*mpv4",
        video_fps=30,
        vis_width=1920,
        vis_height=1027,
        target_creation_time=0,
    ):
        self.size = size  # The size of the square grid
        self.window_size = 512  # The size of the render window
        self.num_agents = num_agents
        self.num_targets = num_targets
        self.num_ground_targets = num_ground_targets
        self.moving_targets = moving_targets
        # keep track of the first index in the target list that is a ground target.
        self.ground_target_first_index = num_targets - num_ground_targets
        self._agent_location = None  # initialized by reset()
        self._agent_velocity = None  # initialized by reset()
        self._target_location = None  # initialized by reset()
        self._target_velocity = None  # initialized by reset()
        self._ground_target_location = None
        self._ground_target_velocity = None
        self.mesh_scene = None  # initialized by reset()
        self.max_dist_from_center = 3
        self.agent_shape = agent_shape.upper()
        self.agent_color = agent_color
        self.target_scale = target_scale
        self.action_scale = agent_scale
        self.agent_mean_init_velocity = agent_mean_init_velocity
        self.agent_variance_init_velocity = agent_variance_init_velocity
        self.box_size = box_size  # tne size of the cube boundary around the world
        self.show_box = show_box
        self.walking = walking
        self.target_creation_time = target_creation_time
        self.agent_scale = agent_scale
        self.scene_scale = scene_scale
        self.scene_filename = scene_filename
        self.scene_position = np.array(scene_position)
        self.scene_angle = np.array(scene_angle)
        self.show_visualizer = show_visualizer
        self.store_video = store_video
        self.video_file_path = video_file_path
        self.video_codec = video_codec
        self.video_fps = video_fps
        self.vis_width = vis_width
        self.vis_height = vis_height
        logger.debug(f"video path is {self.video_file_path}")
        logger.debug(f"store video is {self.store_video}")

        self.time_step = 0
        self.observation_space = spaces.Dict(
            {
                "agent_loc": spaces.Tuple(
                    [
                        spaces.Box(
                            low=-torch.inf, high=torch.inf, shape=(3,), dtype=np.float64
                        )
                        for _ in range(num_agents)
                    ]
                ),
                "agent_vel": spaces.Tuple(
                    [
                        spaces.Box(
                            low=-torch.inf, high=torch.inf, shape=(3,), dtype=np.float64
                        )
                        for _ in range(num_agents)
                    ]
                ),
                # TOC -- 080525 2:01PM
                # replaced this with list of targets.
                # "target_loc": spaces.Box(
                #     low=-torch.inf, high=torch.inf, shape=(3,), dtype=np.float64
                # ),
                "target_loc": spaces.Tuple(
                    [
                        spaces.Box(
                            low=-torch.inf, high=torch.inf, shape=(3,), dtype=np.float64
                        )
                        for _ in range(num_targets)
                    ]
                ),
                #
                # TOC -- 080525 2:06PM
                # updated to deal with multiple targets.
                #
                "distances_to_targets": spaces.Tuple(
                    [
                        spaces.Tuple(
                            [
                                spaces.Box(
                                    low=-torch.inf,
                                    high=torch.inf,
                                    shape=(1,),
                                    dtype=np.float64,
                                )
                                for _ in range(self.num_targets)
                            ]
                        )
                        for _ in range(self.num_agents)
                    ]
                ),
                "mesh_distance": spaces.Tuple(
                    [
                        spaces.Box(
                            low=-torch.inf, high=torch.inf, shape=(1,), dtype=np.float64
                        )
                        for _ in range(self.num_agents)
                    ]
                ),
                "mesh_closest_points": spaces.Tuple(
                    [
                        spaces.Box(
                            low=-torch.inf, high=torch.inf, shape=(3,), dtype=np.float64
                        )
                        for _ in range(self.num_agents)
                    ]
                ),
            }
        )

        logger.debug("obs space: " + str(self.observation_space))

        # actions are velocities
        self.action_space = spaces.Tuple(
            [
                spaces.Box(low=-torch.inf, high=torch.inf, shape=(3,), dtype=np.float64)
                for _ in range(num_agents)
            ]
        )

        """
        TOC -- 080125 1154PM 
        What is this? This must be from the original gymnasium example.
        Maybe this is standard Gymnasium stuff that we should stick to.  
        """
        # assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        """
        If human-rendering is used, `self.window` will be a reference to the window that we draw to. `self.clock` 
        will be a clock that is used to ensure that the environment is rendered at the correct frame rate in 
        human-mode. They will remain `None` until human-mode is used for the first time.
        """
        self.vis = None
        self.video_out = None
        self.geometry = None
        self.mesh_sphere_agent = None
        self.mesh_agent = None

    def get_action_array(self):
        return np.array(self._action_to_direction.values())

    def _get_obs(self):
        #
        # TOC -- 080425 7:50PM
        # distances and closest points should be set to some bogus value when there is no scene
        # we cannot set them to None, or we will fail the observation space checker in Gymnasium.
        # This is already done in compute_distance_and_closest_points() but it is better to do
        # it here rather than making the call for no reason.
        #
        if self.mesh_scene is None:
            distances = -np.ones(self.num_agents)
            closest_points = -np.ones(self.num_agents)
        else:
            distances, closest_points = self.compute_distance_and_closest_points(
                self.mesh_scene, self._agent_location
            )
        return {
            "agent_loc": tuple(self._agent_location),
            "agent_vel": tuple(self._agent_velocity),
            "target_loc": (tuple(self._target_location)),
            # "target_loc":
            "distances_to_targets": self.compute_target_distances(),
            "mesh_distance": distances,
            "mesh_closest_points": closest_points,
        }

    def _get_obs_no_target_list(self):
        #
        # TOC -- 080425 7:50PM
        # distances and closest points should be set to some bogus value when there is no scene
        # we cannot set them to None, or we will fail the observation space checker in Gymnasium.
        # This is already done in compute_distance_and_closest_points() but it is better to do
        # it here rather than making the call for no reason.
        #
        if self.mesh_scene is None:
            distances = -np.ones(self.num_agents)
            closest_points = -np.ones(self.num_agents)
        else:
            distances, closest_points = self.compute_distance_and_closest_points(
                self.mesh_scene, self._agent_location
            )
        return {
            "agent_loc": tuple(self._agent_location),
            "agent_vel": tuple(self._agent_velocity),
            "target_loc": (
                self._ground_target_location if self.walking else self._target_location
            ),
            # "target_loc":
            "distances_to_targets": self.compute_target_distances(),
            "mesh_distance": distances,
            "mesh_closest_points": closest_points,
        }

    def compute_target_distances(self):
        """
        TOC -- 071825
        need to make this faster
        might want to have a different target for each agent -- in fact, definitely do

        TOC -- 080425
        This needs to work with multiple targets and ground versus non ground. We could
        just have a set of indices that are ground targets or a set that are not since
        ground targets seem more likely. That way we only need one array. Not sure that is
        worth it.
        """
        # logger.debug('compute distances(): agent location' + str(self._agent_location))
        norms = tuple(
            np.array(
                [
                    [np.linalg.norm(a - t) for t in self._target_location]
                    for a in self._agent_location
                ],
                dtype="f",
            )
        )
        logger.debug(f"norms = {norms}")
        return norms

    def _get_info(self):
        """
        TOC -- 071625
        need to look at this for performance later.

        TOC -- 080325
        Took this out since it was unused.
        """

        # norms = [np.linalg.norm(a - self._target_location) for a in self._agent_location]
        # norms = self.compute_distances()

        return {"distance": None}

    def init_targets(self):
        """
        initialize list of targets with random positions
        Returns:

        """
        self._target_location = self.np_random.uniform(
            low=0.1 * self.box_size,
            high=0.9 * self.box_size,
            size=(self.num_targets, 3),
        )

        #
        # TOC -- 080525 3:54PM
        # I think this will work without the if, but better to be safe
        #
        if self.num_ground_targets > 0:
            self._target_location[self.ground_target_first_index :, 1] = 0.0

        # self._target_location = np.array(
        #    [self.box_size / 2, self.box_size / 2, self.box_size / 2]
        # )
        # self._ground_target_location = np.array(
        #    [self.box_size / 2.0, 0, self.box_size / 2]
        # )

        """
        TOC -- 071625
        Cut this out for now because we are dealing with multiple agents. Can add it back later in a 
        loop or something.
        """
        # while np.linalg.norm(self._target_location - self._agent_location) < 1:
        #    self._target_location = self.np_random.normal(0, self.size, size=3)

        """
        TOC -- 080325 
        This need to be configurable. No idea what I did with these while loops. 
        """
        # set the target velocity
        self._target_velocity = self.np_random.normal(
            0, 0.1, size=(self.num_targets, 3)
        )
        # while (
        #        np.linalg.norm(self._target_location - np.array([0.0, 0.0, 0.0])) < 0.000001
        # ):
        #   self._target_velocity = self.np_random.normal(0, 0.1, size=3)

        # self._ground_target_velocity = self.np_random.normal(0, 0.01, size=3)
        # while (
        #        np.linalg.norm(self._target_location - np.array([0.0, 0.0, 0.0])) < 0.000001
        # ):
        #    self._ground_target_velocity = self.np_random.normal(0, 0.1, size=3)

        """
        TOC -- 072125
        stop moving the target so we can debug the boids

        TOC -- 080325
        these need to be removed once velocities are configurable. 
        """
        # self._ground_target_velocity = np.zeros(3)
        # self._target_velocity = np.zeros((self.num_targets,3)

    def init_agents(self):
        if self.walking:
            self._agent_location = [
                np.array(
                    [
                        np.random.uniform(
                            low=0.1 * self.box_size, high=0.9 * self.box_size
                        ),
                        0.0,
                        np.random.uniform(
                            low=0.1 * self.box_size, high=0.9 * self.box_size
                        ),
                    ]
                )
                for _ in range(self.num_agents)
            ]

        else:
            self._agent_location = [
                self.np_random.uniform(
                    low=0.1 * self.box_size, high=0.9 * self.box_size, size=3
                )
                for _ in range(self.num_agents)
            ]
            logger.debug("agent loc:" + str(self._agent_location))

        """
        TOC -- 072325 
        We have to wait for rendering to be initialized for the mesh_scene to be created I think. 
        """
        if self.mesh_scene is not None and self.walking:
            distances, closest_points = self.compute_distance_and_closest_points(
                self.mesh_scene, self._agent_location
            )
            self._agent_location = closest_points.numpy()

        """
        TOC -- 073025 
        Need to replace hard coded numbers. 

        TOC -- 080325
        Replaced
        """
        self._agent_velocity = [
            self.np_random.normal(
                self.agent_mean_init_velocity, self.agent_variance_init_velocity, size=3
            )
            for _ in range(self.num_agents)
        ]
        """
        TOC -- 072125
        """
        # Set the initial velocities fast to debug the reverse
        # self._agent_velocity = np.array([np.array([2.0, 2.0, 2.0]) for _ in range(self.num_agents)])
        """
        TOC -- 073025 
        Get rid of all my logger.debug code since logger.debug doesn't work with gymnasium anyway. 
        """
        logger.debug("agent vel:" + str(self._agent_velocity))

    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)
        """
        TOC -- 071025 
        Switched these to fit the Euclidean space types that they should be 
        """
        # Choose the agent's location normally centered at 0
        # self._agent_location = [self.np_random.normal(0, self.size, size=3) for _ in range(self.num_agents)]
        # Choose uniformly inside the box
        """
        TOC -- 072525 
        if walking, start the agents off at height 0, otherwise start them anywhere. We could always start the flying
        boids on the ground, but we would need to change the initial placement code elsewhere, because the flying boids get 
        stuck under the mesh when I start them at 0 height.   
        
        TOC -- 080425
        These numbers need to be configurable. 
        
        TOC -- 080525 
        replace with function 
        """
        self.init_agents()
        # if self.walking:
        #     self._agent_location = [
        #         np.array(
        #             [
        #                 np.random.uniform(
        #                     low=0.1 * self.box_size, high=0.9 * self.box_size
        #                 ),
        #                 0.0,
        #                 np.random.uniform(
        #                     low=0.1 * self.box_size, high=0.9 * self.box_size
        #                 ),
        #             ]
        #         )
        #         for _ in range(self.num_agents)
        #     ]
        #
        # else:
        #     self._agent_location = [
        #         self.np_random.uniform(
        #             low=0.1 * self.box_size, high=0.9 * self.box_size, size=3
        #         )
        #         for _ in range(self.num_agents)
        #     ]
        #     logger.debug("agent loc:" + str(self._agent_location))
        #
        # """
        # TOC -- 072325
        # We have to wait for rendering to be initialized for the mesh_scene to be created I think.
        # """
        # if self.mesh_scene is not None and self.walking:
        #     distances, closest_points = self.compute_distance_and_closest_points(
        #         self.mesh_scene, self._agent_location
        #     )
        #     self._agent_location = closest_points.numpy()
        #
        # """
        # TOC -- 073025
        # Need to replace hard coded numbers.
        #
        # TOC -- 080325
        # Replaced
        # """
        # self._agent_velocity = [
        #     self.np_random.normal(
        #         self.agent_mean_init_velocity, self.agent_variance_init_velocity, size=3
        #     )
        #     for _ in range(self.num_agents)
        # ]
        # """
        # TOC -- 072125
        # """
        # # Set the initial velocities fast to debug the reverse
        # # self._agent_velocity = np.array([np.array([2.0, 2.0, 2.0]) for _ in range(self.num_agents)])
        # """
        # TOC -- 073025
        # Get rid of all my logger.debug code since logger.debug doesn't work with gymnasium anyway.
        # """
        # logger.debug("agent vel:" + str(self._agent_velocity))

        # integers(0, self.size, size=3, dtype=int)

        # We will sample the target's location randomly until it does not
        # coincide with the agent's location
        """
        TOC -- 071625
        not sure why I was starting this at the agent location
        
        TOC -- 080325 
        hard coded numbers need to be replaced. 
        
        TOC -- 080525 
        replace with function 
        """
        self.init_targets()
        # # self._target_location = self.np_random.uniform(low=0.1 * self.box_size, high=0.9 * self.box_size, size=3)
        # self._target_location = np.array(
        #     [self.box_size / 2, self.box_size / 2, self.box_size / 2]
        # )
        # self._ground_target_location = np.array(
        #     [self.box_size / 2.0, 0, self.box_size / 2]
        # )
        #
        # """
        # TOC -- 071625
        # Cut this out for now because we are dealing with multiple agents. Can add it back later in a
        # loop or something.
        # """
        # # while np.linalg.norm(self._target_location - self._agent_location) < 1:
        # #    self._target_location = self.np_random.normal(0, self.size, size=3)
        #
        # """
        # TOC -- 080325
        # This need to be configurable. No idea what I did with these while loops.
        # """
        # # set the target velocity
        # self._target_velocity = self.np_random.normal(0, 0.1, size=3)
        # while (
        #         np.linalg.norm(self._target_location - np.array([0.0, 0.0, 0.0])) < 0.000001
        # ):
        #     self._target_velocity = self.np_random.normal(0, 0.1, size=3)
        #
        # self._ground_target_velocity = self.np_random.normal(0, 0.01, size=3)
        # while (
        #         np.linalg.norm(self._target_location - np.array([0.0, 0.0, 0.0])) < 0.000001
        # ):
        #     self._ground_target_velocity = self.np_random.normal(0, 0.1, size=3)
        #
        # """
        # TOC -- 072125
        # stop moving the target so we can debug the boids
        #
        # TOC -- 080325
        # these need to be removed once velocities are configurable.
        # """
        # self._ground_target_velocity = np.zeros(3)
        # self._target_velocity = np.zeros(3)

        # self._target_velocity  = np.array([0.0, 0.0, 0.0])
        # logger.debug('reset(): target velocity: ' + str(self._target_velocity))

        # # Choose the agent's location uniformly at random
        # self._agent_location = self.np_random.integers(0, self.size, size=3, dtype=int)
        #
        # # We will sample the target's location randomly until it does not
        # # coincide with the agent's location
        # self._target_location = self._agent_location
        # while np.array_equal(self._target_location, self._agent_location):
        #     self._target_location = self.np_random.integers(
        #         0, self.size, size=3, dtype=int
        #     )

        observation = self._get_obs()
        info = self._get_info()

        """
        TOC -- 080225 8:08AM
        Why am I setting these to None after I called render_frame() above? 
        These need to be moved -- moved and it still works. 
        """
        self.vis = None
        self.geometry = None
        self.mesh_sphere_agent = None

        # TOC -- 080225 4:40PM
        # Always need to call _render_frame from reset because we rely on the meshes
        # for the simulation.
        # if self.render_mode == "human":
        self._render_frame()

        return observation, info

    def compute_direction(self, action_list):
        return [self._action_to_direction[a] for a in action_list]

    def convert_to_velocity_along_mesh(self, location, velocity):
        distances, closest_points = self.compute_distance_and_closest_points(
            self.mesh_scene, location + velocity
        )
        new_locations = closest_points.numpy()
        velocity = new_locations - location
        return velocity

    def update_target_locations(self):
        #
        # TOC -- 080425 8:00PM
        # If there is no mesh_scene, just move the ground target along the bottom of the box
        # by setting its velocity in the vertical direction to 0, but right now there is no
        # ground target if we are not walking. We can go back to this when we create multiple
        # targets. On second thought, we should have the ability to have both ground and air
        # targets, so this is the right way to do this. When we have multiple targets, this
        # should be an array of ground targets.
        #

        #
        # TOC -- 080525 3:18PM
        # make sure the ground targets will move to the ground
        #
        for i in range(self.num_ground_targets):
            if self.mesh_scene is not None:
                self._target_velocity[i + self.ground_target_first_index] = (
                    self.convert_to_velocity_along_mesh(
                        np.array(
                            [self._target_location[i + self.ground_target_first_index]]
                        ),
                        np.array(
                            [self._target_velocity[i + self.ground_target_first_index]]
                        ),
                    )[0]
                )
            else:
                self._target_velocity[i + self.ground_target_first_index][1] = 0.0
            """
            TOC -- 080425 2:33PM
            Need not do to this until the targets are created. Also need one target or a list instead of using ground. 
            """
            # self._ground_target_location = (
            #        self._ground_target_location + self._ground_target_velocity
            # )
            """
            TOC -- 080525 -- 3:16PM
            This should work as a numpy array instead of needing to do this is a for loop
            """
            self._target_location = self._target_location + self._target_velocity

    def step(self, action):
        self.time_step += 1
        if self.time_step == self.target_creation_time:
            self.create_targets()
        """
        Some of this code, like the wall avoidance, should be in action
        selection of the agent. Maybe not. 
        """
        # logger.debug('step(): called with action: ' + str(action))
        # Map the action (element of {0,1,2,3}) to the direction we walk in
        # self.direction = self._action_to_direction[action]
        # self.direction = self.compute_direction(action)
        # prev_distance = np.linalg.norm(self._agent_location - self._target_location, ord=1)
        # prev_distance = self.compute_distances()

        """
        TOC -- 071125
        Why am I clipping this? This was the problem. Clipping was from the original grid world code. 
        But when the target was on the negative side of the origin, it couldn't be found I guess -- still 
        doesn't seem like it should have gone really far away from the target though.  
        """
        # We use `np.clip` to make sure we don't leave the grid
        # self._agent_location = np.clip(
        #    self._agent_location + self.direction, 0, self.size - 1
        # )
        # logger.debug('before + agent location = ' + str(self._agent_location))
        """
        TOC -- 072125
        Need to fix all these types to make sure I consistently have ndarrays.
        
        needed to update the agent velocities to be the current action  
        """
        self._agent_velocity = action

        """
        TOC -- 072125 16:42 
        if outside the box, go backwards along the axis that hit but do so randomly so it looks less
        like a bounce. (Maybe I should do this for hitting the obstacles as well but obstacle avoidance
        should be done in the Agent class when it chooses an action. The environment can bounce the agents
        but they should control their own steering.)
        
        TOC -- 072925 
        Also need to bounce if the agents hit the mesh when not walking. Not sure what to do with walking 
        to make sure they don't move into the mesh -- this might be done naturally because we move to the 
        closest point on the mess but I am not sure if that code includes point inside the mesh.    
        """
        for i in range(self.num_agents):
            for coordinate in range(3):
                if (
                    self._agent_location[i][coordinate]
                    + self._agent_velocity[i][coordinate]
                    < 0
                ) or (
                    self._agent_location[i][coordinate]
                    + self._agent_velocity[i][coordinate]
                    > self.box_size
                ):
                    self._agent_velocity[i][coordinate] = (
                        np.random.normal(-0.8, 0.1, size=1)
                        * self._agent_velocity[i][coordinate]
                    )

        """
        TOC -- 072325 -- 0808PM 
        Take out the movement to check to see if initialization puts the agents on the mesh.
        Another function is moving these, which is bad. Only step should move the agents. 
        
        The design idea here, I guess, is that the agent chooses a desired velocity and the 
        environment determines where that desired velocity gets the agent, kind of like 
        Robin Williams telling the eggs to be free.   
        """
        if self.walking:
            self._agent_velocity = self.convert_to_velocity_along_mesh(
                self._agent_location, self._agent_velocity
            )
            # distances, closest_points = self.compute_distance_and_closest_points(self.mesh_scene,
            #                                                                     self._agent_location + self._agent_velocity)
            # new_locations = closest_points.numpy()
            # self._agent_velocity = new_locations - self._agent_location

        # update the agent's position based on its velocity
        self._agent_location = np.array(self._agent_location) + np.array(
            self._agent_velocity
        )

        # logger.debug('after + agent location = ' + str(self._agent_location))
        """
        TOC -- 072325
        Added this for debugging. All of the agents should be on the mesh after the update if walking is set.
        
        TOC -- 073125 7:32AM
        Took this out to see if rendering performance improves. Might have.   
        """
        # distances, closest_points = self.compute_distance_and_closest_points(self.mesh_scene, self._agent_location)

        # update the targets
        if self.moving_targets:
            self.update_target_locations()
        # #
        # # TOC -- 080425 8:00PM
        # # If there is no mesh_scene, just move the ground target along the bottom of the box
        # # by setting its velocity in the vertical direction to 0, but right now there is no
        # # ground target if we are not walking. We can go back to this when we create multiple
        # # targets. On second thought, we should have the ability to have both ground and air
        # # targets, so this is the right way to do this. When we have multiple targets, this
        # # should be an array of ground targets.
        # #
        # if self.mesh_scene is not None:
        #     self._ground_target_velocity = self.convert_to_velocity_along_mesh(
        #         np.array([self._ground_target_location]),
        #         np.array([self._ground_target_velocity]),
        #     )[0]
        # else:
        #     self._ground_target_velocity[1] = 0.0
        # """
        # TOC -- 080425 2:33PM
        # Need not do to this until the targets are created. Also need one target or a list instead of using ground.
        # """
        # self._ground_target_location = (
        #     self._ground_target_location + self._ground_target_velocity
        # )
        # self._target_location = self._target_location + self._target_velocity
        # logger.debug('step(): new location ' + str(self._agent_location))
        # An episode is done iff the agent has reached the target
        """
        TOC -- 071625
        needs distances to be a list 
        
        TOC -- 072825 
        not sure I am using distances anymore. Ah, I was treating the world to be constrained within a sphere but went 
        with the cube when the sphere calculations didn't work so well. Perhaps I should go back to the sphere approach
        since Matt suggested that is what he likes to do with dynamical systems involving particles running out to 
        infinity. Also, Reynolds indicated that having a target point to go to created unnatural behavior in boids.
        
        TOC -- 073125 2:51PM
        Looks like I am using this to determine whether anyone hit the target. Not sure this is necessary. I think
        we have this computed in the observation, but not sure we have that in time for checking the hit if we need
        to do that to determine whether the agent ate the food.  
        """
        # distance = np.linalg.norm(self._agent_location - self._target_location, ord=1)
        distance = np.array(self.compute_target_distances())

        # logger.debug('new distance:' + str(distance))
        """
        TOC -- 072125 12:41PM -- reverse direction if too far from the center -- maybe change this to add the target as a 
        part of the velocity calculation at some point.
        
        Skipping the distance approach and going with the box idea to see if that is better. 
        """
        # norms = np.apply_along_axis(np.linalg.norm, 1, self._agent_location)
        # indices = np.where(norms > self.max_dist_from_center)
        # print('step(): indices: ' + str(indices))
        # print('step(): norms: ' + str(norms))
        # print('step(): velocities : ' + str(self._agent_velocity))
        # reversed = False
        # for i in indices[0]:
        #     reversed = True
        #     # reverse direction
        #     print("step(): i = " + str(i))
        #     self._agent_velocity[i] = -self._agent_velocity[i]
        #     # added this to try to avoid going past the end point and bouncing back and forth -- didn't work
        #     self._agent_location[i] = self._agent_location[i] + self._agent_velocity[i]
        # print('step(): after reverse velocities : ' + str(self._agent_velocity))

        """
        TOC -- 072125 -- Check norms 
        """
        # if reversed:
        #     norms = np.apply_along_axis(np.linalg.norm, 1, self._agent_location)
        #     indices = np.where(norms > self.max_dist_from_center)
        #     logger.debug('step() indices ' + str(indices))
        #     logger.debug('step(): norms: ' + str(norms))
        #     logger.debug('step(): velocities : ' + str(self._agent_velocity))

        # logger.debug('distance condition thingy ' + str(distance[distance < HIT_DISTANCE]))
        """
        TOC -- 072925 
        I am setting terminated to be when an agent hits the goal. Somewhere, I must have set this to false. I should 
        set this to false here if that is what I want. Terminated should be controlled here and nowhere else. 
        runboids() is just ignoring terminated -- Never mind.  
        """
        """
        TOC -- 080325
        Hit distance should be configurable
        """
        terminated = (
            len(distance[distance < HIT_DISTANCE]) > 0
        )  # np.array_equal(self._agent_location, self._target_location)
        # logger.debug('terminated ' + str(terminated))
        reward = 0
        # reward = prev_distance - distance # rewarded for gaining ground
        # reward = 1 if terminated else 0  # Binary sparse rewards
        observation = self._get_obs()

        """
        TOC -- 073125 2:51PM
        This isn't used. Can probably speed things up by skipping this.
        """
        info = self._get_info()

        # TOC -- 080225 4:41PM
        # always want to call _render_frame because we rely on the meshes for the simulation
        # if self.render_mode == "human":
        self._render_frame()

        # logger.debug('step(): observation: ' + str(observation), level=1)
        # logger.debug('step(): reward: ' + str(reward), level=3)
        # logger.debug('step(): info: ' + str(info), level=4)
        return observation, reward, terminated, False, info

    """
    TOC -- 073125 2:30PM
    Why is this not getting called from reset? I guess that makes sense. I haven't played with this rgb_array thing 
    yet. I should probably do that to record video of each run without having to actually visualize it.  
    """

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def add_box(self):
        # self.scene = open3d.t.geometry.RaycastingScene()
        # _ = self.scene.add_triangles(self.mesh_sphere2)  # we do not need the geometry ID for mesh
        # self.rat.translate([-10, 1, 10])
        # self.vis.add_geometry(self.rat)

        #
        # Create all binary arrays of length 3 to represent the vertices on a cube.
        #
        bit_string_list = ["{0:03b}".format(n) for n in range(8)]
        logger.debug("bit_string: " + str(bit_string_list))
        bit_array = [list(b) for b in bit_string_list]

        logger.debug("bit_array: " + str(bit_array))
        points = [[self.box_size * int(b) for b in b_list] for b_list in bit_array]
        logger.debug("_render_frame(): points: " + str(points))

        #
        # This is the set of lines representing the edges of the cube
        #
        lines = [
            [0, 1],
            [0, 2],
            [1, 3],
            [2, 3],
            [4, 5],
            [4, 6],
            [5, 7],
            [6, 7],
            [0, 4],
            [1, 5],
            [2, 6],
            [3, 7],
        ]

        """
        TOC -- 080325
        This color should be configurable.
        """

        colors = [[1, 0, 0] for i in range(len(lines))]
        self.line_set = open3d.geometry.LineSet(
            points=open3d.utility.Vector3dVector(points),
            lines=open3d.utility.Vector2iVector(lines),
        )

        self.line_set.colors = open3d.utility.Vector3dVector(colors)
        self.vis.add_geometry(self.line_set)

        # q1 = copy.deepcopy(self.line_set)
        # q1.rotate(self.line_set.get_rotation_matrix_from_xyz((np.pi / 2, 0, 0)))
        #
        # # self.vis.add_geometry(q1)
        #
        # negative_points = [[-w for w in point_list] for point_list in points]
        # logger.debug('negative points: ' + str(negative_points))
        # negative_colors = [[0, 1, 0] for i in range(len(lines))]
        # self.negative_line_set = open3d.geometry.LineSet(
        #     points=open3d.utility.Vector3dVector(negative_points),
        #     lines=open3d.utility.Vector2iVector(lines),
        # )
        #
        # self.negative_line_set.colors = open3d.utility.Vector3dVector(negative_colors)
        # # self.vis.add_geometry(self.negative_line_set)

    def compute_distance_and_closest_points(self, splat_mesh, agents_loc):
        """
        TOC -- 072325
        This should be done more intelligently.
        """
        if splat_mesh is None:
            return np.zeros(self.num_agents), np.zeros(self.num_agents)

        """
        TOC -- 073125 
        This Raycasting setup shouldn't be redone everytime we need to calculate distances. 
        """
        mesh_scene = open3d.t.geometry.TriangleMesh.from_legacy(splat_mesh)

        # Create a scene and add the triangle mesh
        scene = open3d.t.geometry.RaycastingScene()
        _ = scene.add_triangles(mesh_scene)  # we do not need the geometry ID for mesh

        query_points = open3d.core.Tensor(agents_loc, dtype=open3d.core.Dtype.Float32)

        """ 
        TOC -- 073125 
        Do we really need both signed and unsigned distances? Also, do we need to call
        both compute_distance() and compute_closest_points()?  
        """
        # Compute distance of the query point from the surface
        unsigned_distance = scene.compute_distance(query_points)
        signed_distance = scene.compute_signed_distance(query_points)
        occupancy = scene.compute_occupancy(query_points)

        closest_points_dict = scene.compute_closest_points(query_points)

        logger.debug(f"unsigned distance {unsigned_distance.numpy()}")
        logger.debug(f"signed_distance {signed_distance.numpy()}")
        logger.debug(f"occupancy {occupancy.numpy()}")
        return unsigned_distance.numpy(), closest_points_dict["points"]

    def move_agent_meshes(self):
        # Reset to home position
        for i in range(self.num_agents):
            """
            TOC -- 072625
            We have the acceleration in the BoidsAgent where it chooses actions. That is what we 
            need I think to compute the orientation of the agent meshes. Or not.
            """

            """
            TOC -- 072525
            I believe this is necessary because the rotations are not relative. Oh, but if the rotations
            are not relative, then there is no need to keep track of the previous velocity. We can just 
            rotate to the new velocity from [0,0,1], which is apparently the default direction. So I don't
            know why this is here or if we need it.   
            """
            # self.mesh_arrow_agent[i].transform(np.eye(4))

            """
            TOC -- 072725 -- 11:28AM
            Rotations not working yet. Fix that in open3DTests project before incorporating
            """
            #
            # # Increment the rotation angle by 1 degree in radians
            # #total_rotation_angle_y += np.radians(.2)
            # angle_x, angle_y, angle_z = calc_angles(np.array([0,0,1]), self._agent_velocity[i])
            # # Create a rotation matrix for the new total rotation angle
            # R = self.mesh_arrow_agent[i].get_rotation_matrix_from_xyz((-angle_x, angle_y, angle_z))
            # # R = mesh_arrow_1.get_rotation_matrix_from_xyz((angle_x, 0, 0))
            #
            # # Apply the rotation
            # self.mesh_arrow_agent[i].rotate(R, center=self.mesh_arrow_agent[i].get_center())
            self.mesh_agent[i].translate(np.array(self._agent_velocity[i]))
            self.vis.update_geometry(self.mesh_agent[i])

    def move_target_meshes(self):
        for i in range(self.num_targets):
            self.mesh_target[i].translate(np.array(self._target_velocity[i]))
            self.vis.update_geometry(self.mesh_target[i])

    def load_rotate_scene(
        self, filename, position=np.array([0.0, 0.0, 0.0]), angles=(-np.pi / 2, 0, 0)
    ):
        # self.mesh_scene = open3d.io.read_triangle_mesh("example_meshes/example_mesh.ply")
        mesh_scene = open3d.io.read_triangle_mesh(filename)
        logger.debug(f"tommy's mesh {mesh_scene}")
        logger.debug(f"Center of mesh: {mesh_scene.get_center()}")

        """
        TOC -- 072325 
        Took this from the example code from Open3D but this wasn't necessary. 
        """
        # self.mesh_scene_for_distance = open3d.t.geometry.TriangleMesh.from_legacy(self.mesh_scene)

        """
        TOC -- 072325 -- Test distance calculation to splat mesh. 
        """
        # self.compute_distance_to_splat_mesh(self.mesh_scene)

        """
        TOC -- 073025 -- 8:21PM
        Not sure why this is translating to this particular position -- needed that for first
        splat but I need to generalize. 
        Why am I scaling this centered at the target location instead of the center of the scene?
        This is a bug.  
        """
        #
        # TOC -- 080325 1:19PM
        # The new meshes seem to come in centered at about the origin once they are rotated 90
        # degrees, but the mesh ground is a little below the bottom of the box. We could
        # translate the box, but I sort of like having the box bottom and sides to be 0 -- though
        # I guess we could translate the box to match the mesh rather than the other way around
        # and then use the box positions in the bounce off the wall conditions. Maybe let's just
        # make the mesh position configurable. I presume different meshes that people load may
        # have different properties, and it would be nice to allow the position of the mesh to
        # be configurable. We should translate and rotate it once though. Not sure what was up
        # with the double translation I had in here.

        mesh_scene.translate(position)
        # TOC -- 080225 8:30AM
        # This scaling should be happening centered at the center of the mesh_scene
        # rather than the target location. Seems to work.
        #
        # mesh_scene.scale(scale=self.scene_scale, center=self._target_location)
        mesh_scene.scale(scale=self.scene_scale, center=mesh_scene.get_center())

        """
        TOC -- 072725 -- 11:37AM

        This is specific to the first of Tommy's splats. This needs to be generalized to reorient 
        whatever scene mesh we are given to match the ground.  
        """

        """
        TOC -- 073025 -- 7:45PM
        Put all translation and rotation of the scene here
        
        Tommy's first mesh came in at an odd angle. The new meshes seem to be rotated 90 degrees.
        """
        # mesh_scene.translate(-mesh_scene.get_center() + np.array([self.box_size / 2.0, 0.0, self.box_size / 2.0]))
        # mesh_scene.translate(-mesh_scene.get_center() + np.array([self.box_size / 2.0, -self.box_size / 2.0, 0.0]))
        # mesh_scene.translate(position - mesh_scene.get_center())
        #    np.array([self.box_size / 2.0 + 0.25, self.box_size / 2.0 - 1, self.box_size / 2]))

        # R = mesh_scene.get_rotation_matrix_from_xyz((-np.pi / 18, 0, -np.pi / 1.6))
        R = mesh_scene.get_rotation_matrix_from_xyz(angles)

        # Apply the rotation
        """
        TOC -- 073025
        Need to make sure this is the correct center for rotation. Seems like it should 
        be the center of the mesh scene after it has been translated. 
        """
        # mesh_scene.rotate(R, center=(self.box_size / 2.0, 0, 0))
        mesh_scene.rotate(R, center=mesh_scene.get_center())
        # R = np.array([ [0.1, 0, 0], [0,0,0], [0,0,0] ])
        # self.mesh_scene.rotate(R, center=self._target_location)
        return mesh_scene

    def initialize_visualizer(self):
        """
        Returns:
        """
        """ 
        TOC -- 073125 
        Is this the right way to visualize. Kind of think there may have been a newer way to do this.

        TOC -- 080125 10:59PM
        Do we need to create the window if we aren't visualizing? Need to investigate
        our ability to compute things about the scene mesh separately from the 
        visualizer. The simulation seems to run when we don't create the window. Need 
        to check on other used of the visualizer in the code. 
        """

        # Initialize Open3D visualizer
        self.vis = open3d.visualization.Visualizer()
        self.vis.create_window(
            width=self.vis_width, height=self.vis_height, visible=self.show_visualizer
        )

        if self.store_video:
            #
            # Create VideoWriter object.
            #
            # TOC -- 080325 12:13PM
            # This file type should be configurable.
            #
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            self.video_out = cv2.VideoWriter(
                str(self.video_file_path),
                fourcc,
                self.video_fps,
                (self.vis_width, self.vis_height),
            )
            logger.debug(f"video file {self.video_file_path}")

    def init_agent_meshes(self):
        #
        # TOC -- 072325
        # Move all agents to the closest point on the mesh
        #
        if self.walking:
            distances, closest_points = self.compute_distance_and_closest_points(
                self.mesh_scene, self._agent_location
            )
            self._agent_location = closest_points.numpy()

        # self.mesh_sphere_agent = [None] * self.num_agents
        self.mesh_agent = [None] * self.num_agents
        for i in range(self.num_agents):
            """
            TOC -- 080225 1:57PM 
            Have a config option to make this a sphere instead of a cone. 
            """
            if self.agent_shape == "SPHERE":
                self.mesh_agent[i] = open3d.geometry.TriangleMesh.create_sphere(
                    radius=0.4
                )

            elif self.agent_shape == "BUNNY":
                bunny = open3d.data.BunnyMesh()
                self.mesh_agent[i] = open3d.io.read_triangle_mesh(bunny.path)
            else:  # default to cone
                self.mesh_agent[i] = open3d.geometry.TriangleMesh.create_arrow(
                    cone_height=0.8,
                    cone_radius=0.4,
                    cylinder_radius=0.4,
                    cylinder_height=1.0,
                )

            self.mesh_agent[i].compute_vertex_normals()
            self.mesh_agent[i].paint_uniform_color(self.agent_color)
            self.mesh_agent[i].translate(self._agent_location[i])
            self.mesh_agent[i].scale(
                scale=self.agent_scale, center=self._agent_location[i]
            )
            self.vis.update_geometry(self.mesh_agent[i])

            self.vis.add_geometry(self.mesh_agent[i])

    def create_targets(self):
        """
        This method simply adds the target meshes that were already created to the visualizer.
        Returns:

        """

        """
        TOC -- 080525
        Right now all targets will be created at the same time. This needs to be changed 
        to have configurable times for each target. 
        """
        for i in range(self.num_targets):
            self.vis.add_geometry(self.mesh_target[i])
        # if self.walking:
        #     self.vis.add_geometry(self.mesh_ground_target[i])
        # else:
        #     self.vis.add_geometry(self.mesh_target)

    def init_target_meshes(self):
        self.mesh_target = [None] * self.num_targets
        for i in range(self.num_targets):
            self.mesh_target[i] = open3d.geometry.TriangleMesh.create_sphere(radius=1.0)
            self.mesh_target[i].compute_vertex_normals()
            self.mesh_target[i].paint_uniform_color([0.1, 0.6, 0.1])
            self.mesh_target[i].scale(
                scale=self.target_scale, center=self.mesh_target[i].get_center()
            )

            # self.mesh_ground_target = open3d.geometry.TriangleMesh.create_sphere(radius=1.0)
            # self.mesh_ground_target.compute_vertex_normals()
            # self.mesh_ground_target.paint_uniform_color([0.1, 0.6, 1.0])
            # self.mesh_ground_target.scale(
            #     scale=self.target_scale, center=self.mesh_ground_target.get_center()
            # )
            """
            TOC -- 072325 
            If agents are walking, put the target on the mesh scene. Screws up the scale 
            somehow. It must not be drawn to the correct spot. Try not drawing it.   
            """
            if i >= self.ground_target_first_index and self.mesh_scene is not None:
                distances, closest_points = self.compute_distance_and_closest_points(
                    self.mesh_scene,
                    [self._target_location[i]],
                )
                # self.mesh_sphere_target.translate(closest_points.numpy()[0] - self._target_location)
                self._target_location[i] = closest_points.numpy()[0]

            """
            TOC -- 073125 -- 7:30AM 
            Need to decide on how we are dealing with ground and non-ground targets. The initial non-ground
            target was to keep boids away from the walls, but that doesn't make sense from a food source 
            point of view. Each of the targets is going to have to have some weight (possibly changing)
            associated with it. 
            """
            self.mesh_target[i].translate(self._target_location[i])
            # self.mesh_ground_target.translate(self._ground_target_location)
            self.vis.update_geometry(self.mesh_target[i])
            # self.vis.update_geometry(self.mesh_ground_target)

    def init_meshes(self):
        """
        TOC -- 073125
        Scene may need to be read in reset() since it is needed even if not rendering.
        """
        # self.mesh_scene = self
        # .load_rotate_scene("example_meshes/example_mesh.ply", position=np.array([self.box_size/2.0, 0.0, self.box_size/2.0]), angles=(-np.pi / 18, 0, -np.pi / 1.6)
        if self.scene_filename == "":
            self.mesh_scene = None
            if self.walking:
                logger.warning(
                    "Walking requires a scene to walk on. Walking will be turned off."
                )
                self.walking = False
        else:
            filename = expand_path(self.scene_filename, get_project_root())
            self.mesh_scene = self.load_rotate_scene(
                filename,
                # position=np.array([self.box_size / 2.0, -self.box_size / 2.0, 8.0]),
                position=self.scene_position,
                # angles=(-np.pi / 2, 0, 0),
                angles=self.scene_angle,
            )

        if self.show_box:
            self.add_box()

        self.init_agent_meshes()

        self.init_target_meshes()

        # self.mesh_target = open3d.geometry.TriangleMesh.create_sphere(radius=1.0)
        # self.mesh_target.compute_vertex_normals()
        # self.mesh_target.paint_uniform_color([0.1, 0.6, 0.1])
        # self.mesh_target.scale(
        #     scale=self.target_scale, center=self.mesh_target.get_center()
        # )
        #
        # self.mesh_ground_target = open3d.geometry.TriangleMesh.create_sphere(radius=1.0)
        # self.mesh_ground_target.compute_vertex_normals()
        # self.mesh_ground_target.paint_uniform_color([0.1, 0.6, 1.0])
        # self.mesh_ground_target.scale(
        #     scale=self.target_scale, center=self.mesh_ground_target.get_center()
        # )
        # """
        # TOC -- 072325
        # If agents are walking, put the target on the mesh scene. Screws up the scale
        # somehow. It must not be drawn to the correct spot. Try not drawing it.
        # """
        # if self.walking:
        #     distances, closest_points = self.compute_distance_and_closest_points(
        #         self.mesh_scene, [self._ground_target_location]
        #     )
        #     # self.mesh_sphere_target.translate(closest_points.numpy()[0] - self._target_location)
        #     self._ground_target_location = closest_points.numpy()[0]
        #
        # """
        # TOC -- 073125 -- 7:30AM
        # Need to decide on how we are dealing with ground and non-ground targets. The initial non-ground
        # target was to keep boids away from the walls, but that doesn't make sense from a food source
        # point of view. Each of the targets is going to have to have some weight (possibly changing)
        # associated with it.
        # """
        # self.mesh_target.translate(self._target_location)
        # self.mesh_ground_target.translate(self._ground_target_location)
        # self.vis.update_geometry(self.mesh_target)
        # self.vis.update_geometry(self.mesh_ground_target)

        """
        TOC -- 080225 2:56PM
        Some of these meshes are not used and need to be removed.
        """
        self.mesh_sphere_world1 = open3d.geometry.TriangleMesh.create_sphere(radius=0.1)
        self.mesh_sphere_world1.compute_vertex_normals()
        self.mesh_sphere_world1.paint_uniform_color([0.0, 0.0, 0.0])
        self.mesh_sphere_world1.translate([0.0, 0.0, self.max_dist_from_center])

        self.mesh_sphere_center = open3d.geometry.TriangleMesh.create_sphere(radius=0.1)
        self.mesh_sphere_center.compute_vertex_normals()
        self.mesh_sphere_center.paint_uniform_color([1.0, 0.0, 0.0])
        self.mesh_sphere_center.translate([0.0, 0.0, 0.0])

        # self.mesh_sphere_start = open3d.geometry.TriangleMesh.create_sphere(radius=0.1)
        # self.mesh_sphere_start.compute_vertex_normals()
        # self.mesh_sphere_start.paint_uniform_color([0.6, 0.1, 0.1])
        # self.mesh_sphere_start.translate(self._target_location + [2, 2, 2])

        """
        TOC -- 073125 -- 7:22AM
        Move all the add geometries into one spot. What is this top corner thing? (That 
        was for debugging the problem with all the boids going to the corner.)
        Do I want to put all of the adds for the agents in this spot too? That requires
        extra for loop. I think we might because then we can skip all of this if we are
        not visualizing.   
        """
        if self.mesh_scene is not None:
            self.vis.add_geometry(self.mesh_scene)

        # self.vis.add_geometry(self.mesh_top_corner)

        self.vis.add_geometry(self.mesh_sphere_world1)
        self.vis.add_geometry(self.mesh_sphere_center)
        # self.vis.add_geometry(self.mesh_sphere_start)

    def _render_frame(self):
        """
        :return:
        """

        """
        TOC -- 073125
        Since we depend on the scene mesh for calculating distances to agents in the simulation, the scene mesh at
        least needs to be initialized regardless of whether the render mode is human or not. So this code needs to 
        change. Perhaps the scene should be initialized in reset() or init() instead. reset() makes sense if the 
        scene could change based on some user defined configuration between trials. 
        
        TOC -- 080225 8:35AM
        Lots of work needs to be done here to deal with render mode. We need to make sure we 
        can calculate the distances to the mesh when we are not using the viewer and that 
        everything gets initialized correctly. There is lots of garbage code in here. 
        """

        if self.vis is None:
            self.initialize_visualizer()
            self.init_meshes()

        self.move_agent_meshes()

        # for i in range(self.num_agents):
        #     ''' '''
        #     # logger.debug('render velocity ' + str(self._agent_velocity[i]), level=1)
        #     # self.mesh_sphere_agent[i].translate(np.array(self._agent_velocity[i]))
        #     # this has to be velocity because it moves by this amount not to this position apparently
        #     # logger.debug('render() velocity = ' + str(self._agent_velocity[i]))
        #     ''' TOC -- 072325 -- 0813PM
        #     This is a problem. render needs to translate the mesh for the agent but the movement
        #     should be controlled by the step function. I guess I should update the velocity to 0
        #     when I mess with the agents or calculate the distances between then new location at the
        #     mesh to see how much it needs to translate. That might be the best option. For now, take
        #     the translate out to test that the agents get put on the mesh to start. This may also be
        #     the problem with the agents flying away when they should be walking along the ground. It
        #     seems like only render should be dealing with meshes, but maybe not since the mesh really
        #     tells us the positions. In any event, we can grab the location of the mesh and take the
        #     difference between the _agent_location and the mesh location as the amount we need to
        #     translate. No we can't. We need to update the velocity so render know how to translate
        #     mesh. This makes sense anyway, since the velocities should be accurate. We definitely will
        #     have a problem with agents falling off of trees and what not. Maybe we could do a little
        #     binary search for the point closest to where we want to go that is still on the mesh. Seems
        #     like someone should have figured this out already -- maybe Isabel will know.
        #     '''
        #     self.mesh_arrow_agent[i].translate(np.array(self._agent_velocity[i]))
        #     '''
        #     TOC --
        #     Rotation nonsense that doesn't work.
        #     '''

        """
        TOC -- 080225 2:14PM 
        Why am I not translating the ground target by the ground target velocity?
        """
        if self.moving_targets:
            self.move_target_meshes()
        # self.mesh_target.translate(self._target_velocity)

        # TOC -- 080525
        # this is done in move_agent_meshes()
        # for i in range(self.num_agents):
        #     # self.vis.update_geometry(self.mesh_sphere_agent[i])
        #     self.vis.update_geometry(self.mesh_agent[i])
        #     # logger.debug("render(): mesh center " + str(self.mesh_arrow_agent[i].get_center()))
        #     # logger.debug("render(): agent location " + str(self._agent_location[i]))
        #

        # self.vis.update_geometry(self.mesh_ground_target)
        self.vis.poll_events()
        self.vis.update_renderer()

        if self.store_video:
            # Capture video
            img = self.vis.capture_screen_float_buffer()
            # logger.debug(f'img is {img}')
            img = (255 * np.asarray(img)).astype(np.uint8)
            # logger.debug(f'after astype {img}')
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            # logger.debug(f'after cvtColor {img}')

            # Write to video
            self.video_out.write(img)
            # logger.debug(f'video out {self.video_out} result = {result}')

    def close(self):
        """ """

        """
        TOC6 -- 080125 
        Added this as part of Issue #13 to clean up the simulator. The 
        visualization window should close properly with this fix. 
        """
        if self.vis is not None:
            self.vis.destroy_window()

        if self.video_out is not None:
            self.video_out.release()
            assert False
