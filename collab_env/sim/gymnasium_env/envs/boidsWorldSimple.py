import gymnasium as gym
import torch
from gymnasium import spaces
import numpy as np
import open3d
import cv2

# from Boids.sim_utils import calc_angles
from loguru import logger

from collab_env.data.file_utils import get_project_root, expand_path
from collab_env.sim.boids.sim_utils import get_submesh_indices_from_ply

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
        agent_init_range_low=0.1,
        agent_init_range_high=0.9,
        agent_height_range_low=0.0,
        agent_height_range_high=1.0,
        agent_height_init_min=0,
        agent_height_init_max=1000,
        target_init_range_low=0.1,
        target_init_range_high=0.9,
        target_height_init_max=40,
        target_positions=None,
        target_mesh_file=None,
        target_mesh_color=[0, 1, 0],
        target_mesh_init_color=[1, 0, 1],
        box_size=40,
        show_box=False,
        scene_scale=100.0,
        scene_filename="meshes/Open3dTSDFfusion_mesh.ply",
        scene_position=[20, 20, 20],
        scene_angle=[np.pi / 2.0, 0, 0],
        show_visualizer=True,
        store_video=False,
        video_file_path=None,
        saved_image_path=None,
        video_codec="*mpv4",
        video_fps=30,
        vis_width=1920,
        vis_height=1027,
        target_creation_time=0,
        agent_trajectories=None,
        target_trajectories=None,
        run_trajectories=False,  # run trajectories is for actually moving the agent meshes
        show_trajectory_lines=False,  # show the trajectory lines for the agent
        save_image=False,
        color_tracks_by_time=False,
        number_track_color_groups=1,
        track_color_rate=4,
    ):
        if target_mesh_color is None:
            target_mesh_color = [1, 0, 1]
        self.box_line_set = None
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
        self.specified_target_position = target_positions
        self._target_location = None  # initialized by reset()
        self._target_velocity = None  # initialized by reset()
        self._ground_target_location = None
        self._ground_target_velocity = None
        self.target_mesh_file = target_mesh_file
        self.submesh_init_color = target_mesh_init_color
        self.submesh_color = target_mesh_color
        self.mesh_scene = None  # initialized by reset()
        self.max_dist_from_center = 3
        self.agent_shape = agent_shape.upper()
        self.agent_color = agent_color
        self.target_scale = target_scale
        self.action_scale = agent_scale
        self.agent_mean_init_velocity = agent_mean_init_velocity
        self.agent_variance_init_velocity = agent_variance_init_velocity
        self.agent_init_range_low = agent_init_range_low
        self.agent_init_range_high = agent_init_range_high
        self.agent_height_range_low = agent_height_range_low
        self.agent_height_range_high = agent_height_range_high
        self.agent_height_init_min = agent_height_init_min
        self.agent_height_init_max = agent_height_init_max
        self.target_init_range_low = target_init_range_low
        self.target_init_range_high = target_init_range_high
        self.target_height_init_max = target_height_init_max
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
        self.mesh_target_list = [None] * self.num_targets
        self.agent_trajectories = agent_trajectories
        self.target_trajectories = target_trajectories
        self.run_trajectories = run_trajectories
        self.show_trajectory_lines = show_trajectory_lines
        self.save_image = save_image
        self.saved_image_path = saved_image_path
        self.color_tracks_by_time = color_tracks_by_time
        self.number_track_color_groups = number_track_color_groups
        self.track_color_rate = track_color_rate

        self.trajectory_line_set = None

        self.time_step = 0
        self.image_count = 0
        self.terminated = False
        self.truncated = False

        """
        TOC -- 081225 3:58PM
        If we have a target mesh file to read in, we need to get the vertices labeled as the 
        target from the file and rotate and translate them to their corresponding positions
        on the scene mesh. That was silly. We get the indices by calling get_submesh_indices_from_ply(). 
        Then we don't have to translate and rotate anything because the indicies stay the same. We can
        then just get the vertices out of the scene vertices by choosing by index after we load and rotate
        the scene in load_and_rotate()  
        """
        self.submesh_vertices = None
        self.submesh_target = False
        if self.target_mesh_file is not None:
            self.submesh_target = True
            self.submesh_vertex_indices = get_submesh_indices_from_ply(
                expand_path(self.target_mesh_file, get_project_root())
            )
            logger.info(f"{len(self.submesh_vertex_indices)} found in sub-mesh target")

        self.observation_space = spaces.Dict(
            {
                "agent_loc": spaces.Tuple(
                    [
                        spaces.Box(
                            low=-torch.inf, high=torch.inf, shape=(3,), dtype=np.float32
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
                "distances_to_target_centers": spaces.Tuple(
                    [
                        spaces.Tuple(
                            [
                                spaces.Box(
                                    low=-torch.inf,
                                    high=torch.inf,
                                    shape=(1,),
                                    dtype=np.float32,
                                )
                                for _ in range(self.num_targets)
                            ]
                        )
                        for _ in range(self.num_agents)
                    ]
                ),
                "distances_to_target_mesh_closest_points": spaces.Tuple(
                    [
                        spaces.Tuple(
                            [
                                spaces.Box(
                                    low=-torch.inf,
                                    high=torch.inf,
                                    shape=(1,),
                                    dtype=np.float32,
                                )
                                for _ in range(self.num_targets)
                            ]
                        )
                        for _ in range(self.num_agents)
                    ]
                ),
                "target_mesh_closest_points": spaces.Tuple(
                    [
                        spaces.Tuple(
                            [
                                spaces.Box(
                                    low=-torch.inf,
                                    high=torch.inf,
                                    shape=(3,),
                                    dtype=np.float32,
                                )
                                for _ in range(self.num_targets)
                            ]
                        )
                        for _ in range(self.num_agents)
                    ]
                ),
                "mesh_scene_distance": spaces.Tuple(
                    [
                        spaces.Box(
                            low=-torch.inf, high=torch.inf, shape=(1,), dtype=np.float64
                        )
                        for _ in range(self.num_agents)
                    ]
                ),
                "mesh_scene_closest_points": spaces.Tuple(
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
        Maybe this is standard Gymnasium stuff that we should stick to, but take
        it out for now.  
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
        # TOC -- 080625 9:03PM
        # Modifying to move to the closest points on the target meshes rather than the
        # center of the target meshes. This requires a change to the observation space
        # to include the closest points between agents and targets. This will be an array
        # of shape (num_agents, num_targets) with each entry being a box of length 3.
        # TOC -- 080425 7:50PM
        # distances and closest points should be set to some bogus value when there is no scene
        # we cannot set them to None, or we will fail the observation space checker in Gymnasium.
        # This is already done in compute_distance_and_closest_points() but it is better to do
        # it here rather than making the call for no reason.
        #
        if self.mesh_scene is None:
            scene_distances = -np.ones(self.num_agents)
            scene_closest_points = -np.ones(self.num_agents)
        else:
            scene_distances, scene_closest_points = (
                self.compute_distance_and_closest_points(
                    self.mesh_scene, self._agent_location
                )
            )
        if self.submesh_target:
            distances_to_target_mesh, target_mesh_closest_points = (
                self.compute_target_distances_to_vertices()
            )
        else:
            distances_to_target_mesh, target_mesh_closest_points = (
                self.compute_mesh_target_distances()
            )
        return {
            "agent_loc": tuple(self._agent_location),
            "agent_vel": tuple(self._agent_velocity),
            "target_loc": (tuple(self._target_location)),
            # "target_loc":
            "distances_to_target_centers": self.compute_target_center_distances(),
            "distances_to_target_mesh_closest_points": distances_to_target_mesh,  # modified 080625 9:08PM
            "target_mesh_closest_points": target_mesh_closest_points,  # modified 080625 9:08PM
            "mesh_scene_distance": scene_distances,
            "mesh_scene_closest_points": scene_closest_points,
        }

    def _get_obs_to_target_centers(self):
        """ """

        """
        #
        # TOC -- 080425 7:50PM
        # distances and closest points should be set to some bogus value when there is no scene.
        # We cannot set them to None, or we will fail the observation space checker in Gymnasium.
        # This is already done in compute_distance_and_closest_points() but it is better to do
        # it here rather than making the call for no reason.
        #
        """

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
            "distances_to_targets": self.compute_target_center_distances(),
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
            "distances_to_targets": self.compute_target_center_distances(),
            "mesh_distance": distances,
            "mesh_closest_points": closest_points,
        }

    def get_distances(agent_locations, submesh_vertices):
        distances = np.zeros(len(agent_locations))
        closest_points = np.zeros((len(agent_locations), 3))
        for i in range(len(agent_locations)):
            diff = submesh_vertices - agent_locations[i]
            norms = np.apply_along_axis(np.linalg.norm, 1, diff)
            argmin = np.argmin(norms)
            closest_points[i] = submesh_vertices[argmin]
            distances[i] = norms[argmin]
        return distances, closest_points

    def compute_target_distances_to_vertices(self):
        """
        Computes the distances from each agent to each vertex on the submesh

        For now, I am assuming there is only one target. This needs to be fixed.

        Returns:

        """

        d, c = self.compute_distance_and_closest_points_to_vertices(
            self.submesh_vertices, self._agent_location
        )
        distances = np.array([d])
        closest_points = np.array([c])
        distances = np.array(distances).transpose()

        logger.debug(f"dist {distances}")
        logger.debug(f"dist shape {distances.shape}")
        closest_points = np.array(closest_points).transpose((1, 0, 2))
        logger.debug(f"points {closest_points}")
        logger.debug(f"points shape {closest_points.shape}")

        return distances, closest_points

    def compute_mesh_target_distances(self):
        """
        Computes the distances from each agent to each target mesh using open3D
        Returns:

        """
        logger.debug(f"len{len(self.mesh_target_list)}")
        distances = []
        closest_points = []
        #
        # TOC -- 080625 10:27PM
        # There should be a more efficient way to do this without a for loop, but
        # I don't have time to worry about it right now
        #
        for t in self.mesh_target_list:
            d, c = self.compute_distance_and_closest_points(t, self._agent_location)
            distances.append(d)
            closest_points.append(c)

        distances = np.array(distances).transpose()
        logger.debug(f"dist {distances}")
        logger.debug(f"dist shape {distances.shape}")
        closest_points = np.array(closest_points).transpose((1, 0, 2))
        logger.debug(f"points {closest_points}")
        logger.debug(f"points shape {closest_points.shape}")

        # distance_point_matrix = np.array(the_list, dtype=np.float32)

        # switch target and agent axes but leave (distance,closest point) pair alone
        # result = distance_point_matrix.transpose((1,0,2))
        #
        # distances = result[:, :, 0]
        # closest_points =  result[:, :, 1]

        logger.debug(f"distances =\n{distances}")
        logger.debug(f"distances shape = {distances.shape}")
        logger.debug(f"closest_points =\n{closest_points}")
        logger.debug(f"closest_points shape = {closest_points.shape}")

        return distances, closest_points

    def compute_target_center_distances(self):
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
        norms = np.array(
            # np.array(
            [
                [np.linalg.norm(a - t) for t in self._target_location]
                for a in self._agent_location
            ],
            dtype="f",
            # )
        )
        logger.debug(f"norms =\n{norms}")
        logger.debug(f"norms shape = {norms.shape}")

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
        Returns:
        """
        if self.run_trajectories:
            """ 
            TOC -- 080825 11:21AM
            Fix this after figure out agent trajectoies. Actually, we may never need this since we may never do
            moving targets.  
            """
            if self.target_trajectories is not None:
                self._target_location = self.target_trajectories[0]
            else:
                self._target_location = np.zeros((self.num_targets, 3))
        else:
            """
            TOC -- 081125 7:00PM
            If we do not have a fixed target location, initialize list of targets with random positions
            We know we have a fixed location for a target if the user specified a non-empty list for the target.  
            """
            # self._target_location = self.np_random.uniform(
            #     low=0.1 * self.box_size,
            #     high=0.9 * self.box_size,
            #     size=(self.num_targets, 3),
            # )
            if self.specified_target_position is not None:
                self._target_location = np.zeros((self.num_targets, 3))
                for i in range(self.num_targets):
                    if len(self.specified_target_position[i]) > 0:
                        self._target_location[i] = np.array(
                            self.specified_target_position[i]
                        )
                    else:
                        self._target_location[i] = np.array(
                            [
                                self.np_random.uniform(
                                    low=self.target_init_range_low * self.box_size,
                                    high=self.target_init_range_high * self.box_size,
                                ),
                                self.np_random.uniform(
                                    low=self.target_init_range_low
                                    * self.target_height_init_max,
                                    high=self.target_init_range_high
                                    * self.target_height_init_max,
                                ),
                                self.np_random.uniform(
                                    low=self.target_init_range_low * self.box_size,
                                    high=self.target_init_range_high * self.box_size,
                                ),
                            ]
                        )

            else:
                self._target_location = np.array(
                    [
                        np.array(
                            [
                                self.np_random.uniform(
                                    low=self.target_init_range_low * self.box_size,
                                    high=self.target_init_range_high * self.box_size,
                                ),
                                self.np_random.uniform(
                                    low=self.target_init_range_low
                                    * self.target_height_init_max,
                                    high=self.target_init_range_high
                                    * self.target_height_init_max,
                                ),
                                self.np_random.uniform(
                                    low=self.target_init_range_low * self.box_size,
                                    high=self.target_init_range_high * self.box_size,
                                ),
                            ]
                        )
                        for _ in range(self.num_targets)
                    ]
                )
            # print(self._target_location.shape)
            # assert(False)

            #
            # TOC -- 080525 3:54PM
            # I think this will work without the if, but better to be safe
            #
            if self.num_ground_targets > 0:
                self._target_location[self.ground_target_first_index :, 1] = 0.0

            """
            TOC -- 080325 
            This need to be configurable.
            """
            # set the target velocity
            self._target_velocity = self.np_random.normal(
                0, 0.1, size=(self.num_targets, 3)
            )

    def init_agents(self):
        if self.show_trajectory_lines or self.run_trajectories:
            # use the first time step in the trajectory to
            # initialize the agent positions.
            self._agent_location = self.agent_trajectories[:, 0]
            self._agent_velocity = np.zeros(self.num_agents)
        else:
            if self.walking:
                self._agent_location = [
                    np.array(
                        [
                            self.np_random.uniform(
                                low=self.agent_init_range_low * self.box_size,
                                high=self.agent_init_range_high * self.box_size,
                            ),
                            0.0,
                            self.np_random.uniform(
                                low=self.agent_init_range_low * self.box_size,
                                high=self.agent_init_range_high * self.box_size,
                            ),
                        ]
                    )
                    for _ in range(self.num_agents)
                ]

            else:
                self._agent_location = [
                    np.array(
                        [
                            self.np_random.uniform(
                                low=self.agent_init_range_low * self.box_size,
                                high=self.agent_init_range_high * self.box_size,
                            ),
                            self.np_random.uniform(
                                low=self.agent_height_range_low
                                * (
                                    self.agent_height_init_max
                                    - self.agent_height_init_min
                                )
                                + self.agent_height_init_min,
                                high=self.agent_height_range_high
                                * (
                                    self.agent_height_init_max
                                    - self.agent_height_init_min
                                )
                                + self.agent_height_init_min,
                            ),
                            self.np_random.uniform(
                                low=self.agent_init_range_low * self.box_size,
                                high=self.agent_init_range_high * self.box_size,
                            ),
                        ]
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
                self._agent_location = closest_points

            """
            TOC -- 073025 
            Need to replace hard coded numbers. 
    
            TOC -- 080325
            Replaced
            """
            self._agent_velocity = [
                self.np_random.normal(
                    self.agent_mean_init_velocity,
                    self.agent_variance_init_velocity,
                    size=3,
                )
                for _ in range(self.num_agents)
            ]

            logger.debug("agent vel:" + str(self._agent_velocity))

    def create_trajectory_line_group(self, agent_index, points, time_range, color):
        # print(points[time_range[-1]])
        # assert(False)
        """
        Args:
            agent_index:
            points:
            time_range:
            color:

        Returns:

        """

        """
        TOC -- 081225 9:17AM
        Need to look into the fact that there seems to be an extra set of points all at 0 at the end of the
        list of points. 
        """
        lines = [[t, t + 1] for t in range(len(points) - 1)]
        line_set = open3d.geometry.LineSet(
            points=open3d.utility.Vector3dVector(points),
            lines=open3d.utility.Vector2iVector(lines),
        )

        """
        TOC -- 080825 12:05PM 
        Make these colors configurable.

        TOC -- 081125 3:18PM
        The colors need to change through time, so I think I am going to have
        to split these up into groups of lines sets based on time step and 
        color the groups. 
        """
        # colors = [
        #     [
        #         (agent_index % 2) * 0.5,
        #         agent_index / self.num_agents,
        #         1 - (agent_index / self.num_agents),
        #     ]
        #     for i in range(len(lines))
        # ]
        color_list = [color] * len(lines)

        line_set.colors = open3d.utility.Vector3dVector(color_list)
        self.vis.add_geometry(line_set)
        self.trajectory_line_set.append(line_set)

    """
    TOC -- 082025 9:26PM 
    There are better, probably more efficient, ways to do this turbo colormap. Fix 
    this after the paper is submitted. 
    """

    def init_trajectory_lines(self):
        self.trajectory_line_set = []
        color_list = []
        # blue = (0, 0, 1.0)
        # cyan = (0, 1.0, 1.0)
        # green = (0, 1.0, 0)
        # yellow = (1.0, 1.0, 0)
        # red = (1.0, 0, 0)
        # turbo color list from:
        # https://gist.github.com/mikhailov-work/ee72ba4191942acecc03fe6da94fc73f
        # should also take the interpolater COPYRIGHT 2019 Google
        turbo_colormap_data = [
            [0.18995, 0.07176, 0.23217],
            [0.19483, 0.08339, 0.26149],
            [0.19956, 0.09498, 0.29024],
            [0.20415, 0.10652, 0.31844],
            [0.20860, 0.11802, 0.34607],
            [0.21291, 0.12947, 0.37314],
            [0.21708, 0.14087, 0.39964],
            [0.22111, 0.15223, 0.42558],
            [0.22500, 0.16354, 0.45096],
            [0.22875, 0.17481, 0.47578],
            [0.23236, 0.18603, 0.50004],
            [0.23582, 0.19720, 0.52373],
            [0.23915, 0.20833, 0.54686],
            [0.24234, 0.21941, 0.56942],
            [0.24539, 0.23044, 0.59142],
            [0.24830, 0.24143, 0.61286],
            [0.25107, 0.25237, 0.63374],
            [0.25369, 0.26327, 0.65406],
            [0.25618, 0.27412, 0.67381],
            [0.25853, 0.28492, 0.69300],
            [0.26074, 0.29568, 0.71162],
            [0.26280, 0.30639, 0.72968],
            [0.26473, 0.31706, 0.74718],
            [0.26652, 0.32768, 0.76412],
            [0.26816, 0.33825, 0.78050],
            [0.26967, 0.34878, 0.79631],
            [0.27103, 0.35926, 0.81156],
            [0.27226, 0.36970, 0.82624],
            [0.27334, 0.38008, 0.84037],
            [0.27429, 0.39043, 0.85393],
            [0.27509, 0.40072, 0.86692],
            [0.27576, 0.41097, 0.87936],
            [0.27628, 0.42118, 0.89123],
            [0.27667, 0.43134, 0.90254],
            [0.27691, 0.44145, 0.91328],
            [0.27701, 0.45152, 0.92347],
            [0.27698, 0.46153, 0.93309],
            [0.27680, 0.47151, 0.94214],
            [0.27648, 0.48144, 0.95064],
            [0.27603, 0.49132, 0.95857],
            [0.27543, 0.50115, 0.96594],
            [0.27469, 0.51094, 0.97275],
            [0.27381, 0.52069, 0.97899],
            [0.27273, 0.53040, 0.98461],
            [0.27106, 0.54015, 0.98930],
            [0.26878, 0.54995, 0.99303],
            [0.26592, 0.55979, 0.99583],
            [0.26252, 0.56967, 0.99773],
            [0.25862, 0.57958, 0.99876],
            [0.25425, 0.58950, 0.99896],
            [0.24946, 0.59943, 0.99835],
            [0.24427, 0.60937, 0.99697],
            [0.23874, 0.61931, 0.99485],
            [0.23288, 0.62923, 0.99202],
            [0.22676, 0.63913, 0.98851],
            [0.22039, 0.64901, 0.98436],
            [0.21382, 0.65886, 0.97959],
            [0.20708, 0.66866, 0.97423],
            [0.20021, 0.67842, 0.96833],
            [0.19326, 0.68812, 0.96190],
            [0.18625, 0.69775, 0.95498],
            [0.17923, 0.70732, 0.94761],
            [0.17223, 0.71680, 0.93981],
            [0.16529, 0.72620, 0.93161],
            [0.15844, 0.73551, 0.92305],
            [0.15173, 0.74472, 0.91416],
            [0.14519, 0.75381, 0.90496],
            [0.13886, 0.76279, 0.89550],
            [0.13278, 0.77165, 0.88580],
            [0.12698, 0.78037, 0.87590],
            [0.12151, 0.78896, 0.86581],
            [0.11639, 0.79740, 0.85559],
            [0.11167, 0.80569, 0.84525],
            [0.10738, 0.81381, 0.83484],
            [0.10357, 0.82177, 0.82437],
            [0.10026, 0.82955, 0.81389],
            [0.09750, 0.83714, 0.80342],
            [0.09532, 0.84455, 0.79299],
            [0.09377, 0.85175, 0.78264],
            [0.09287, 0.85875, 0.77240],
            [0.09267, 0.86554, 0.76230],
            [0.09320, 0.87211, 0.75237],
            [0.09451, 0.87844, 0.74265],
            [0.09662, 0.88454, 0.73316],
            [0.09958, 0.89040, 0.72393],
            [0.10342, 0.89600, 0.71500],
            [0.10815, 0.90142, 0.70599],
            [0.11374, 0.90673, 0.69651],
            [0.12014, 0.91193, 0.68660],
            [0.12733, 0.91701, 0.67627],
            [0.13526, 0.92197, 0.66556],
            [0.14391, 0.92680, 0.65448],
            [0.15323, 0.93151, 0.64308],
            [0.16319, 0.93609, 0.63137],
            [0.17377, 0.94053, 0.61938],
            [0.18491, 0.94484, 0.60713],
            [0.19659, 0.94901, 0.59466],
            [0.20877, 0.95304, 0.58199],
            [0.22142, 0.95692, 0.56914],
            [0.23449, 0.96065, 0.55614],
            [0.24797, 0.96423, 0.54303],
            [0.26180, 0.96765, 0.52981],
            [0.27597, 0.97092, 0.51653],
            [0.29042, 0.97403, 0.50321],
            [0.30513, 0.97697, 0.48987],
            [0.32006, 0.97974, 0.47654],
            [0.33517, 0.98234, 0.46325],
            [0.35043, 0.98477, 0.45002],
            [0.36581, 0.98702, 0.43688],
            [0.38127, 0.98909, 0.42386],
            [0.39678, 0.99098, 0.41098],
            [0.41229, 0.99268, 0.39826],
            [0.42778, 0.99419, 0.38575],
            [0.44321, 0.99551, 0.37345],
            [0.45854, 0.99663, 0.36140],
            [0.47375, 0.99755, 0.34963],
            [0.48879, 0.99828, 0.33816],
            [0.50362, 0.99879, 0.32701],
            [0.51822, 0.99910, 0.31622],
            [0.53255, 0.99919, 0.30581],
            [0.54658, 0.99907, 0.29581],
            [0.56026, 0.99873, 0.28623],
            [0.57357, 0.99817, 0.27712],
            [0.58646, 0.99739, 0.26849],
            [0.59891, 0.99638, 0.26038],
            [0.61088, 0.99514, 0.25280],
            [0.62233, 0.99366, 0.24579],
            [0.63323, 0.99195, 0.23937],
            [0.64362, 0.98999, 0.23356],
            [0.65394, 0.98775, 0.22835],
            [0.66428, 0.98524, 0.22370],
            [0.67462, 0.98246, 0.21960],
            [0.68494, 0.97941, 0.21602],
            [0.69525, 0.97610, 0.21294],
            [0.70553, 0.97255, 0.21032],
            [0.71577, 0.96875, 0.20815],
            [0.72596, 0.96470, 0.20640],
            [0.73610, 0.96043, 0.20504],
            [0.74617, 0.95593, 0.20406],
            [0.75617, 0.95121, 0.20343],
            [0.76608, 0.94627, 0.20311],
            [0.77591, 0.94113, 0.20310],
            [0.78563, 0.93579, 0.20336],
            [0.79524, 0.93025, 0.20386],
            [0.80473, 0.92452, 0.20459],
            [0.81410, 0.91861, 0.20552],
            [0.82333, 0.91253, 0.20663],
            [0.83241, 0.90627, 0.20788],
            [0.84133, 0.89986, 0.20926],
            [0.85010, 0.89328, 0.21074],
            [0.85868, 0.88655, 0.21230],
            [0.86709, 0.87968, 0.21391],
            [0.87530, 0.87267, 0.21555],
            [0.88331, 0.86553, 0.21719],
            [0.89112, 0.85826, 0.21880],
            [0.89870, 0.85087, 0.22038],
            [0.90605, 0.84337, 0.22188],
            [0.91317, 0.83576, 0.22328],
            [0.92004, 0.82806, 0.22456],
            [0.92666, 0.82025, 0.22570],
            [0.93301, 0.81236, 0.22667],
            [0.93909, 0.80439, 0.22744],
            [0.94489, 0.79634, 0.22800],
            [0.95039, 0.78823, 0.22831],
            [0.95560, 0.78005, 0.22836],
            [0.96049, 0.77181, 0.22811],
            [0.96507, 0.76352, 0.22754],
            [0.96931, 0.75519, 0.22663],
            [0.97323, 0.74682, 0.22536],
            [0.97679, 0.73842, 0.22369],
            [0.98000, 0.73000, 0.22161],
            [0.98289, 0.72140, 0.21918],
            [0.98549, 0.71250, 0.21650],
            [0.98781, 0.70330, 0.21358],
            [0.98986, 0.69382, 0.21043],
            [0.99163, 0.68408, 0.20706],
            [0.99314, 0.67408, 0.20348],
            [0.99438, 0.66386, 0.19971],
            [0.99535, 0.65341, 0.19577],
            [0.99607, 0.64277, 0.19165],
            [0.99654, 0.63193, 0.18738],
            [0.99675, 0.62093, 0.18297],
            [0.99672, 0.60977, 0.17842],
            [0.99644, 0.59846, 0.17376],
            [0.99593, 0.58703, 0.16899],
            [0.99517, 0.57549, 0.16412],
            [0.99419, 0.56386, 0.15918],
            [0.99297, 0.55214, 0.15417],
            [0.99153, 0.54036, 0.14910],
            [0.98987, 0.52854, 0.14398],
            [0.98799, 0.51667, 0.13883],
            [0.98590, 0.50479, 0.13367],
            [0.98360, 0.49291, 0.12849],
            [0.98108, 0.48104, 0.12332],
            [0.97837, 0.46920, 0.11817],
            [0.97545, 0.45740, 0.11305],
            [0.97234, 0.44565, 0.10797],
            [0.96904, 0.43399, 0.10294],
            [0.96555, 0.42241, 0.09798],
            [0.96187, 0.41093, 0.09310],
            [0.95801, 0.39958, 0.08831],
            [0.95398, 0.38836, 0.08362],
            [0.94977, 0.37729, 0.07905],
            [0.94538, 0.36638, 0.07461],
            [0.94084, 0.35566, 0.07031],
            [0.93612, 0.34513, 0.06616],
            [0.93125, 0.33482, 0.06218],
            [0.92623, 0.32473, 0.05837],
            [0.92105, 0.31489, 0.05475],
            [0.91572, 0.30530, 0.05134],
            [0.91024, 0.29599, 0.04814],
            [0.90463, 0.28696, 0.04516],
            [0.89888, 0.27824, 0.04243],
            [0.89298, 0.26981, 0.03993],
            [0.88691, 0.26152, 0.03753],
            [0.88066, 0.25334, 0.03521],
            [0.87422, 0.24526, 0.03297],
            [0.86760, 0.23730, 0.03082],
            [0.86079, 0.22945, 0.02875],
            [0.85380, 0.22170, 0.02677],
            [0.84662, 0.21407, 0.02487],
            [0.83926, 0.20654, 0.02305],
            [0.83172, 0.19912, 0.02131],
            [0.82399, 0.19182, 0.01966],
            [0.81608, 0.18462, 0.01809],
            [0.80799, 0.17753, 0.01660],
            [0.79971, 0.17055, 0.01520],
            [0.79125, 0.16368, 0.01387],
            [0.78260, 0.15693, 0.01264],
            [0.77377, 0.15028, 0.01148],
            [0.76476, 0.14374, 0.01041],
            [0.75556, 0.13731, 0.00942],
            [0.74617, 0.13098, 0.00851],
            [0.73661, 0.12477, 0.00769],
            [0.72686, 0.11867, 0.00695],
            [0.71692, 0.11268, 0.00629],
            [0.70680, 0.10680, 0.00571],
            [0.69650, 0.10102, 0.00522],
            [0.68602, 0.09536, 0.00481],
            [0.67535, 0.08980, 0.00449],
            [0.66449, 0.08436, 0.00424],
            [0.65345, 0.07902, 0.00408],
            [0.64223, 0.07380, 0.00401],
            [0.63082, 0.06868, 0.00401],
            [0.61923, 0.06367, 0.00410],
            [0.60746, 0.05878, 0.00427],
            [0.59550, 0.05399, 0.00453],
            [0.58336, 0.04931, 0.00486],
            [0.57103, 0.04474, 0.00529],
            [0.55852, 0.04028, 0.00579],
            [0.54583, 0.03593, 0.00638],
            [0.53295, 0.03169, 0.00705],
            [0.51989, 0.02756, 0.00780],
            [0.50664, 0.02354, 0.00863],
            [0.49321, 0.01963, 0.00955],
            [0.47960, 0.01583, 0.01055],
        ]
        # [1.0, 1.0, 1.0]] # I think the last one is skipped by not sure
        # color_transition_pairs = [
        #     [turbo_colormap_data[0], turbo_colormap_data[1]],
        #     [turbo_colormap_data[1], blue],
        #     [blue, cyan],
        #     [cyan, green],
        #     [green, yellow],
        #     [yellow, red],
        # ]
        color_transition_pairs = list(
            zip(
                turbo_colormap_data[: len(turbo_colormap_data) - 1],
                turbo_colormap_data[1:],
            )
        )
        for agent_index in range(self.num_agents):
            points = self.agent_trajectories[agent_index]
            logger.debug(f"number of points = {len(points)}")
            if self.color_tracks_by_time:
                group_size = int(np.ceil(len(points) / self.number_track_color_groups))
                logger.debug(f"group size = {group_size}")
                steps_per_color_pair = np.ceil(
                    self.number_track_color_groups / len(color_transition_pairs)
                )
                logger.debug(f"steps per color pair = {steps_per_color_pair}")
                # steps_per_color = self.number_track_color_groups / (len(turbo_colormap_data)-1)
                logger.debug(f"group size = {group_size}")
                logger.debug(f"steps per color = {steps_per_color_pair}")
                step = 0
                color_transition_index = 0
                for time in range(0, len(points), group_size):
                    logger.debug(f"time = {time}")
                    # color_transition_index = int(time / steps_per_color)
                    start_color = color_transition_pairs[color_transition_index][0]
                    end_color = color_transition_pairs[color_transition_index][1]
                    # start_color = turbo_colormap_data[color_transition_index]
                    # end_color = turbo_colormap_data[color_transition_index+1]
                    logger.debug(f"start_color = {start_color}")
                    logger.debug(f"end_color = {end_color}")
                    logger.debug(f"transition index = {color_transition_index}")
                    logger.debug(
                        f"percent of new color = {(step / steps_per_color_pair)}"
                    )
                    logger.debug(
                        f"amount of red {(end_color[0] - start_color[0]) * (step / steps_per_color_pair)}"
                    )
                    logger.debug(
                        f"amount of green {(end_color[1] - start_color[1]) * (step / steps_per_color_pair)}"
                    )
                    logger.debug(
                        f"amount of blue {(end_color[2] - start_color[2]) * (step / steps_per_color_pair)}"
                    )

                    color = [
                        start_color[j]
                        + (end_color[j] - start_color[j])
                        * (step / steps_per_color_pair)
                        for j in range(3)
                    ]

                    if agent_index == 0:
                        color_list.append(color)

                    self.create_trajectory_line_group(
                        agent_index,
                        points[time - 1 if time > 0 else time : time + group_size],
                        list(range(len(points))),
                        # color=[
                        #     self.track_color_rate * time / len(points),
                        #     0,
                        #     1 - self.track_color_rate * time / len(points),
                        # ],
                        color=color,
                    )
                    logger.debug(f"color = {color}")
                    step = (step + 1) % steps_per_color_pair
                    if step == 0:
                        color_transition_index += 1
                    logger.debug(f"step = {step}")
                logger.debug(f"color list\n{color_list}")
            else:
                # Create a LineSet
                """
                TOC -- 081225 8:41AM
                To get these colors to depend on time, we will need to group time ranges and
                create a different line_set for each time grouping.
                """
                lines = [[t, t + 1] for t in range(len(points) - 1)]
                line_set = open3d.geometry.LineSet(
                    points=open3d.utility.Vector3dVector(points),
                    lines=open3d.utility.Vector2iVector(lines),
                )

                """
                TOC -- 080825 12:05PM
                Make these colors configurable.
    
                TOC -- 081125 3:18PM
                The colors need to change through time, so I think I am going to have
                to split these up into groups of lines sets based on time step and
                color the groups.
                """
                colors = [
                    [
                        (agent_index % 2) * 0.5,
                        agent_index / self.num_agents,
                        1 - (agent_index / self.num_agents),
                    ]
                    for i in range(len(lines))
                ]
                line_set.colors = open3d.utility.Vector3dVector(colors)
                self.vis.add_geometry(line_set)
                self.trajectory_line_set.append(line_set)

                self.vis.get_render_option().line_width = 50

    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)
        self.terminated = False
        self.truncated = False

        """ 
        TOC -- 080825 11:12AM
        # The only option we have at the moment is to show the lines -- this design needs to change
        
        TOC -- 081325 8:01AM 
        Need to be able to run the animation to display the specified trajectories. So this option should
        be a dictionary with an option to either show the tracks or animate the given trajectories.  
        """

        if options is not None:
            # turn the video off when we are just showing the visualizer to grab the trajectory figure.
            self.store_video = False
            self.show_trajectory_lines = True
            self.agent_trajectories = options

        """
        TOC -- 080525 
        replace with function 
        """
        self.init_agents()

        """
        TOC -- 080525 
        replace with function 
        """
        self.init_targets()

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

        # TOC -- 080525 9:38PM
        # These have to be called after render_frame sets up the target meshes.
        # I don't like this because init doesn't set up all of the instance variables. That
        # should be fixed -- though this still needs to be done in this order because we
        # can't compute distances until the target meshes are set up.
        observation = self._get_obs()
        info = self._get_info()

        return observation, info

    """
    TOC -- 081225 
    Removed because I don't use this anymore. 
    """
    # def compute_direction(self, action_list):
    #     return [self._action_to_direction[a] for a in action_list]

    def convert_to_velocity_along_mesh(self, location, velocity):
        distances, closest_points = self.compute_distance_and_closest_points(
            self.mesh_scene, location + velocity
        )
        new_locations = closest_points
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
        """

        Args:
            action:

        Returns:

        """

        """
        TOC 080825 11:16AM
        For now there is nothing to change when we draw the trajectories, but we 
        really should probably be doing something else. 
        """
        if self.show_trajectory_lines or self.run_trajectories:
            terminated = False

        else:
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
                            self.np_random.normal(-0.8, 0.1, size=1)
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

            # update the agent's position based on its velocity
            self._agent_location = np.array(self._agent_location) + np.array(
                self._agent_velocity
            )

            # update the targets
            if self.moving_targets:
                self.update_target_locations()

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
            distance = np.array(self.compute_target_center_distances())

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
            
            TOC -- 081425 8:49AM
            Need some sort of systematic way to say a run is terminated. This one doesn't make
            much sense, 
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
        """
        TOC -- 081425 8:42AM
        return self.truncated so that user can quit out of visualizer to 
        quit the run instead of having to Ctrl-C. 
        """
        return observation, reward, terminated, self.truncated, info

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

        self.box_line_set = open3d.geometry.LineSet(
            points=open3d.utility.Vector3dVector(points),
            lines=open3d.utility.Vector2iVector(lines),
        )

        """
        TOC -- 080325
        This color should be configurable.
        """
        colors = [[1, 0, 0] for _ in range(len(lines))]
        self.box_line_set.colors = open3d.utility.Vector3dVector(colors)
        self.vis.add_geometry(self.box_line_set)

    def compute_distance_and_closest_points_to_vertices(
        self,
        submesh_vertices,
        locations,
    ):
        distances = np.zeros(len(locations))
        closest_points = np.zeros((len(locations), 3))
        for i in range(len(locations)):
            diff = submesh_vertices - locations[i]
            norms = np.apply_along_axis(np.linalg.norm, 1, diff)
            argmin = np.argmin(norms)
            closest_points[i] = submesh_vertices[argmin]
            distances[i] = norms[argmin]
        return distances, closest_points

    def compute_distance_and_closest_points(self, mesh, agents_loc):
        """
        TOC -- 072325
        This should be done more intelligently.
        """
        if mesh is None:
            return np.zeros(self.num_agents), np.zeros(self.num_agents)

        """
        TOC -- 073125 
        This Raycasting setup shouldn't be redone everytime we need to calculate distances. 
        """
        mesh_scene = open3d.t.geometry.TriangleMesh.from_legacy(mesh)

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
        return unsigned_distance.numpy(), closest_points_dict["points"].numpy()

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
            if self.run_trajectories:
                self.mesh_agent[i].translate(
                    np.array(self._agent_location[i]), relative=True
                )
            else:
                self.mesh_agent[i].translate(np.array(self._agent_velocity[i]))
            self.vis.update_geometry(self.mesh_agent[i])

    def move_target_meshes(self):
        for i in range(self.num_targets):
            if self.run_trajectories:
                self.mesh_target_list[i].translate(
                    np.array(self._target_locations[i]), relative=False
                )
            else:
                self.mesh_target_list[i].translate(np.array(self._target_velocity[i]))

            self.vis.update_geometry(self.mesh_target_list[i])

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
        
        TOC -- 081225 3:20PM
        I am switching this to rotate around the origin and to do the translate to the 
        position after the rotate. I am not sure this will work for general rotations 
        and positions, so this needs to be tested. Though it seems to work for the 
        weird mesh angle.   
        """
        mesh_scene.rotate(R, center=(0, 0, 0))
        # mesh_scene.rotate(R, center=mesh_scene.get_center())
        # R = np.array([ [0.1, 0, 0], [0,0,0], [0,0,0] ])
        # self.mesh_scene.rotate(R, center=self._target_location)

        mesh_scene.translate(position)
        return mesh_scene

    def key_callback_save_image(self, vis):
        self.save_image_to_file()

    def key_callback_quit(self, vis):
        self.truncated = True

    def key_callback_reset_view(self, vis):
        self.vis.reset_view_point()

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
        self.vis = open3d.visualization.VisualizerWithKeyCallback()
        self.vis.create_window(
            width=self.vis_width, height=self.vis_height, visible=self.show_visualizer
        )

        """
        TOC -- 081325 -- 7:00PM
        """

        self.vis.register_key_callback(ord("P"), self.key_callback_save_image)
        self.vis.register_key_callback(ord("Q"), self.key_callback_quit)
        self.vis.register_key_callback(ord("R"), self.key_callback_reset_view)

        if self.store_video:
            #
            # Create VideoWriter object.
            #
            # TOC -- 080325 12:13PM
            # This file type should be configurable.
            #
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")

            """
            TOC -- 080925 4:10PM
            I think this needs to use the expand path function to find the correct
            path to the output folder.
            """
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
        if (
            self.walking
            and not self.run_trajectories
            and not self.show_trajectory_lines
        ):
            distances, closest_points = self.compute_distance_and_closest_points(
                self.mesh_scene, self._agent_location
            )
            #
            # TOC -- 080625 10:49PM
            # closest points is now already converted to numpy so don't need to do that
            # any more.
            #
            self._agent_location = closest_points

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

        if self.submesh_target:
            colors = np.asarray(self.mesh_scene.vertex_colors)

            # # If no existing colors, create a default white array
            # if len(colors) == 0:
            #     colors = np.ones_like(vertices)

            # Modify colors for specific indices
            colors[self.submesh_vertex_indices] = self.submesh_color

            # Update mesh vertex colors
            self.mesh_scene.vertex_colors = open3d.utility.Vector3dVector(colors)
            self.vis.update_geometry(self.mesh_scene)
        else:
            """
            TOC -- 080525
            Right now all targets will be created at the same time. This needs to be changed 
            to have configurable times for each target. 
            """
            for i in range(self.num_targets):
                self.vis.add_geometry(self.mesh_target_list[i])
            # if self.walking:
            #     self.vis.add_geometry(self.mesh_ground_target[i])
            # else:
            #     self.vis.add_geometry(self.mesh_target)

    def init_sub_mesh_target(self):
        """
        Initializes the mesh necessary to compute the positions of the vertices
        from the target submesh. This function creates the submesh, rotates the
        submesh and translates the submesh. We can then use the submesh indices
        we extracted from the query file to find the new vertices. Maybe there
        is a direct way to do the rotation on our list of vertices, but it is
        probably safer right now to rely on open3d. Come back to this later.

        Returns:

        """

    def init_target_meshes(self):
        self.mesh_target_list = [None] * self.num_targets
        for i in range(self.num_targets):
            self.mesh_target_list[i] = open3d.geometry.TriangleMesh.create_sphere(
                radius=1.0
            )
            self.mesh_target_list[i].compute_vertex_normals()
            self.mesh_target_list[i].paint_uniform_color([0.1, 0.6, 0.1])
            self.mesh_target_list[i].scale(
                scale=self.target_scale, center=self.mesh_target_list[i].get_center()
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
                #
                # TOC -- 080625 10:38PM
                # In the new mesh target distance calculation, closest_points is already converted to
                # numpy. Make sure this is consistent throughout the different configurations.
                #
                self._target_location[i] = closest_points[0]

            """
            TOC -- 073125 -- 7:30AM 
            Need to decide on how we are dealing with ground and non-ground targets. The initial non-ground
            target was to keep boids away from the walls, but that doesn't make sense from a food source 
            point of view. Each of the targets is going to have to have some weight (possibly changing)
            associated with it. 
            """
            self.mesh_target_list[i].translate(self._target_location[i])
            # self.mesh_ground_target.translate(self._ground_target_location)
            self.vis.update_geometry(self.mesh_target_list[i])
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

            if self.target_mesh_file is not None:
                vertex_mask = np.zeros(len(self.mesh_scene.vertices), dtype=bool)
                vertex_mask[self.submesh_vertex_indices] = True
                """
                TOC -- 081225 4:08PM
                This doesn't work without converting to np.array. Need to work this out. 
                """
                self.submesh_vertices = np.array(self.mesh_scene.vertices)[
                    vertex_mask
                ]  # + mesh.get_center()

        if self.show_box:
            self.add_box()

        self.init_agent_meshes()

        self.init_target_meshes()

        """
        TOC -- 080225 2:56PM
        Some of these meshes are not used and need to be removed.
        """
        self.mesh_sphere_world1 = open3d.geometry.TriangleMesh.create_sphere(radius=0.1)
        self.mesh_sphere_world1.compute_vertex_normals()
        self.mesh_sphere_world1.paint_uniform_color([0.0, 0.0, 0.0])
        self.mesh_sphere_world1.translate([0.0, 0.0, self.max_dist_from_center])

        self.mesh_sphere_center = open3d.geometry.TriangleMesh.create_sphere(radius=4.0)
        self.mesh_sphere_center.compute_vertex_normals()
        self.mesh_sphere_center.paint_uniform_color([0.0, 0.0, 1.0])
        self.mesh_sphere_center.translate(
            [self.box_size / 2.0, self.box_size / 2.0, self.box_size / 2.0]
        )

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
            if self.submesh_target:
                """
                TOC -- 081425 11:32AM
                Change the color of the target submesh so it is discernible in the visualizer. 
                """
                colors = np.asarray(self.mesh_scene.vertex_colors)

                # # If no existing colors, create a default white array
                # if len(colors) == 0:
                #     colors = np.ones_like(vertices)

                # Modify colors for specific indices
                colors[self.submesh_vertex_indices] = self.submesh_init_color

                # Update mesh vertex colors
                self.mesh_scene.vertex_colors = open3d.utility.Vector3dVector(colors)
            self.vis.add_geometry(self.mesh_scene)

        # self.vis.add_geometry(self.mesh_top_corner)

        # self.vis.add_geometry(self.mesh_sphere_world1)
        # self.vis.add_geometry(self.mesh_sphere_center)
        # self.vis.add_geometry(self.mesh_sphere_start)

    def save_image_to_file(self):
        self.image_count += 1
        # Capture video
        img = self.vis.capture_screen_float_buffer()
        # logger.debug(f'img is {img}')

        img = (255 * np.asarray(img)).astype(np.uint8)
        # logger.debug(f'after astype {img}')
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        # logger.debug(f'after cvtColor {img}')
        image_path = expand_path(f"image-{self.image_count}.png", self.saved_image_path)
        cv2.imwrite(image_path, img)
        # don't keep saving the image, just save it once
        logger.debug(f"image written to {image_path}")

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

            # TOC -- 080825 11:48AM
            if self.show_trajectory_lines:
                self.init_trajectory_lines()

        # If we are showing the trajectory lines, just show the lines, don't
        # move the agents around.
        if self.show_trajectory_lines:
            for i in range(self.num_agents):
                self.vis.update_geometry(self.trajectory_line_set[i])
                self.vis.update_geometry(self.mesh_agent[i])
        else:
            self.move_agent_meshes()

            """
            TOC -- 080225 2:14PM 
            Why am I not translating the ground target by the ground target velocity?
            """
            if self.moving_targets:
                self.move_target_meshes()

        self.vis.poll_events()
        self.vis.update_renderer()

        #
        # TOC -- 080925 4:03PM
        # Save the image to a file when user requests. Then reset the save flag
        # so we don't keep saving images.
        if self.save_image:
            self.save_image_to_file()

            # self.image_count += 1
            # # Capture video
            # img = self.vis.capture_screen_float_buffer()
            # # logger.debug(f'img is {img}')
            #
            # img = (255 * np.asarray(img)).astype(np.uint8)
            # # logger.debug(f'after astype {img}')
            # img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            # # logger.debug(f'after cvtColor {img}')
            # image_path = expand_path(
            #     f"image-{self.image_count}.png", self.saved_image_path
            # )
            # cv2.imwrite(image_path, img)
            # # don't keep saving the image, just save it once
            # logger.debug(f"image written to {image_path}")
            self.save_image = False

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
        TOC -- 080125 
        Added this as part of Issue #13 to clean up the simulator. The 
        visualization window should close properly with this fix. 
        """
        if self.vis is not None:
            self.vis.destroy_window()

        if self.video_out is not None:
            self.video_out.release()
