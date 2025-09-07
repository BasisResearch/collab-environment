import gymnasium as gym
import numpy as np
import open3d
import cv2
import torch
from gymnasium import spaces

# from Boids.sim_utils import calc_angles
from loguru import logger
from scipy.spatial import cKDTree as KDTree

from collab_env.data.file_utils import get_project_root, expand_path
from collab_env.sim.boids.sim_utils import get_submesh_indices_from_ply
from collab_env.sim.util.color_maps import ColorMaps


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
        target_creation_time=None,  # need to figure this out
        agent_trajectories=None,
        target_trajectories=None,
        run_trajectories=False,  # run trajectories is for actually moving the agent meshes
        show_trajectory_lines=False,  # show the trajectory lines for the agent
        save_image=False,
        color_tracks_by_time=False,
        number_track_color_groups=1,
        track_color_rate=4,
        kd_workers=1, # 1 seems to work better than -1 on Macbook Pro M4 MAX
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
        self.target_creation_time = (
            [0] if target_creation_time is None else target_creation_time
        )
        self.agent_scale = agent_scale
        self.scene_scale = scene_scale
        self.scene_filename = scene_filename
        self.scene_position = np.array(scene_position)
        self.scene_angle = np.array(scene_angle)
        self.show_visualizer = show_visualizer
        self.store_video = store_video
        self.store_video_init = store_video
        self.video_file_path = video_file_path
        self.video_codec = video_codec
        self.video_fps = video_fps
        self.vis_width = vis_width
        self.vis_height = vis_height
        logger.debug(f"video path is {self.video_file_path}")
        logger.debug(f"store video is {self.store_video}")
        self.mesh_target_list = [None] * self.num_targets
        self.mesh_target_raycasting = [None] * self.num_targets

        self.agent_trajectories = agent_trajectories
        self.target_trajectories = target_trajectories
        self.run_trajectories = run_trajectories
        self.show_trajectory_lines = show_trajectory_lines
        self.save_image = save_image
        self.save_image_init = (
            save_image  # used to reset for saving images for each episode
        )
        self.saved_image_path = saved_image_path
        self.color_tracks_by_time = color_tracks_by_time
        self.number_track_color_groups = number_track_color_groups
        self.track_color_rate = track_color_rate

        self.trajectory_line_set = None

        self.time_step = 0
        self.image_count = 0
        self.terminated = False
        self.truncated = False
        self.raycasting_scene = None
        self.kd_workers = kd_workers

        """
        -- 081225 3:58PM
        If we have a target mesh file to read in, we need to get the vertices labeled as the 
        target from the file and rotate and translate them to their corresponding positions
        on the scene mesh. That was silly. We get the indices by calling get_submesh_indices_from_ply(). 
        Then we don't have to translate and rotate anything because the indicies stay the same. We can
        then just get the vertices out of the scene vertices by choosing by index after we load and rotate
        the scene in load_and_rotate()  
        """
        self.submesh_target = False
        #
        # -- 082425 1:01PM
        # Create a ragged numpy array. Not sure I need this to be a numpy array. Should
        # probably just use a python list. This blew up as an np.array, just go with
        # a list.
        #
        self.num_submesh_targets = 0
        self.submesh_vertex_indices = []
        self.submesh_kdtree = []
        self.submesh_vertices = []
        if self.target_mesh_file is not None:
            self.num_submesh_targets = len(self.target_mesh_file)
            self.submesh_target = True
            self.submesh_vertices = [None] * self.num_submesh_targets
            self.submesh_kdtree = [None] * self.num_submesh_targets
            for t in range(self.num_submesh_targets):
                logger.debug(f"target mesh file {self.target_mesh_file[t]}")
                self.submesh_vertex_indices.append(
                    get_submesh_indices_from_ply(
                        expand_path(self.target_mesh_file[t], get_project_root())
                    )
                )
                logger.info(
                    f"{len(self.submesh_vertex_indices[t])} found in sub-mesh target"
                )
            """
            -- 082525 12:42PM 
            Stubbed for testing second mesh since I don't have a second query file yet.
            
            TAKE THIS OUT AFTER TESTING 
            """
            # self.submesh_vertex_indices[1] = list(range(121350, 121550))

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
                # -- 080525 2:01PM
                # replaced this with list of targets.
                # "target_loc": spaces.Box(
                #     low=-torch.inf, high=torch.inf, shape=(3,), dtype=np.float64
                # ),
                "target_loc": spaces.Tuple(
                    [
                        spaces.Box(
                            low=-torch.inf, high=torch.inf, shape=(3,), dtype=np.float32
                        )
                        for _ in range(max(num_targets, 1))
                    ]
                ),
                #
                # -- 080525 2:06PM
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
                                for _ in range(max(num_targets, 1))
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
                                for _ in range(max(num_targets, 1))
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
                                for _ in range(max(num_targets, 1))
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
        -- 080125 1154PM 
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

    def _get_obs(self):
        # -- 080625 9:03PM
        # Modifying to move to the closest points on the target meshes rather than the
        # center of the target meshes. This requires a change to the observation space
        # to include the closest points between agents and targets. This will be an array
        # of shape (num_agents, num_targets) with each entry being a box of length 3.
        # -- 080425 7:50PM
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

        if self.num_targets > self.num_submesh_targets:
            distances_to_target_mesh, target_mesh_closest_points = (
                self.compute_mesh_target_distances()
            )

        if self.submesh_target:
            distances_to_submesh_targets, closest_points_submesh_targets = (
                self.compute_target_distances_to_vertices()
            )
            if self.num_targets > self.num_submesh_targets:
                distances_to_target_mesh = np.concatenate(
                    (distances_to_submesh_targets, distances_to_target_mesh), axis=1
                )
                target_mesh_closest_points = np.concatenate(
                    (closest_points_submesh_targets, target_mesh_closest_points), axis=1
                )
            else:
                distances_to_target_mesh = distances_to_submesh_targets
                target_mesh_closest_points = closest_points_submesh_targets

        result = {
            "agent_loc": self._agent_location,
            "agent_vel": self._agent_velocity,
            "target_loc": self._target_location,
            # "target_loc":
            "distances_to_target_centers": self.compute_target_center_distances(),
            "distances_to_target_mesh_closest_points": distances_to_target_mesh,  # modified 080625 9:08PM
            "target_mesh_closest_points": target_mesh_closest_points,  # modified 080625 9:08PM
            "mesh_scene_distance": scene_distances,
            "mesh_scene_closest_points": scene_closest_points,
        }
        logger.debug(f"len of obs agent_loc {len(result['agent_loc'])}")
        logger.debug(f"len of obs agent_vel {len(result['agent_vel'])}")
        logger.debug(f"{result['target_loc'].shape}")
        logger.debug(f"{result['distances_to_target_centers'].shape}")
        logger.debug(f"{result['distances_to_target_mesh_closest_points'].shape}")
        logger.debug(f"{result['target_mesh_closest_points'].shape}")
        logger.debug(f"{result['mesh_scene_distance'].shape}")
        logger.debug(f"{result['mesh_scene_closest_points'].shape}")
        return result

    def _get_obs_to_target_centers(self):
        """ """

        """
        #
        # -- 080425 7:50PM
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
            "agent_loc": spaces.Tuple(self._agent_location),
            "agent_vel": spaces.Tuple(self._agent_velocity),
            "target_loc": spaces.Tuple(self._target_location),
            # "target_loc":
            "distances_to_targets": self.compute_target_center_distances(),
            "mesh_distance": distances,
            "mesh_closest_points": closest_points,
        }

    def _get_obs_no_target_list(self):
        #
        # -- 080425 7:50PM
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

    """
    -- 082425 1:17PM
    I don't think this function is called anymore. It will blow up if it is, so I 
    guess we will find out.  
    """

    def get_distances(self, agent_locations, submesh_vertices):
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

        # d_v, c_v = self.compute_distance_and_closest_points_to_vertices(
        #      self.submesh_vertices, self._agent_location
        # )

        d, c = self.compute_distance_and_closest_points_to_vertices_kdtree(
             self.submesh_vertices, self.submesh_kdtree, self._agent_location
        )

        '''
        -- 090625 7:20PM 
        Tested kdtree to make sure it matched the norm calculation, which it does with negligible 
        numerical differences. 
        '''
        # diff_d = d - d_v
        # indices = np.nonzero(diff_d)
        # diff_c = c[indices]
        # diff_cv = c_v[indices]
        # logger.debug(f'diff c:\n {diff_c}')
        # logger.debug(f'diff c_v:\n {diff_cv}')
        """
        -- 082425 1:15PM
        These are now coming back as lists, so no need to force that here. 
        """
        distances = np.array(d)
        closest_points = np.array(c)
        distances = np.array(distances).transpose()

        # logger.debug(f"dist {distances}")
        # logger.debug(f"dist shape {distances.shape}")

        closest_points = np.array(closest_points).transpose((1, 0, 2))

        # logger.debug(f"points {closest_points}")
        # logger.debug(f"points shape {closest_points.shape}")

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
        # -- 080625 10:27PM
        # There should be a more efficient way to do this without a for loop, but
        # I don't have time to worry about it right now
        #
        for i in range(len(self.mesh_target_list)):
            d, c = self.compute_distance_and_closest_points(
                self.mesh_target_list[i],
                self._agent_location,
                raycasting=self.mesh_target_raycasting[i],
            )
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
        -- 071825
        need to make this faster
        might want to have a different target for each agent -- in fact, definitely do

        -- 080425
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
        -- 071625
        need to look at this for performance later.

        -- 080325
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
            -- 080825 11:21AM
            Fix this after figure out agent trajectoies. Actually, we may never need this since we may never do
            moving targets.  
            """
            if self.target_trajectories is not None:
                self._target_location = self.target_trajectories[0]
            else:
                self._target_location = np.zeros((self.num_targets, 3))
        else:
            """
            -- 081125 7:00PM
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
                    """
                    -- 082425 2:18PM (comment added)
                    This gives an option of not putting a specified position for each 
                    target. Those unspecified will be chosen randomly. 
                    """
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
            # -- 080525 3:54PM
            # I think this will work without the if, but better to be safe
            #
            """
            -- 082125 9:48PM 
            This doesn't seem right. This should be put on the mesh scene not 0 on the 
            bottom of the cube. This needs to be fixed. 
            """
            if self.num_ground_targets > 0:
                self._target_location[self.ground_target_first_index :, 1] = 0.0

            """
            -- 080325 
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
            self._agent_velocity = np.zeros((self.num_agents, 3))
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
            -- 072325 
            We have to wait for rendering to be initialized for the mesh_scene to be created I think. 
            """
            if self.mesh_scene is not None and self.walking:
                distances, closest_points = self.compute_distance_and_closest_points(
                    self.mesh_scene, self._agent_location
                )
                self._agent_location = closest_points

            """
            -- 073025 
            Need to replace hard coded numbers. 
    
            -- 080325
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
        -- 081225 9:17AM
        Need to look into the fact that there seems to be an extra set of points all at 0 at the end of the
        list of points. 
        """
        lines = [[t, t + 1] for t in range(len(points) - 1)]
        line_set = open3d.geometry.LineSet(
            points=open3d.utility.Vector3dVector(points),
            lines=open3d.utility.Vector2iVector(lines),
        )

        """
        -- 082525 10:41PM 
        Occasionally have a problem with empty linesets and get warning from Open3D. 
        This could be caused by a weird group size, but need to investigate. 
        """

        """
        -- 080825 12:05PM 
        Make these colors configurable.

        -- 081125 3:18PM
        The colors need to change through time, so I think I am going to have
        to split these up into groups of lines sets based on time step and 
        color the groups. 
        """
        color_list = [color] * len(lines)

        line_set.colors = open3d.utility.Vector3dVector(color_list)
        self.vis.add_geometry(line_set)
        self.trajectory_line_set.append(line_set)

    """
    -- 082025 9:26PM 
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

        turbo_colormap_data = ColorMaps.turbo_colormap_data
        logger.debug(f"number of colors = {len(turbo_colormap_data)}")

        color_transition_pairs = list(
            zip(
                turbo_colormap_data[: len(turbo_colormap_data) - 1],
                turbo_colormap_data[1:],
            )
        )

        """
        -- 082425 9:42AM
        This code assumes that the length of the agents' trajectories could be 
        different for each agent. This could probably run a little faster if
        we didn't assume that and instead computed the colors once. 
        """
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
                logger.debug(f"len of color list = {len(color_list)}")
                logger.debug(f"len of color map = {len(turbo_colormap_data)}")
            else:
                # Create a LineSet
                """
                -- 081225 8:41AM
                To get these colors to depend on time, we will need to group time ranges and
                create a different line_set for each time grouping.
                """
                lines = [[t, t + 1] for t in range(len(points) - 1)]
                line_set = open3d.geometry.LineSet(
                    points=open3d.utility.Vector3dVector(points),
                    lines=open3d.utility.Vector2iVector(lines),
                )

                """
                -- 080825 12:05PM
                Make these colors configurable.
    
                -- 081125 3:18PM
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
        logger.debug(f"seed = {seed}")
        self.terminated = False
        self.truncated = False
        self.time_step = 0

        """
        -- 082525 2:30PM 
        Reset the save image flag to what it was at initialization since it is turned off
        once the first image is saved so that we don't save every frame. 
        
        Do the same for store_video since we turn that off for show_trajectories. 
        """

        self.save_image = self.save_image_init
        self.store_video = self.store_video_init

        """ 
        -- 080825 11:12AM
        # The only option we have at the moment is to show the lines -- this design needs to change
        
        -- 081325 8:01AM 
        Need to be able to run the animation to display the specified trajectories. So this option should
        be a dictionary with an option to either show the tracks or animate the given trajectories.  
        """

        if options is not None:
            # turn the video off when we are just showing the visualizer to grab the trajectory figure.
            """
            -- 082525 2:28PM
            Turning store video off doesn't work. This screws up the next episode because video
            is off when they want it one. 
            """
            self.store_video = False
            self.show_trajectory_lines = True
            self.agent_trajectories = options

        """
        -- 080525 
        replace with function 
        """
        self.init_agents()

        """
        -- 080525 
        replace with function 
        """
        self.init_targets()

        """
        -- 080225 8:08AM
        Why am I setting these to None after I called render_frame() above? 
        These need to be moved -- moved and it still works. 
        """
        self.vis = None
        self.geometry = None
        self.mesh_sphere_agent = None

        # -- 080225 4:40PM
        # Always need to call _render_frame from reset because we rely on the meshes
        # for the simulation.
        # if self.render_mode == "human":
        self._render_frame()

        # -- 080525 9:38PM
        # These have to be called after render_frame sets up the target meshes.
        # I don't like this because init doesn't set up all of the instance variables. That
        # should be fixed -- though this still needs to be done in this order because we
        # can't compute distances until the target meshes are set up.
        observation = self._get_obs()
        info = self._get_info()
        return observation, info

    """
    -- 081225 
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
        # -- 080425 8:00PM
        # If there is no mesh_scene, just move the ground target along the bottom of the box
        # by setting its velocity in the vertical direction to 0, but right now there is no
        # ground target if we are not walking. We can go back to this when we create multiple
        # targets. On second thought, we should have the ability to have both ground and air
        # targets, so this is the right way to do this. When we have multiple targets, this
        # should be an array of ground targets.
        #

        #
        # -- 080525 3:18PM
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
            -- 080425 2:33PM
            Need not do to this until the targets are created. Also need one target or a list instead of using ground. 
            """
            # self._ground_target_location = (
            #        self._ground_target_location + self._ground_target_velocity
            # )
            """
            -- 080525 -- 3:16PM
            This should work as a numpy array instead of needing to do this is a for loop
            """
            self._target_location = self._target_location + self._target_velocity

    def step(self, action):
        """

        Args:
            action:

        Returns:

        """

        self.time_step += 1

        """
        -- 080825 11:16AM
        For now there is nothing to change when we draw the trajectories, but we 
        really should probably be doing something else. 
        """
        if self.show_trajectory_lines:
            terminated = not self.show_visualizer

        elif self.run_trajectories:
            terminated = False

        else:
            for t in range(self.num_targets):
                if self.time_step == self.target_creation_time[t]:
                    self.create_targets(t)

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
            -- 072125
            Need to fix all these types to make sure I consistently have ndarrays.
            
            needed to update the agent velocities to be the current action  
            """
            self._agent_velocity = action

            """
            -- 072125 16:42 
            if outside the box, go backwards along the axis that hit but do so randomly so it looks less
            like a bounce. (Maybe I should do this for hitting the obstacles as well but obstacle avoidance
            should be done in the Agent class when it chooses an action. The environment can bounce the agents
            but they should control their own steering.)
            
            -- 072925 
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
            -- 072325 -- 0808PM 
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
            -- 071625
            needs distances to be a list 
            
            -- 072825 
            not sure I am using distances anymore. Ah, I was treating the world to be constrained within a sphere but went 
            with the cube when the sphere calculations didn't work so well. Perhaps I should go back to the sphere approach
            since Matt suggested that is what he likes to do with dynamical systems involving particles running out to 
            infinity. Also, Reynolds indicated that having a target point to go to created unnatural behavior in boids.
            
            -- 073125 2:51PM
            Looks like I am using this to determine whether anyone hit the target. Not sure this is necessary. I think
            we have this computed in the observation, but not sure we have that in time for checking the hit if we need
            to do that to determine whether the agent ate the food.  
            """
            # distance = np.linalg.norm(self._agent_location - self._target_location, ord=1)
            distance = np.array(self.compute_target_center_distances())

            # logger.debug('new distance:' + str(distance))
            """
            -- 072125 12:41PM -- reverse direction if too far from the center -- maybe change this to add the target as a 
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
            -- 072125 -- Check norms 
            """
            # if reversed:
            #     norms = np.apply_along_axis(np.linalg.norm, 1, self._agent_location)
            #     indices = np.where(norms > self.max_dist_from_center)
            #     logger.debug('step() indices ' + str(indices))
            #     logger.debug('step(): norms: ' + str(norms))
            #     logger.debug('step(): velocities : ' + str(self._agent_velocity))

            # logger.debug('distance condition thingy ' + str(distance[distance < HIT_DISTANCE]))
            """
            -- 072925 
            I am setting terminated to be when an agent hits the goal. Somewhere, I must have set this to false. I should 
            set this to false here if that is what I want. Terminated should be controlled here and nowhere else. 
            runboids() is just ignoring terminated -- Never mind.  
            """
            """
            -- 080325
            Hit distance should be configurable
            
            -- 081425 8:49AM
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
        -- 073125 2:51PM
        This isn't used. Can probably speed things up by skipping this.
        """
        info = self._get_info()

        # -- 080225 4:41PM
        # always want to call _render_frame because we rely on the meshes for the simulation
        # if self.render_mode == "human":
        self._render_frame()

        # logger.debug('step(): observation: ' + str(observation), level=1)
        # logger.debug('step(): reward: ' + str(reward), level=3)
        # logger.debug('step(): info: ' + str(info), level=4)
        """
        -- 081425 8:42AM
        return self.truncated so that user can quit out of visualizer to 
        quit the run instead of having to Ctrl-C. 
        """
        return observation, reward, terminated, self.truncated, info

    """
    -- 073125 2:30PM
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
        -- 080325
        This color should be configurable.
        """
        colors = [[1, 0, 0] for _ in range(len(lines))]
        self.box_line_set.colors = open3d.utility.Vector3dVector(colors)
        self.vis.add_geometry(self.box_line_set)

    def compute_distance_and_closest_points_to_vertices_kdtree(
        self,
        submesh_vertices,
        submesh_kdtree,
        locations,
    ):
        """

        Args:
            submesh_vertices:
            submesh_kdtree:
            locations:

        Returns:

        """

        distances = np.zeros((self.num_submesh_targets, len(locations)))
        closest_points = np.zeros((self.num_submesh_targets, len(locations), 3))

        for t in range(self.num_submesh_targets):
            logger.debug(f"t: {t}")
            dist_kd, closest_indices = submesh_kdtree[t].query(locations, workers=self.kd_workers)
            logger.debug(f"distances[t] shape: {distances[t].shape}")
            logger.debug(f"dist_kd shape: {dist_kd.shape}")
            logger.debug(f"closest indices shape: {closest_indices.shape}")
            # logger.debug(f'closest indices: {closest_indices}')
            logger.debug(f"closest points shape: {closest_points[t].shape}")
            logger.debug(f"submesh_vertices[t] shape: {submesh_vertices[t].shape}")
            logger.debug(
                f"submesh_vertices[t][closest_indices] shape: {submesh_vertices[t][closest_indices].shape}"
            )
            # closest_points[t] = np.squeeze(submesh_vertices[t][closest_indices], axis=1)
            # scipy doesn't require the squeeze, sklearn KDTree does.
            closest_points[t] = submesh_vertices[t][closest_indices]
            logger.debug(f"closest points: {closest_points[t]}")
            # distances[t] = np.squeeze(dist_kd, axis=1)
            distances[t] = dist_kd
        return distances, closest_points

    def compute_distance_and_closest_points_to_vertices(
        self,
        submesh_vertices,
        locations,
    ):
        """
        -- 082225 10:41AM

        Need to look into k-d trees for doing the distance computations.

        Args:
            submesh_vertices:
            locations:

        Returns:

        """
        distances = np.zeros((self.num_submesh_targets, len(locations)))
        closest_points = np.zeros((self.num_submesh_targets, len(locations), 3))
        for t in range(self.num_submesh_targets):
            for i in range(len(locations)):
                diff = submesh_vertices[t] - locations[i]
                norms = np.apply_along_axis(np.linalg.norm, 1, diff)
                argmin = np.argmin(norms)
                closest_points[t][i] = submesh_vertices[t][argmin]
                distances[t][i] = norms[argmin]
        return distances, closest_points

    def compute_distance_and_closest_points(self, mesh, agents_loc, raycasting=None):
        """
        -- 072325
        This should be done more intelligently.

        """
        if mesh is None:
            return np.zeros(self.num_agents), np.zeros(self.num_agents)

        #
        # If no raycaster is passed in, we will assume the distances to the mesh scene
        # are being computed.
        #
        if raycasting is None:
            raycasting = self.raycasting_scene

        """
        -- 073125 
        This Raycasting setup shouldn't be redone every time we need to calculate distances. 
        
        -- 082225 10:32AM
        There is an issue with the order in which initialization happens. init_targets() calls this function 
        before init_meshes has been called. Setting up the raycasting make sense in init_meshes but 
        """
        # if self.raycasting_scene is None:
        #     temp_mesh_scene = open3d.t.geometry.TriangleMesh.from_legacy(mesh)
        #
        #     # Create a scene and add the triangle mesh
        #     self.raycasting_scene = open3d.t.geometry.RaycastingScene()
        #     _ = self.raycasting_scene .add_triangles(temp_mesh_scene)  # we do not need the geometry ID for mesh

        query_points = open3d.core.Tensor(agents_loc, dtype=open3d.core.Dtype.Float32)

        """ 
        -- 073125 
        Do we really need both signed and unsigned distances? Also, do we need to call
        both compute_distance() and compute_closest_points()? I think that is information
        the GNN wanted in the output file. 
        """
        # Compute distance of the query point from the surface
        # unsigned_distance = scene.compute_distance(query_points)
        # signed_distance = scene.compute_signed_distance(query_points)
        # occupancy = scene.compute_occupancy(query_points)
        #
        # closest_points_dict = scene.compute_closest_points(query_points)

        unsigned_distance = raycasting.compute_distance(query_points)
        # signed_distance = self.raycasting_scene.compute_signed_distance(query_points)
        # occupancy = self.raycasting_scene.compute_occupancy(query_points)

        closest_points_dict = raycasting.compute_closest_points(query_points)

        logger.debug(f"unsigned distance {unsigned_distance.numpy()}")
        # logger.debug(f"signed_distance {signed_distance.numpy()}")
        # logger.debug(f"occupancy {occupancy.numpy()}")
        return unsigned_distance.numpy(), closest_points_dict["points"].numpy()

    def move_agent_meshes(self):
        # Reset to home position
        for i in range(self.num_agents):
            """
            -- 072625
            We have the acceleration in the BoidsAgent where it chooses actions. That is what we
            need I think to compute the orientation of the agent meshes. Or not.
            """

            """
            -- 072525
            I believe this is necessary because the rotations are not relative. Oh, but if the rotations
            are not relative, then there is no need to keep track of the previous velocity. We can just 
            rotate to the new velocity from [0,0,1], which is apparently the default direction. So I don't
            know why this is here or if we need it.   
            """
            # self.mesh_arrow_agent[i].transform(np.eye(4))

            """
            -- 072725 -- 11:28AM
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
        -- 072325 
        Took this from the example code from Open3D but this wasn't necessary. 
        """
        # self.mesh_scene_for_distance = open3d.t.geometry.TriangleMesh.from_legacy(self.mesh_scene)

        """
        -- 072325 -- Test distance calculation to splat mesh. 
        """
        # self.compute_distance_to_splat_mesh(self.mesh_scene)

        """
        -- 073025 -- 8:21PM
        Not sure why this is translating to this particular position -- needed that for first
        splat but I need to generalize. 
        Why am I scaling this centered at the target location instead of the center of the scene?
        This is a bug.  
        """
        #
        # -- 080325 1:19PM
        # The new meshes seem to come in centered at about the origin once they are rotated 90
        # degrees, but the mesh ground is a little below the bottom of the box. We could
        # translate the box, but I sort of like having the box bottom and sides to be 0 -- though
        # I guess we could translate the box to match the mesh rather than the other way around
        # and then use the box positions in the bounce off the wall conditions. Maybe let's just
        # make the mesh position configurable. I presume different meshes that people load may
        # have different properties, and it would be nice to allow the position of the mesh to
        # be configurable. We should translate and rotate it once though. Not sure what was up
        # with the double translation I had in here.

        # -- 080225 8:30AM
        # This scaling should be happening centered at the center of the mesh_scene
        # rather than the target location. Seems to work.
        #
        # mesh_scene.scale(scale=self.scene_scale, center=self._target_location)
        mesh_scene.scale(scale=self.scene_scale, center=mesh_scene.get_center())

        """
        -- 072725 -- 11:37AM

        This is specific to the first of Tommy's splats. This needs to be generalized to reorient 
        whatever scene mesh we are given to match the ground.  
        """

        """
        -- 073025 -- 7:45PM
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
        -- 073025
        Need to make sure this is the correct center for rotation. Seems like it should 
        be the center of the mesh scene after it has been translated.
        
        -- 081225 3:20PM
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
        -- 073125 
        Is this the right way to visualize. Kind of think there may have been a newer way to do this.

        -- 080125 10:59PM
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

        self.vis.register_key_callback(ord("P"), self.key_callback_save_image)
        self.vis.register_key_callback(ord("Q"), self.key_callback_quit)
        self.vis.register_key_callback(ord("R"), self.key_callback_reset_view)

        if self.store_video:
            #
            # Create VideoWriter object.
            #
            # -- 080325 12:13PM
            # This file type should be configurable.
            #
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")

            """
            -- 080925 4:10PM
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
        # -- 072325
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
            # -- 080625 10:49PM
            # closest points is now already converted to numpy so don't need to do that
            # any more.
            #
            self._agent_location = closest_points

        # self.mesh_sphere_agent = [None] * self.num_agents
        self.mesh_agent = [None] * self.num_agents
        for i in range(self.num_agents):
            """
            -- 080225 1:57PM 
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

    def create_targets(self, target_index):
        """
        This method simply adds the target meshes that were already created to the visualizer
        or recolors the submesh target if we have one.
        Returns:

        """

        # if self.submesh_target:
        #

        # # If no existing colors, create a default white array
        # if len(colors) == 0:
        #     colors = np.ones_like(vertices)

        """
        -- 082425 1:52PM 
        We will treat the initial indices as submesh targets. If there are 
        more targets than that, we will treat them as plain targets that are
        not a submesh of the scene and need to be created. That doesn't really
        make a lot of sense. Maybe we just shouldn't allow mixing and matching. 
        """
        if target_index < self.num_submesh_targets:
            colors = np.asarray(self.mesh_scene.vertex_colors)
            # Modify colors for specific target
            colors[self.submesh_vertex_indices[target_index]] = self.submesh_color[
                target_index
            ]

            # Update mesh vertex colors
            self.mesh_scene.vertex_colors = open3d.utility.Vector3dVector(colors)
            self.vis.update_geometry(self.mesh_scene)
        else:
            """
            -- 080525
            Right now all targets will be created at the same time. This needs to be changed 
            to have configurable times for each target. 
            
            -- 082425 1:58PM 
            The targets will be created based on the target_index if the target_index is beyond
            the submesh targets. Need to change init_targets to handle this. Also need to fix
            how we are handling the ground targets, since I am doing something janky with that
            index too. This needs a better design. Maybe initialization should handle all of
            this and the other methods should split up ground, air, and submesh targets more
            simply. 
            """
            # for i in range(self.num_targets):
            self.vis.add_geometry(
                self.mesh_target_list[target_index - self.num_submesh_targets]
            )
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

    def init_target_mesh_raycasting(self, target_mesh):
        temp_mesh = open3d.t.geometry.TriangleMesh.from_legacy(target_mesh)

        # Create a scene and add the triangle mesh
        raycasting = open3d.t.geometry.RaycastingScene()
        _ = raycasting.add_triangles(
            temp_mesh
        )  # we do not need the geometry ID for mesh
        return raycasting

    def init_target_meshes(self):
        """
        -- 082425 2:33PM
        This needs to change to only create targets beyond the number of submesh targets.
        Returns:

        """
        self.mesh_target_list = [None] * (self.num_targets - self.num_submesh_targets)
        for i in range(self.num_targets - self.num_submesh_targets):
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
            -- 072325 
            If agents are walking, put the target on the mesh scene. Screws up the scale 
            somehow. It must not be drawn to the correct spot. Try not drawing it.   
            """
            if (
                i + self.num_submesh_targets >= self.ground_target_first_index
                and self.mesh_scene is not None
            ):
                distances, closest_points = self.compute_distance_and_closest_points(
                    self.mesh_scene,
                    [self._target_location[i + self.num_submesh_targets]],
                )
                # self.mesh_sphere_target.translate(closest_points.numpy()[0] - self._target_location)
                #
                # -- 080625 10:38PM
                # In the new mesh target distance calculation, closest_points is already converted to
                # numpy. Make sure this is consistent throughout the different configurations.
                #
                self._target_location[i] = closest_points[0]

            """
            -- 073125 -- 7:30AM 
            Need to decide on how we are dealing with ground and non-ground targets. The initial non-ground
            target was to keep boids away from the walls, but that doesn't make sense from a food source 
            point of view. Each of the targets is going to have to have some weight (possibly changing)
            associated with it. 
            """
            self.mesh_target_list[i].translate(self._target_location[i])
            # self.mesh_ground_target.translate(self._ground_target_location)
            self.vis.update_geometry(self.mesh_target_list[i])
            # self.vis.update_geometry(self.mesh_ground_target)
            self.mesh_target_raycasting[i] = self.init_target_mesh_raycasting(
                self.mesh_target_list[i]
            )

    def init_meshes(self):
        """
        -- 073125
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

            if self.submesh_target is not None:
                for t in range(self.num_submesh_targets):
                    vertex_mask = np.zeros(len(self.mesh_scene.vertices), dtype=bool)
                    vertex_mask[self.submesh_vertex_indices[t]] = True
                    """
                    -- 081225 4:08PM
                    This doesn't work without converting to np.array. Need to work this out. 
                    """
                    self.submesh_vertices[t] = np.array(self.mesh_scene.vertices)[
                        vertex_mask
                    ]  # + mesh.get_center()
                    """
                    -- 090625 4:39PM 
                    """
                    self.submesh_kdtree[t] = KDTree(self.submesh_vertices[t])

        if self.show_box:
            self.add_box()

        """
        -- 080225 2:56PM
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
        -- 073125 -- 7:22AM
        Move all the add geometries into one spot. What is this top corner thing? (That 
        was for debugging the problem with all the boids going to the corner.)
        Do I want to put all of the adds for the agents in this spot too? That requires
        extra for loop. I think we might because then we can skip all of this if we are
        not visualizing.   
        """
        if self.mesh_scene is not None:
            if self.submesh_target:
                """
                -- 081425 11:32AM
                Change the color of the target submesh so it is discernible in the visualizer. 
                """
                colors = np.asarray(self.mesh_scene.vertex_colors)

                # # If no existing colors, create a default white array
                # if len(colors) == 0:
                #     colors = np.ones_like(vertices)
                """
                -- 082425 1:20PM
                This now handles a list of colors, one for each submesh. 
                """
                for t in range(self.num_submesh_targets):
                    # Modify colors for specific indices
                    colors[self.submesh_vertex_indices[t]] = self.submesh_init_color[t]

                # Update mesh vertex colors
                self.mesh_scene.vertex_colors = open3d.utility.Vector3dVector(colors)

            self.vis.add_geometry(self.mesh_scene)

            """
            -- 082225 1008AM 
            This Raycasting setup shouldn't be redone every time we need to calculate distances, 
            so moved it here.  
            
            Do I need the legacy thing? 
            """
            temp_mesh_scene = open3d.t.geometry.TriangleMesh.from_legacy(
                self.mesh_scene
            )

            # Create a scene and add the triangle mesh
            self.raycasting_scene = open3d.t.geometry.RaycastingScene()
            _ = self.raycasting_scene.add_triangles(
                temp_mesh_scene
            )  # we do not need the geometry ID for mesh

        """
        -- 082525 10:35AM 
        These must be called after the raycasting scene has been initialized. 
        Seems like there must be a better design for this, especially if we 
        separate the rendering out into a different class. 
        """
        self.init_agent_meshes()
        self.init_target_meshes()

        # self.vis.add_geometry(self.mesh_top_corner)

        # self.vis.add_geometry(self.mesh_sphere_world1)
        # self.vis.add_geometry(self.mesh_sphere_center)
        # self.vis.add_geometry(self.mesh_sphere_start)

    def save_image_to_file(self):
        self.image_count += 1
        logger.debug(f"image count = {self.image_count}")
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
        -- 073125
        Since we depend on the scene mesh for calculating distances to agents in the simulation, the scene mesh at
        least needs to be initialized regardless of whether the render mode is human or not. So this code needs to 
        change. Perhaps the scene should be initialized in reset() or init() instead. reset() makes sense if the 
        scene could change based on some user defined configuration between trials. 
        
        -- 080225 8:35AM
        Lots of work needs to be done here to deal with render mode. We need to make sure we 
        can calculate the distances to the mesh when we are not using the viewer and that 
        everything gets initialized correctly. There is lots of garbage code in here. 
        """

        if self.vis is None:
            self.initialize_visualizer()
            self.init_meshes()

            # -- 080825 11:48AM
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
            -- 080225 2:14PM 
            Why am I not translating the ground target by the ground target velocity?
            """
            if self.moving_targets:
                self.move_target_meshes()

        self.vis.poll_events()
        self.vis.update_renderer()

        #
        # -- 080925 4:03PM
        # Save the image to a file when user requests. Then reset the save flag
        # so we don't keep saving images.
        if self.save_image:
            logger.debug("calling save image to file and resetting save flag")
            self.save_image_to_file()
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
        -- 080125 
        Added this as part of Issue #13 to clean up the simulator. The 
        visualization window should close properly with this fix. 
        """
        if self.vis is not None:
            self.vis.destroy_window()

        if self.video_out is not None:
            self.video_out.release()
