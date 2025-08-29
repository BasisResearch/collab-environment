from typing import Optional, Tuple, TypeAlias, Union, List
from dataclasses import dataclass
import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree
from tqdm import trange, tqdm

import torch
from torch.nn import functional as F
from torch.func import vmap

# Pyro distributions
import pyro
import pyro.distributions as dist

# Utils
from collab_env.utils.utils import array_nan_equal, get_backend

#########################################################
####### TLB CLEAN UP THESE BUT PUT HERE FOR NOW #########
#########################################################

def bbox_to_coords(bbox, method="bottom_center"):
    """
    Converts a bounding box (x1, y1, x2, y2) to image coordinates (u, v)
    using the method specified.

    Args:
        bbox: (N, 4) array of (x1, y1, x2, y2)
        method: "bottom_center" or "center"

    Returns:
        uv: (N, 2) array of (u, v)
    """
    x1, y1, x2, y2 = bbox.astype(int).T

    u = (x1 + x2) // 2
    v = y2 - 1 if method == "bottom_center" else (y1 + y2) // 2
    uv = np.stack([u, v], axis=1)
    return uv

def filter_coords(coords, window=3, std_threshold=3):
    """
    Filter outlier coordinates in a trajectory using rolling statistics.

    Args:
        coords (ndarray or Tensor): shape (N, 2), with columns (u, v).
        window (int): Size of rolling window.
        std_threshold (float): Std deviation threshold.

    Returns:
        ndarray or Tensor: Filtered coordinates (same type as input).
    """
    xp = get_backend(coords)
    coords = coords.clone() if xp is torch else coords.copy()
    n = coords.shape[0]

    pad = window // 2
    pad_mode = "replicate" if xp is torch else "edge"
    padded = (
        xp.nn.functional.pad(coords[None], (0,0,pad,pad), mode=pad_mode)[0]
        if xp is torch else xp.pad(coords, ((pad, pad), (0, 0)), mode=pad_mode)
    )

    for i in range(n):
        window_slice = padded[i:i+window]
        mean = window_slice.mean(dim=0) if xp is torch else window_slice.mean(axis=0)
        std  = window_slice.std(dim=0, unbiased=False) if xp is torch else window_slice.std(axis=0)

        std = xp.where(std==0, xp.full_like(std, 1e-6), std) if xp is torch else np.where(std==0, 1e-6, std)

        diff = (coords[i] - mean) / std
        if xp.any(xp.abs(diff) > std_threshold):
            coords[i] = mean

    return coords

def smooth_coords(coords, window=3):
    """
    Smooth coordinates using rolling mean.

    Args:
        coords (ndarray or Tensor): shape (N, 2), with columns (u, v).
        window (int): Rolling window size.

    Returns:
        ndarray or Tensor: Smoothed coordinates (same type as input).
    """
    xp = get_backend(coords)
    coords = coords.clone() if xp is torch else coords.copy()
    n = coords.shape[0]

    pad = window // 2
    pad_mode = "replicate" if xp is torch else "edge"
    padded = (
        xp.nn.functional.pad(coords[None], (0,0,pad,pad), mode=pad_mode)[0]
        if xp is torch else xp.pad(coords, ((pad, pad), (0, 0)), mode=pad_mode)
    )

    out = []
    for i in range(n):
        window_slice = padded[i:i+window]
        mean = xp.nanmean(window_slice, dim=0) if xp is torch else np.nanmean(window_slice, axis=0)
        out.append(mean)

    return xp.stack(out, dim=0) if xp is torch else np.stack(out, axis=0)

################################
########### Agent ##############
################################

"""
Agent is an object that moves within the environment. It
starts from an initial position.
"""

State = Tuple[torch.Tensor, torch.Tensor]
ObservedState = Tuple[torch.Tensor, torch.Tensor]
# StateTrajectory = Tuple[torch.Tensor, torch.Tensor]


@dataclass
class StateTrajectory:
    """
    Stores trajectory of an agent in a batched tensor format.
    positions: (T, 3)
    sizes: (T, D) or (T,)
    """
    positions: torch.Tensor
    sizes: torch.Tensor

    def __post_init__(self):
        # Ensure tensors
        self.positions = torch.as_tensor(self.positions, dtype=torch.float32)
        self.sizes = torch.as_tensor(self.sizes, dtype=torch.float32)
        # Ensure matching first dimension
        assert self.positions.shape[0] == self.sizes.shape[0], "Positions and sizes must have same timesteps"

    @property
    def length(self) -> int:
        return self.positions.shape[0]

    def get_state(self, t: int) -> State:
        return self.positions[t], self.sizes[t]
    
    def append(self, state: State):
        """Append new state along time dimension."""
        pos, size = state
        pos = torch.as_tensor(pos, dtype=torch.float32).view(1, -1)
        size = torch.as_tensor(size, dtype=torch.float32).view(1, -1)
        self.positions = torch.cat([self.positions, pos], dim=0)
        self.sizes = torch.cat([self.sizes, size], dim=0)

    def __len__(self) -> int:
        return self.positions.shape[0]

@dataclass
class Agent:
    """
    Agent is an object that moves within the environment. It has
    an identifier and a trajectory of states.
    """
    id: str
    trajectory: StateTrajectory  # reference to the trajectory
    t: int = 0  # current timestep

    def __post_init__(self):
        # Ensure trajectory has at least one state
        if self.trajectory.length == 0:
            raise ValueError("Trajectory must contain at least one state")

    #########################################################
    #################### Properties #########################
    #########################################################

    @property
    def position(self) -> torch.Tensor:
        """Current position from trajectory"""
        return self.trajectory.positions[self.t]

    @property
    def size(self) -> torch.Tensor:
        """Current size from trajectory"""
        return self.trajectory.sizes[self.t]

    @property
    def area(self) -> torch.Tensor:
        """Geometric area (prod of size dimensions)"""
        return torch.prod(self.size)

    @property
    def state(self) -> State:
        return self.position, self.size

    @state.setter
    def state(self, state: State):
        pos, sz = state
        # Optionally, update current timestep in trajectory
        self.trajectory.positions[self.t] = torch.as_tensor(pos, dtype=torch.float32)
        self.trajectory.sizes[self.t] = torch.as_tensor(sz, dtype=torch.float32)

    def project_area(self, state: State, fx: float) -> torch.Tensor:
        """
        Project the area of the agent to the camera.
        state can be either a single (position, size) tuple,
        or batched: (positions: (T,3), sizes: (T,) or (T,1))
        Returns:
            area: (T,)
        """
        positions, sizes = state  # positions: (T,3), sizes: (T,) or (T,1)
        fx = torch.as_tensor(fx, dtype=torch.float32)

        # ensure positions are batched
        if positions.ndim == 1:
            positions = positions[None, :]

        # reshape sizes to (T,1)
        sizes = sizes.view(-1, 1)

        # compute safe x to avoid division by zero
        x_3d = positions[:, 0].view(-1, 1)  # shape (T,1)
        eps = 1e-8
        safe_x = torch.sign(x_3d) * torch.maximum(torch.abs(x_3d), torch.tensor(eps))

        # compute projected area
        area = torch.pi * sizes**2 * (fx / torch.abs(safe_x))**2  # shape (T,1)

        return area

    #########################################################
    #################### Navigation ##########################
    #########################################################

    def step(self, dt: int = 1):
        """Advance along trajectory"""
        self.t = min(self.t + dt, self.trajectory.length - 1)

    def reset(self):
        """Reset to start of trajectory"""
        self.t = 0

    def jump_to(self, t: int):
        """Jump to a specific timestep"""
        if t < 0 or t >= self.trajectory.length:
            raise IndexError(f"Timestep {t} out of bounds for trajectory length {self.trajectory.length}")
        self.t = t

#########################################################
################## Camera ##############################
#########################################################
@dataclass
class Camera:
    """
    Camera represents a view within the environment. It contains a
    set of transformations that relate the world points to the image.
    """

    def __init__(self, K: Union[torch.Tensor, np.array], c2w: Union[torch.Tensor, np.array], width: int, height: int):
        # Ensure everything is torch tensor
        self.K: torch.Tensor = torch.as_tensor(K, dtype=torch.float32)
        self.c2w: torch.Tensor = torch.as_tensor(c2w, dtype=torch.float32)
        self.width: int = width
        self.height: int = height

        # Image / Depth of the mesh
        self.image: Optional[torch.Tensor] = None   # (H, W, C) or (H, W)
        self.depth: Optional[torch.Tensor] = None   # (H, W)

        # Point distribution of camera viewing mesh
        self.dist: Optional[pyro.distributions.Distribution] = None

    def set_view(self, image, depth) -> None:
        self.image = torch.as_tensor(image, dtype=torch.float32)
        self.depth = torch.as_tensor(depth, dtype=torch.float32)

    def get_view(self) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        return self.image, self.depth

    @property
    def valid_pixels(self) -> torch.Tensor:
        """
        Get valid pixels in the camera view as integer indices (u, v).
        """
        assert self.depth is not None, "Depth is not set"
        mask = ~torch.isinf(self.depth)
        coords = mask.nonzero(as_tuple=False)  # (N, 2) with (v, u)
        return coords[:, [1, 0]]  # swap to (u, v)

    #########################################################
    #################### Distributions ######################
    #########################################################

    def set_distribution(self, method: str = "weighted", n_agents: int = 1) -> pyro.distributions.Distribution:
        """
        Set the distributions of world points for the camera.
        """
        _, depth_map = self.get_view()
        assert depth_map is not None, "Depth is not set"

        # Extract depth at valid pixels
        depth_values = depth_map[self.valid_pixels[:, 1], self.valid_pixels[:, 0]]

        if method == "weighted":
            weights = depth_values ** 2
        else:
            weights = torch.ones_like(depth_values, dtype=torch.float32)

        # Normalize weights
        weights = weights.clamp(min=1e-6)
        weights = weights / weights.sum()

        # Define categorical distribution
        self.dist = dist.Categorical(probs=weights).expand([n_agents])
    
    def sample_pixels(self, n_points: Optional[int] = None) -> torch.Tensor:
        """
        Sample a set of pixels from the camera view.
        """
        assert self.dist is not None, "Distribution is not set"

        pixel_idxs = pyro.sample(
            name="pixels",
            fn=self.dist,
            sample_shape=[] if n_points is None else [n_points]
        )
        
        pixels = self.valid_pixels[pixel_idxs]

        if pixels.ndim == 1:
            pixels = pixels.unsqueeze(0)

        return pixels

    def sample_world_points(self, n_points: Optional[int] = None) -> torch.Tensor:
        """
        Sample world points from the distribution.

        CURRENTLY WILL BREAK IF N_POINTS IS PROVIDED --> need to include batching dimension
        """
        assert self.dist is not None, "Distribution is not set"
        pixels = self.sample_pixels(n_points)
        return self.project_to_world(pixels)

    #########################################################
    #################### Camera properties ##################
    #########################################################

    @property
    def fx(self):
        return self.K[0, 0]

    @property
    def fy(self):
        return self.K[1, 1]

    @property
    def cx(self):
        return self.K[0, 2]

    @property
    def cy(self):
        return self.K[1, 2]

    @property
    def w2c(self):
        return torch.linalg.inv(self.c2w)

    #########################################################
    #################### Projections #######################
    #########################################################

    def project_to_camera(self, points3d: torch.Tensor):
        """
        Project 3D points (N, 3) in world frame to 2D camera frame.
        """
        assert points3d.ndim == 2 and points3d.shape[1] == 3

        N = points3d.shape[0]
        points_world_h = torch.cat([points3d, torch.ones((N, 1), dtype=torch.float32)], dim=1)

        # World to camera
        points_cam = (self.w2c @ points_world_h.T).T  # (N, 4)

        x, y, z, _ = points_cam.T
        z = torch.clamp(z, min=1e-6)

        # Normalize by depth
        x_norm = x / z
        y_norm = y / z

        # Apply intrinsics
        u = self.fx * x_norm + self.cx
        v = self.fy * y_norm + self.cy

        points2d = torch.stack([u, v], dim=1)
        return points2d, z

    def project_to_world(self, points2d: torch.Tensor) -> torch.Tensor:
        """
        Backproject 2D pixel coordinates (u, v) into 3D world points using depth.
        """
        assert self.depth is not None, "Depth not set"

        points2d = points2d.to(torch.int64)
        u = torch.clamp(points2d[:, 0], 0, self.depth.shape[1] - 1)
        v = torch.clamp(points2d[:, 1], 0, self.depth.shape[0] - 1)

        depth = self.depth[v, u].unsqueeze(-1)

        valid_mask = depth.squeeze(-1) > 0
        if not valid_mask.any():
            return torch.zeros((0, 3), dtype=torch.float32)

        # Backproject to camera space
        x = (u - self.cx).unsqueeze(-1) * depth / self.fx
        y = (v - self.cy).unsqueeze(-1) * depth / self.fy
        z = depth

        points_cam = torch.cat([x, y, z, torch.ones_like(z)], dim=1)
        points_world_h = (self.c2w @ points_cam.T).T
        points_world = points_world_h[:, :3] / points_world_h[:, 3:4]

        return points_world

    #########################################################
    #################### Observe Agent ######################
    #########################################################

    def observe_agent(self, agent: Agent) -> ObservedState:
        """
        Observe the agent at a given timestep (default current) or entire trajectory.
        Returns:
            uv: (2,) for single step or (T,2) for full trajectory
            area: scalar or (T,)
        """
        
        pos = agent.trajectory.positions  # (T,3)
        size = agent.trajectory.sizes    # (T,)
        uv, _ = self.project_to_camera(pos)   # vectorized over T
        area = agent.project_area((pos, size), self.fx)

        return uv, area
    
    def observe_batch(self, batch: List[Agent]) -> ObservedState:
        """
        Observe a batch of agents.
        """
        uvs, areas = zip(*(self.observe_agent(agent) for agent in batch))
        return torch.stack(uvs), torch.stack(areas)

#########################################################
################## Environment ##########################
#########################################################
"""
Environment is what surrounds the Agents and Cameras. Cameras
can observe the environment and the Agents can move in the environment.
"""


@dataclass
class MeshEnvironment:
    # Environment is the world in which cameras live
    def __init__(self, mesh_fn):
        self.mesh = o3d.io.read_triangle_mesh(mesh_fn)  # or .ply, .stl, etc.
        self.mesh.compute_vertex_normals()

        # Get bounding box of the mesh
        self.min_bound, self.max_bound = self.extract_bounds()

    def extract_bounds(self):
        # Get the Axis-Aligned Bounding Box (AABB)
        aabb = self.mesh.get_axis_aligned_bounding_box()

        # Extract min and max bounds
        min_bound = aabb.get_min_bound()  # numpy array of shape (3,)
        max_bound = aabb.get_max_bound()  # numpy array of shape (3,)

        return torch.as_tensor(min_bound), torch.as_tensor(max_bound)

    # def constraint(self, state):
    #     """
    #     Constrain the state to the mesh. A state is a tuple of (position, size). 
    #     The position is a 3D point viewed originally from the 2D camera. 
        
    #     The constraint is therefore a factor of the source camera.

    #     Args:
    #         camera: Camera object
    #         state: State object
    #     """
    #     pos, size = state

    #     # Grab the bounds and move to device before clamping
    #     min_bound = self.min_bound.to(pos.device)
    #     max_bound = self.max_bound.to(pos.device)
    #     pos = torch.clamp(pos, min_bound, max_bound)

    #     return pos, size

    def distance_to_mesh(self, camera: Camera, radius=0.01, smooth=True):
        """
        Get the distance from the camera to the mesh.

        Args:
            camera: Camera object
            radius: Radius of the sphere to project the camera to the mesh
            smooth: Whether to smooth the depths

        Returns:
            mesh_depths: Depth values of the mesh
        """

        # Create a KDTree for the mesh
        mesh_points = np.asarray(self.mesh.vertices)
        tree = cKDTree(mesh_points)

        # Project depth indices to world space and back
        image_indices = torch.argwhere(camera.depth)[:, [1, 0]]
        world_points = camera.project_to_world(image_indices)
        _, depths = camera.project_to_camera(world_points)

        # Check if the depth is mapped correctly
        h, w = camera.depth.shape
        depth_patch = depths.reshape(h, w)
        assert array_nan_equal(camera.depth, depth_patch), "Depth mapping is incorrect"

        # Keep only valid world points
        mask = ~torch.isnan(world_points).any(dim=1)
        true_points, true_depths = world_points[mask], depths[mask]

        # Map depths to nearest mesh vertex
        _, indices = tree.query(true_points, k=1)
        sums = np.bincount(indices, weights=true_depths, minlength=len(mesh_points))
        counts = np.bincount(indices, minlength=len(mesh_points))
        mesh_depths = sums / np.maximum(counts, 1)
        mesh_depths[counts == 0] = np.nan

        if smooth:
            # Average neighbor depths (vectorized with list comprehension)
            neighbors = tree.query_ball_point(mesh_points, r=radius)
            mesh_depths = np.array(
                [np.nanmean(mesh_depths[n]) if n else np.nan for n in neighbors]
            )

        return mesh_depths

    def render_camera(self, camera: Camera) -> Tuple[np.ndarray, np.ndarray]:
        """
        Render the camera view of the mesh.
        """

        if camera.image is not None and camera.depth is not None:
            return camera.image, camera.depth

        # Create renderer
        renderer = o3d.visualization.rendering.OffscreenRenderer(
            camera.width, camera.height
        )
        renderer.scene.set_background([1, 1, 1, 1])  # white bg

        # Set material to use vertex colors
        material = o3d.visualization.rendering.MaterialRecord()
        material.shader = "defaultUnlit"  # or "defaultLit" if lighting desired

        # Add mesh with material
        renderer.scene.add_geometry("mesh", self.mesh, material)

        # Set up camera --> convert from torch to numpy for Open3D
        attrs = ['K', 'w2c']
        K, w2c = [np.asarray(getattr(camera, attr)) for attr in attrs] 
        renderer.setup_camera(K, w2c, camera.width, camera.height)

        # Render
        image = renderer.render_to_image()
        depth = renderer.render_to_depth_image(z_in_view_space=True)

        image = np.asarray(image)
        depth = np.asarray(depth)

        camera.set_view(
            image=image,
            depth=depth,  # depth is in world units
        )

        del renderer

        return image, depth


# ################################
# ########### Observer ###########
# ################################


# @dataclass
# class Observer:
#     def __init__(
#         self, mesh: MeshEnvironment, camera: Camera, observe_fx: Optional[str] = None
#     ):
#         """
#         Observer is a class that uses a camera projection to observe movement
#         on a mesh environment.
#         """

#         self.mesh = mesh
#         self.camera = camera
#         self.observe_fx = observe_fx

#     def __observe__(self, *args) -> ObservedState:
#         """
#         Observe from the camera frame.
#         """
#         if self.observe_fx is None:
#             return self.camera.project_to_world(args)
#         else:
#             return getattr(self.camera, self.observe_fx)(args)

#     def observe(self, batch: State) -> ObservedState:
#         agent = batch[0].shape[:-1]
#         # create the tuple of tensors
#         # obs_tuple = (torch.zeros(shp + (2,)), torch.zeros(shp + (1,)))
#         vmap_observe = vmap(self.__observe__, in_dims=0)(
#             *tuple(item.view(-1, item.shape[-1]) for item in batch)
#         )

#         return tuple(state.view(*agent, -1) for state in vmap_observe)

#########################################################
################# Movement generator ####################
#########################################################


@dataclass
class MovementGenerator:
    """
    MovementGenerator is a class that generates movement for a specified number 
    of agents. More or less generates a random walk
    """
    def __init__(
        self, 
        mesh: MeshEnvironment, 
        spatial_increment_params: dict = {
            "loc": torch.tensor([0., 0., 0.]),
            "scale": 0.02
        },
        size_params: dict = {
            "low": torch.tensor([0.1]), 
            "high": torch.tensor([10.0])
        }
    ):
        """
        MovementGenerator is a class that generates movement for a set of agents.
        """

        self.mesh = mesh

        assert isinstance(spatial_increment_params, dict), "Spatial increment parameters must be a dictionary"
        assert isinstance(size_params, dict), "Size parameters must be a dictionary"

        # Check that the spatial increment parameters have the correct shape
        n_dims = [2, 3] # 2D or 3D
        assert spatial_increment_params["loc"].shape[-1] in n_dims, (
            f"Spatial increment parameters must have shape (..., {n_dims}), but got {spatial_increment_params['loc'].shape}"
        )

        # Set the spatial increment and size range
        self.spatial_increment_params = spatial_increment_params
        self.size_params = size_params

    #########################################################
    #################### Distributions ######################
    #########################################################

    @property
    def size_dist(self) -> pyro.distributions.Distribution:
        """
        Size distribution.
        """
        return dist.Uniform(**self.size_params).to_event(1)

    @property
    def spatial_increment_dist(self) -> pyro.distributions.Distribution:
        """
        Spatial increment distribution.
        """

        # Parameters for the multivariate distribution --> to_event(1) treats the col dimension as a single "multivariate event"
        # Therefore each sample will by a vector of size 2
        return dist.Normal(**self.spatial_increment_params).to_event(1)

    #########################################################
    #################### Sampling ###########################
    #########################################################

    def sample_sizes(self, name: str, n_agents: int = 1) -> torch.Tensor:
        """
        Sample sizes for the agents.
        """
        # Create n_agents independent samples
        return pyro.sample(name, self.size_dist.expand([n_agents]))

    def sample_spatial_increment(self, name: str, n_agents: int = 1) -> torch.Tensor:
        """
        Sample spatial increments for the agents.
        """
        # Create n_agents independent samples
        return pyro.sample(name, self.spatial_increment_dist.expand([n_agents]))

    #########################################################
    #################### Initialization #####################
    #########################################################

    def init_states(self, camera: Camera, n_agents: int = 1) -> State:
        """
        Initialize the positions of the agents in the mesh.
        """

        # Sample from world point distribution, which is restricted to points viewed from the camera
        positions = camera.sample_world_points()

        # Sample sizes with provided prior
        sizes = self.sample_sizes(
            name=f"init_size", 
            n_agents=n_agents
        )
        
        return positions, sizes

    def next_state(self, state: State, step: int) -> State:
        """
        Generate the next state of the agent.
        """

        # Extract components
        position, size = state
        n_agents = position.shape[0]

        # Sample spatial increment (independent for each agent) and update position
        spatial_increment = self.sample_spatial_increment(
            name=f"spatial_step_{step}", 
            n_agents=n_agents
        )

        position = position + spatial_increment

        return position, size

    def generate_trajectory(self, init_state: State, n_steps: int): # -> StateTrajectory:
        """
        Generate a trajectory of states.
        """
        state = init_state
        # trajectory = [state]
        trajectory = []

        # We count the initial state as one of the steps (TLB CHECK IF THIS IS RIGHT)
        for step in tqdm(range(n_steps), desc="Generating trajectory"):
            state = self.next_state(state, step)
            trajectory.append(state)

        # Unpack and convert to tensors
        positions, sizes = map(torch.stack, zip(*trajectory))
        return positions, sizes