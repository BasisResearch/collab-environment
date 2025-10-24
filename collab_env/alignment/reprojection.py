import numpy as np
import torch
from torch.func import vmap
from typing import Optional, Tuple, TypeAlias
from dataclasses import dataclass
import open3d as o3d
from scipy.spatial import cKDTree
import platform
import itertools
from collab_env.utils.utils import array_nan_equal

#########################################################
####### TLB CLEAN UP THESE BUT PUT HERE FOR NOW #########
#########################################################


def filter_coords(df_tracks, coord_cols=["u", "v"], window=3, std_threshold=3):
    """
    Filter outlier coordinates based on rolling statistics for a single track

    Args:
        track_data: DataFrame containing a single trajectory
        coord_cols: List of coordinate column names to filter
        window: Size of rolling window for statistics
        std_threshold: Number of standard deviations for outlier detection

    Returns:
        DataFrame with outliers replaced by rolling mean
    """

    # Copy as to not modify the original dataframe
    df_tracks = df_tracks.copy()

    # Calculate rolling statistics
    rolling_mean = (
        df_tracks[coord_cols].rolling(window=window, min_periods=1, center=True).mean()
    )

    rolling_std = (
        df_tracks[coord_cols].rolling(window=window, min_periods=1, center=True).std()
    )

    # Identify outliers
    z_scores = np.abs((df_tracks[coord_cols] - rolling_mean) / rolling_std)
    outliers = z_scores > std_threshold

    # Replace outliers with rolling mean
    for col in coord_cols:
        df_tracks.loc[outliers[col], col] = rolling_mean.loc[outliers[col], col]

    return df_tracks[coord_cols]


def smooth_coords(df_tracks, coord_cols=["u", "v"], window=3):
    """
    Smooth coordinates over time using a rolling average for a single track

    Args:
        track_data: DataFrame containing a single trajectory
        coord_cols: List of coordinate column names to smooth
        window: Size of rolling window for smoothing

    Returns:
        DataFrame with smoothed coordinates
    """
    # Apply smoothing
    smoothed = (
        df_tracks[coord_cols].rolling(window=window, min_periods=1, center=True).mean()
    )

    return smoothed


### TLB putting here for now but likely a better place or could refactor
def get_depths_on_mesh(camera, mesh, radius=0.01, smooth=True):
    # Grab the mesh points and create a KDTree
    mesh_points = np.asarray(mesh.vertices)
    tree = cKDTree(mesh_points)

    height, width = camera.depth.shape

    # Get indices of image
    # image_indices = np.argwhere(camera.depth)
    # image_indices = image_indices[:, [1, 0]]
    image_indices = np.array(
        [
            np.array(pair[::-1])
            for pair in itertools.product(range(height), range(width))
        ]
    )
    # Project image indices to world --> then project back to 2d
    world_points = camera.project_to_world(image_indices)
    _, depths = camera.project_to_camera(world_points)

    # Reshape the points to the bounding box shape
    depth_patch = depths.reshape(height, width)

    # Check if the depths are the same
    same_depths = array_nan_equal(camera.depth, depth_patch)

    print("Correctly mapped depths: ", same_depths)

    # Grab the world points that were found
    true_indices = np.where(~np.isnan(world_points).any(1))[0]
    true_points = world_points[true_indices]
    true_depths = depths[true_indices]  # Trim depths to those found

    # Find the nearest mesh point for each true point
    _, indices = tree.query(true_points, k=1)

    sums = np.bincount(indices, weights=true_depths, minlength=len(mesh_points))
    counts = np.bincount(indices, minlength=len(mesh_points))
    mesh_depths = sums / np.maximum(counts, 1)  # avoid division by zero
    mesh_depths[counts == 0] = np.nan  # keep NaN for no matches

    if smooth:
        # Find neighbors within the radius for all mesh points
        neighbors_list = tree.query_ball_point(mesh_points, r=radius)

        # Preallocate smoothed array
        smoothed_depths = np.full_like(mesh_depths, np.nan)

        # Average over neighbors (ignore NaNs)
        for i, neighbors in enumerate(neighbors_list):
            smoothed_depths[i] = np.nanmean(mesh_depths[neighbors])

        mesh_depths = smoothed_depths

    return mesh_depths


#########################################################
################## Environment ##########################
#########################################################
"""
Camera represents a view within the environment. It contains a
set of transformations that relate the world points to the image
"""


@dataclass
class Camera:
    # A camera is a set of transformations that relates the world points to the camera
    def __init__(self, K: np.ndarray, c2w: np.ndarray, width: int, height: int):
        self.K: np.ndarray = K
        self.c2w: np.ndarray = c2w
        self.width: int = width
        self.height: int = height
        self.image: Optional[np.ndarray] = None
        self.depth: Optional[np.ndarray] = None

    def set_view(self, image: np.ndarray, depth: np.ndarray) -> None:
        self.image = image
        self.depth = depth

    def get_view(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        return self.image, self.depth

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
        return np.linalg.inv(self.c2w)

    #########################################################
    ################# Camera projections ####################
    #########################################################

    def project_to_camera(self, points3d):
        """
        Project a 3D point (x,y,z) in the world frame to
        a 2D point (u,v) in the camera frame.

        Args:
            points3d: 3D points in the world frame (N, 3)

        Returns:
            points2d: 2D points in the camera frame (N, 2)
        """

        # Convert to homogeneous coordinates (N, 4) --> enables matrix multiplication
        points_world_homogenous = np.concatenate(
            [points3d, np.ones((points3d.shape[0], 1))], axis=1
        )

        # Step 2: Project to camera space
        points_cam = self.w2c @ points_world_homogenous.T  # (4,4) @ (4,N) = (4,N)

        # Step 3: Perspective division (to normalize by z)
        x, y, z, _ = points_cam

        # Avoid division by zero
        z = np.clip(z, 1e-6, None)

        # Normalize by depth
        x_norm = x / z
        y_norm = y / z

        # Step 4: Apply camera intrinsics (assuming self.K is a 3x3 intrinsic matrix)
        u = self.fx * x_norm + self.cx
        v = self.fy * y_norm + self.cy

        # Step 5: Stack 2D points
        points2d = np.stack([u, v], axis=1)  # (N, 2)

        # Return 2D points and associated depths
        return points2d, z

    def project_to_world(self, points2d):
        """
        Project 2D pixel coordinates (u, v) into 3D world points using stored depth.

        Args:
            points2d: (N, 2) array of (u, v) pixel coordinates where
                    u = column (horizontal),
                    v = row (vertical)

        Returns:
            points3d_world: (N, 3) array of 3D points in world coordinates
        """
        points2d = np.asarray(points2d).astype(int)
        u = np.clip(points2d[:, 0], 0, self.depth.shape[1] - 1)  # col index (width)
        v = np.clip(points2d[:, 1], 0, self.depth.shape[0] - 1)  # row index (height)

        depth = self.depth[v, u].reshape(
            -1, 1
        )  # note depth indexed by [row, col] = [v, u]

        valid_mask = depth.squeeze() > 0
        if not np.any(valid_mask):
            return np.zeros((0, 3))  # or raise exception / return NaNs

        # Backproject to camera space
        x = (u - self.cx)[:, None] * depth / self.fx
        y = (v - self.cy)[:, None] * depth / self.fy
        z = depth

        points_cam = np.concatenate([x, y, z, np.ones_like(z)], axis=1)
        points_world_hom = (self.c2w @ points_cam.T).T
        points_world = points_world_hom[:, :3] / points_world_hom[:, 3:4]

        return points_world


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

        return torch.tensor(min_bound), torch.tensor(max_bound)

    def constraint(self, state):
        pos, size = state

        # Grab the bounds and move to device before clamping
        min_bound = self.min_bound.to(pos.device)
        max_bound = self.max_bound.to(pos.device)
        pos = torch.clamp(pos, min_bound, max_bound)

        return pos, size

    def render_camera(self, camera: Camera) -> Tuple[np.ndarray, np.ndarray]:
        """
        Render the camera view of the mesh.
        """

        if camera.image is not None and camera.depth is not None:
            return camera.image, camera.depth

        # Use on-screen rendering for macOS, offscreen for others
        if platform.system() == "Darwin":  # macOS
            image, depth = self._render_camera_onscreen(camera)
        else:
            image, depth = self._render_camera_offscreen(camera)

        camera.set_view(
            image=image,
            depth=depth,  # depth is in world units
        )

        return image, depth

    def _render_camera_offscreen(self, camera: Camera) -> Tuple[np.ndarray, np.ndarray]:
        """Offscreen rendering implementation (for Linux/Windows)"""
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

        # Set up camera
        renderer.setup_camera(camera.K, camera.w2c, camera.width, camera.height)

        # Render
        image = renderer.render_to_image()
        depth = renderer.render_to_depth_image(z_in_view_space=True)

        image = np.asarray(image)
        depth = np.asarray(depth)

        del renderer

        return image, depth

    def _render_camera_onscreen(self, camera: Camera) -> Tuple[np.ndarray, np.ndarray]:
        """On-screen rendering implementation for macOS"""
        # Create a non-blocking visualizer
        vis = o3d.visualization.Visualizer()
        vis.create_window(
            window_name="Render",
            width=camera.width,
            height=camera.height,
            visible=False,  # Keep window hidden
        )

        # add white background
        vis.get_render_option().background_color = [1, 1, 1]

        # Add mesh
        vis.add_geometry(self.mesh)

        # Get view control and set camera parameters
        view_ctrl = vis.get_view_control()

        # Convert camera parameters to Open3D format
        cam_params = o3d.camera.PinholeCameraParameters()
        cam_params.intrinsic = o3d.camera.PinholeCameraIntrinsic(
            camera.width,
            camera.height,
            camera.K[0, 0],
            camera.K[1, 1],
            camera.K[0, 2],
            camera.K[1, 2],
        )

        # Set extrinsic matrix (world to camera transform)
        cam_params.extrinsic = camera.w2c

        # Apply camera parameters
        view_ctrl.convert_from_pinhole_camera_parameters(
            cam_params, allow_arbitrary=True
        )

        # Update geometry and render
        vis.poll_events()
        vis.update_renderer()

        # Capture rendered image
        image = vis.capture_screen_float_buffer(do_render=True)
        image = np.asarray(image)
        image = (image * 255).astype(np.uint8)

        # Capture depth buffer (returns depth in view space by default)
        depth = vis.capture_depth_float_buffer(do_render=False)
        depth = np.asarray(depth)

        # The onscreen visualizer returns 0 for background pixels (no geometry)
        # while offscreen returns inf. Map 0 -> inf for consistency.
        # Note: valid geometry will have positive depth values in view space
        depth[depth == 0.0] = np.inf

        # Clean up
        vis.destroy_window()

        return image, depth


################################
########### Observer ###########
################################
"""

"""

State: TypeAlias = Tuple[torch.Tensor, torch.Tensor]  # positions, sizes
ObservedState: TypeAlias = Tuple[
    torch.Tensor, torch.Tensor
]  # projected centroids, projected areas
StateTrajectory: TypeAlias = State  # each tensor has extra dimension


@dataclass
class Observer:
    def __init__(
        self, mesh: MeshEnvironment, camera: Camera, observe_fx: Optional[str] = None
    ):
        """
        Observer is a class that uses a camera projection to observe movement
        on a mesh environment.
        """

        self.mesh = mesh
        self.camera = camera
        self.observe_fx = observe_fx

    def __observe__(self, *args) -> ObservedState:
        """
        Observe from the camera frame.
        """
        if self.observe_fx is None:
            return self.camera.project_to_world(args)
        else:
            return getattr(self.camera, self.observe_fx)(args)

    def observe(self, batch: State) -> ObservedState:
        agent = batch[0].shape[:-1]
        # create the tuple of tensors
        # obs_tuple = (torch.zeros(shp + (2,)), torch.zeros(shp + (1,)))
        vmap_observe = vmap(self.__observe__, in_dims=0)(
            *tuple(item.view(-1, item.shape[-1]) for item in batch)
        )

        return tuple(state.view(*agent, -1) for state in vmap_observe)


################################
########### Agent ##############
################################

"""
Agent is an object that moves within the environment. It
starts from an initial position.
"""


@dataclass
class Agent:
    def __init__(self, position: torch.Tensor, size: torch.Tensor):
        self.position = position
        self.size = size

    @property
    def area(self):
        return np.prod(self.size)

    @property
    def state(self) -> State:
        return self.position, self.area

    @state.setter
    def state(self, state: State):
        self.position, self.size = state


#     def observe(self, batch: State) -> ObservedState:
#         return self.camera.project_to_world(batch)


def bbox_to_coords(bbox, method="bottom_center"):
    x1, y1, x2, y2 = map(int, bbox)

    u = (x1 + x2) // 2
    v = y2 - 1 if method == "bottom_center" else (y1 + y2) // 2
    uv = np.stack([u, v])
    return uv


#########################################################
################# Movement generator ####################
#########################################################


@dataclass
class DynamicMovementGenerator:
    def __init__(self, mesh: MeshEnvironment, spatial_increment: torch.Tensor):
        self.mesh = mesh
        assert spatial_increment.shape[-1] == 2, (
            f"Spatial increment parameters must have shape (..., 2), but got {spatial_increment.shape}"
        )
        self.spatial_increment = spatial_increment


#     def next_state(self, state: State, ind: int) -> State:
#         # extract components
#         position, size = state
#         x_3d, y_3d, z_3d = position[..., 0], position[..., 1], position[..., 2]

#         state_constrained = self.mesh.constraint(state)
#         assert torch.allclose(state[0], state_constrained[0]), \
#             f"Environment constraints violated: state={state}, state_constrained={state_constrained}"

#         # sample independent increments in x and z
#         # Sample independent increments for each agent (dim=-2)
#         batch_shape = pos.shape[:-1]  # Get all dimensions except the last one
#         pos_increment = self.spatial_increment_parameters * pyro.sample(
#             f"pos_increment_{ind}",
#             dist.Normal(0, 1).expand(batch_shape + self.spatial_increment_parameters.shape).to_event(2)
#         )
#         x_3d_new = x_3d + pos_increment[..., 0]
#         z_3d_new = z_3d + pos_increment[..., 1]
#         y_3d_new = y_3d.reshape(x_3d_new.shape)

#         pos_new = torch.stack([x_3d_new, y_3d_new, z_3d_new], dim=-1)

#         # sample independent increments in angles
#         # Sample independent increments for each agent (dim=-2)
#         angle_increment = self.angle_increment_parameters * pyro.sample(
#             f"angle_increment_{ind}",
#             dist.Normal(0, 1).expand(batch_shape + self.angle_increment_parameters.shape).to_event(2)
#         )
#         angle_new = angle + angle_increment
#         return self.env.constrain((pos_new, size, angle_new))

#     def generate_trajectory(self, init_state: State, n_steps: int) -> StateTrajectory:
#         state = init_state
#         # trajectory = [state]
#         trajectory = []
#         for ind in range(n_steps):
#             state = self.next_state(state, ind)
#             trajectory.append(state)
#         return tuple(torch.stack(t, dim=0) for t in zip(*trajectory))
