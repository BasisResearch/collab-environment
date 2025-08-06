import numpy as np
import torch
from torch.func import vmap
from typing import Optional, Tuple, TypeAlias
from dataclasses import dataclass
import open3d as o3d

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
    def __init__(self, K, c2w, width, height):
        self.K = K
        self.c2w = c2w
        self.width = width
        self.height = height
        self.image = None
        self.depth = None

    def set_view(self, image: np.ndarray, depth: np.ndarray):
        self.image = image
        self.depth = depth

    def get_view(self):
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
        points_world_homogenous = np.concatenate([points3d, np.ones((points3d.shape[0], 1))], axis=1)

        # Step 2: Project to camera space
        points_cam = self.w2c @ points_world_homogenous.T # (4,4) @ (4,N) = (4,N)

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

        depth = self.depth[v, u].reshape(-1, 1)  # note depth indexed by [row, col] = [v, u]

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

        if camera.image is not None:
            return camera.get_view()
        
        # Create renderer
        renderer = o3d.visualization.rendering.OffscreenRenderer(camera.width, camera.height)
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

        camera.set_view(
            image=image,
            depth=depth # depth is in world units
        )

        del renderer

        return image, depth

################################
########### Observer ###########
################################
"""

"""

State: TypeAlias = Tuple[torch.Tensor, torch.Tensor] # positions, sizes
ObservedState: TypeAlias = Tuple[torch.Tensor, torch.Tensor] # projected centroids, projected areas
StateTrajectory: TypeAlias = State # each tensor has extra dimension

@dataclass
class Observer:
    def __init__(self, mesh: MeshEnvironment, camera: Camera, observe_fx: Optional[str] = None):
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
        vmap_observe = vmap(self.__observe__, in_dims = 0)(*tuple(item.view(-1, item.shape[-1]) for item in batch))
        
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

def bbox_to_world(camera, bbox, method="bottom_center", bottom_fraction=0.05):
    """
    Extract a 3D world point from a bounding box using the given method.

    Args:
        camera (Camera): Camera object with intrinsics and depth.
        bbox (tuple): (x1, y1, x2, y2) in pixel coordinates.
        method (str): One of ['bottom_center', 'center', 'median', 'consistent_bottom'].

    Returns:
        tuple:
            - point3d (np.ndarray or None): (3,) world coordinate or None if invalid.
            - bbox_size (tuple): (width, height) of the bounding box in pixels.
    """
    x1, y1, x2, y2 = map(int, bbox)
    H, W = camera.depth.shape

    # Clamp bbox to image bounds
    x1, x2 = np.clip([x1, x2], 0, W - 1)
    y1, y2 = np.clip([y1, y2], 0, H - 1)

    bbox_width = x2 - x1
    bbox_height = y2 - y1
    bbox_size = (bbox_height, bbox_width)

    if bbox_width <= 0 or bbox_height <= 0:
        return None, bbox_size

    if method in ["bottom_center", "center"]:
        u = (x1 + x2) // 2
        v = y2 - 1 if method == "bottom_center" else (y1 + y2) // 2
        points3d = camera.project_to_world(np.array([[u, v]]))
        return (points3d[0] if len(points3d) else None), bbox_size

    if method == "median":
        # Generate (u, v) pixel grid inside bbox
        u, v = np.meshgrid(np.arange(x1, x2), np.arange(y1, y2))
        pixels = np.stack([u.ravel(), v.ravel()], axis=1)  # (N, 2), (u, v)
        points3d = camera.project_to_world(pixels)
        if len(points3d) == 0:
            return None, bbox_size
        median_idx = np.argsort(points3d[:, 2])[len(points3d) // 2]
        return points3d[median_idx], bbox_size

    if method == "consistent_bottom":
        bottom_rows = max(int(bottom_fraction * bbox_height), 1)
        v_start = max(y2 - bottom_rows, y1)
        patch = camera.depth[v_start:y2, x1:x2]
        if patch.size == 0:
            return None, bbox_size
        var_per_col = np.var(patch, axis=0)
        best_col = np.argmin(var_per_col)
        u = x1 + best_col
        v = y2 - 1
        points3d = camera.project_to_world(np.array([[u, v]]))
        return (points3d[0] if len(points3d) else None), bbox_size

    raise ValueError(f"Unknown method '{method}'")

#########################################################
################# Movement generator ####################
#########################################################

# @dataclass
# class DynamicMovementGenerator:
#     def __init__(self, mesh: MeshEnvironment, spatial_increment: torch.Tensor):
#         self.mesh = mesh
#         assert spatial_increment.shape[-1] == 2, \
#             f"Spatial increment parameters must have shape (..., 2), but got {spatial_increment.shape}"
#         self.spatial_increment = spatial_increment
    
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