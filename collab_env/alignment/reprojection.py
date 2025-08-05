import numpy as np
import torch
from torch.func import vmap
from typing import Optional, Tuple, TypeAlias
from dataclasses import dataclass
import open3d as o3d

@dataclass
class Camera:
    # A camera is a set of transformations that relates the world points to the camera
    def __init__(self, K, R, t):
        self.K = K
        self.R = R
        self.t = t

        self._set_c2w()
    
    def _set_c2w(self):
        c2w = np.concatenate([self.R, self.t], axis=1)
        c2w = np.vstack([c2w, np.array([0, 0, 0, 1])]) # 4x4
        self.c2w = c2w

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
    def c2w(self):
        return self.c2w

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

        return np.stack([u, v], axis=1)  # (N, 2)

    def project_to_world(self, points2d, depth):
        """
        Project 2D image points (u, v) with depth into 3D world coordinates.

        Args:
            points2d: (N, 2) array of 2D image pixel coordinates (u, v)
            depth: (N, 1) or (N,) array of depths corresponding to each point

        Returns:
            points3d_world: (N, 3) array of 3D points in world coordinates
        """
        # Ensure depth is column vector
        depth = depth.reshape(-1, 1)  # (N, 1)

        u = points2d[:, 0]
        v = points2d[:, 1]

        # Step 1: Unproject to camera space (normalized)
        x = (u - self.cx) * depth / self.fx
        y = (v - self.cy) * depth / self.fy
        z = depth

        # Step 2: Convert to homogeneous coordinates (camera space)
        points_cam = np.stack([x, y, z, np.ones_like(z)], axis=1)  # (N, 4)

        # Step 3: Apply camera-to-world transformation (returns homogeneous coordinates)
        points_world_homogenous = (self.c2w @ points_cam.T).T  # (4, N).T -> (N, 4)

        # Step 4: Return 3D world coordinates
        points_world = points_world_homogenous[:, :3] / points_world_homogenous[:, 3:4]  # handle non-unity w
        return points_world

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


################################
########### Observer ###########
################################
"""

"""

State: TypeAlias = Tuple[torch.Tensor, torch.Tensor] # positions, sizes, angles
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