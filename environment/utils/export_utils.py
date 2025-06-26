# Copyright 2022 the Regents of the University of California, Nerfstudio Team and contributors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Export utils such as structs, point cloud generation, and rendering code.
"""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING, List, Tuple, Union
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from rich.progress import BarColumn, Progress, TaskProgressColumn, TextColumn, TimeRemainingColumn
from torch import Tensor

from nerfstudio.cameras.cameras import Cameras
from nerfstudio.pipelines.base_pipeline import Pipeline
from nerfstudio.utils.rich_utils import CONSOLE, ItersPerSecColumn
from nerfstudio.exporter.tsdf_utils import TSDF

from nerfstudio.models.splatfacto import SplatfactoModel

if TYPE_CHECKING:
    # Importing open3d can take ~1 second, so only do it below if we actually
    # need it.
    import open3d as o3d

def render_trajectory(
    pipeline: Pipeline,
    cameras: Cameras,
    rgb_output_name: str,
    depth_output_name: str,
    rendered_resolution_scaling_factor: float = 1.0,
    disable_distortion: bool = False,
    return_rgba_images: bool = False,
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """Helper function to create a video of a trajectory.

    Args:
        pipeline: Pipeline to evaluate with.
        cameras: Cameras to render.
        rgb_output_name: Name of the RGB output.
        depth_output_name: Name of the depth output.
        rendered_resolution_scaling_factor: Scaling factor to apply to the camera image resolution.
        disable_distortion: Whether to disable distortion.
        return_rgba_images: Whether to return RGBA images (default RGB).

    Returns:
        List of rgb images, list of depth images.
    """
    images = []
    depths = []
    cameras.rescale_output_resolution(rendered_resolution_scaling_factor)

    progress = Progress(
        TextColumn(":cloud: Computing rgb and depth images :cloud:"),
        BarColumn(),
        TaskProgressColumn(show_speed=True),
        ItersPerSecColumn(suffix="fps"),
        TimeRemainingColumn(elapsed_when_finished=True, compact=True),
    )

    # Splatfacto doesn't support camera ray bundles, so we use camera outputs
    use_camera_outputs = isinstance(pipeline.model, SplatfactoModel)

    with progress:
        for camera_idx in progress.track(range(cameras.size), description=""):

            if use_camera_outputs:
                with torch.no_grad():
                    outputs = pipeline.model.get_outputs_for_camera(camera=cameras[camera_idx:camera_idx+1])
            else:
                camera_ray_bundle = cameras.generate_rays(
                    camera_indices=camera_idx, disable_distortion=disable_distortion
                ).to(pipeline.device)
                with torch.no_grad():
                    outputs = pipeline.model.get_outputs_for_camera_ray_bundle(camera_ray_bundle)
            
            if rgb_output_name not in outputs:
                CONSOLE.rule("Error", style="red")
                CONSOLE.print(f"Could not find {rgb_output_name} in the model outputs", justify="center")
                CONSOLE.print(f"Please set --rgb_output_name to one of: {outputs.keys()}", justify="center")
                sys.exit(1)
            if depth_output_name not in outputs:
                CONSOLE.rule("Error", style="red")
                CONSOLE.print(f"Could not find {depth_output_name} in the model outputs", justify="center")
                CONSOLE.print(f"Please set --depth_output_name to one of: {outputs.keys()}", justify="center")
                sys.exit(1)
            if return_rgba_images and "rgba" in outputs:
                image = pipeline.model.get_rgba_image(outputs, rgb_output_name)
            else:
                image = outputs[rgb_output_name]
            images.append(image.cpu().numpy())
            depths.append(outputs[depth_output_name].cpu().numpy())
    return images, depths

def export_tsdf_mesh(
    pipeline: Pipeline,
    output_dir: Path,
    downscale_factor: int = 2,
    depth_output_name: str = "depth",
    rgb_output_name: str = "rgb",
    resolution: Union[int, List[int]] = field(default_factory=lambda: [256, 256, 256]),
    batch_size: int = 10,
    use_bounding_box: bool = True,
    bounding_box_min: Tuple[float, float, float] = (-1.0, -1.0, -1.0),
    bounding_box_max: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    refine_mesh_using_initial_aabb_estimate: bool = False,
    refinement_epsilon: float = 1e-2,
) -> None:
    """Export a TSDF mesh from a pipeline.

    Args:
        pipeline: The pipeline to export the mesh from.
        output_dir: The directory to save the mesh to.
        downscale_factor: Downscale factor for the images.
        depth_output_name: Name of the depth output.
        rgb_output_name: Name of the RGB output.
        resolution: Resolution of the TSDF volume or [x, y, z] resolutions individually.
        batch_size: How many depth images to integrate per batch.
        use_bounding_box: Whether to use a bounding box for the TSDF volume.
        bounding_box_min: Minimum coordinates of the bounding box.
        bounding_box_max: Maximum coordinates of the bounding box.
        refine_mesh_using_initial_aabb_estimate: Whether to refine the TSDF using the initial AABB estimate.
        refinement_epsilon: Epsilon for refining the TSDF. This is the distance in meters that the refined AABB/OBB will
            be expanded by in each direction.
    """

    device = pipeline.device

    assert pipeline.datamanager.train_dataset is not None
    dataparser_outputs = pipeline.datamanager.train_dataset._dataparser_outputs

    # initialize the TSDF volume
    if not use_bounding_box:
        aabb = dataparser_outputs.scene_box.aabb
    else:
        aabb = torch.tensor([bounding_box_min, bounding_box_max])
    if isinstance(resolution, int):
        volume_dims = torch.tensor([resolution] * 3)
    elif isinstance(resolution, List):
        volume_dims = torch.tensor(resolution)
    else:
        raise ValueError("Resolution must be an int or a list.")
    tsdf = TSDF.from_aabb(aabb, volume_dims=volume_dims)
    # move TSDF to device
    tsdf.to(device)

    cameras = dataparser_outputs.cameras
    # we turn off distortion when populating the TSDF
    color_images, depth_images = render_trajectory(
        pipeline,
        cameras,
        rgb_output_name=rgb_output_name,
        depth_output_name=depth_output_name,
        rendered_resolution_scaling_factor=1.0 / downscale_factor,
        disable_distortion=True,
        return_rgba_images=True,
    )

    # TODO: this can be done better by removing transparent points from the TSDF
    color_images = [x[..., :3] for x in color_images]

    # camera extrinsics and intrinsics
    c2w: Float[Tensor, "N 3 4"] = cameras.camera_to_worlds.to(device)
    # make c2w homogeneous
    c2w = torch.cat([c2w, torch.zeros(c2w.shape[0], 1, 4, device=device)], dim=1)
    c2w[:, 3, 3] = 1
    K: Float[Tensor, "N 3 3"] = cameras.get_intrinsics_matrices().to(device)
    color_images = torch.tensor(np.array(color_images), device=device).permute(0, 3, 1, 2)  # shape (N, 3, H, W)
    depth_images = torch.tensor(np.array(depth_images), device=device).permute(0, 3, 1, 2)  # shape (N, 1, H, W)

    CONSOLE.print("Integrating the TSDF")
    for i in range(0, len(c2w), batch_size):
        tsdf.integrate_tsdf(
            c2w[i : i + batch_size],
            K[i : i + batch_size],
            depth_images[i : i + batch_size],
            color_images=color_images[i : i + batch_size],
        )

    CONSOLE.print("Computing Mesh")
    mesh = tsdf.get_mesh()

    if refine_mesh_using_initial_aabb_estimate:
        CONSOLE.print("Refining the TSDF based on the Mesh AABB")

        # Compute the AABB of the mesh and use it to initialize a new TSDF
        vertices_min = torch.min(mesh.vertices, dim=0).values - refinement_epsilon
        vertices_max = torch.max(mesh.vertices, dim=0).values + refinement_epsilon
        aabb = torch.stack([vertices_min, vertices_max]).cpu()
        tsdf = TSDF.from_aabb(aabb, volume_dims=volume_dims)
        # move TSDF to device
        tsdf.to(device)

        CONSOLE.print("Integrating the updated TSDF")
        for i in range(0, len(c2w), batch_size):
            tsdf.integrate_tsdf(
                c2w[i : i + batch_size],
                K[i : i + batch_size],
                depth_images[i : i + batch_size],
                color_images=color_images[i : i + batch_size],
            )

        CONSOLE.print("Computing the updated Mesh")
        mesh = tsdf.get_mesh()

    CONSOLE.print("Saving TSDF Mesh")
    tsdf.export_mesh(mesh, filename=str(output_dir / "tsdf_mesh.ply"))