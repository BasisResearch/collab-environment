import pandas as pd
from typing import Optional, Union
import numpy as np
import matplotlib.pyplot as plt

import pyvista as pv
import open3d as o3d

######################################################################
################## PyVista Default Arguments #########################
######################################################################

# Arguments for what part of the mesh/pcd point data to visualize
MESH_KWARGS = {
    "scalars": "RGB",
    "rgb": True,
}

# Arguments for controlling position of the visualization camera / lighting
VIZ_KWARGS = {
    "position": (2, 2, 1),
    "focal_point": (0, 0, 0),
    "view_up": (0, 0, 1),
    "azimuth": 235,
    "elevation": 15,
    "lighting": [
        {"position": (10, 10, 10), "intensity": 0.8},
        {"position": (-10, -10, 10), "intensity": 0.4},
        {"position": (0, 0, -10), "intensity": 0.2},
    ],
}

# Arguments for controlling the camera frustum visualization
CAMERA_KWARGS = {
    "scale": 0.02,
    "aspect_ratio": 1.33,
    "fov": 60,
    "line_width": 1,
    "opacity": 0.6,
    "n_poses": 3,
    "color": "red",
}

LINE_KWARGS = {
    "line_width": 6,
    "opacity": 1.0,
}

######################################################################
################## PyVista Visualization Functions ###################
######################################################################


def visualize_splat(
    mesh: Union[str, pv.DataObject],
    aligned_cameras: Optional[list[np.ndarray]] = None,
    mesh_kwargs: dict = {},
    camera_kwargs: dict = {},
    viz_kwargs: dict = {},
    out_fn: Optional[str] = None,
) -> pv.Plotter:
    """
    Visualize point cloud with camera frustums using PyVista

    Args:
        ply_path: Path to the ply file
        aligned_poses: List of 4x4 transformation matrices
    """

    # Load mesh if path is provided --> annoying fix to linting
    mesh_data: pv.DataObject
    if isinstance(mesh, str):
        mesh_data = pv.read(mesh)
    else:
        mesh_data = mesh

    # Create PyVista plotter
    plotter = pv.Plotter(
        window_size=viz_kwargs.get("window_size", (3000, 3000)),
    )

    # Cast to DataSet for add_mesh compatibility
    if isinstance(mesh_data, pv.DataSet):
        plotter.add_mesh(mesh_data, **mesh_kwargs)
    else:
        # Handle case where DataObject is not a DataSet
        raise TypeError(f"Expected DataSet, got {type(mesh_data)}")

    # Remove scale from kwargs
    if aligned_cameras is not None:
        n_poses = camera_kwargs.pop("n_poses", 3)

        # Create and add camera frustums for every third pose
        for i in range(0, len(aligned_cameras), n_poses):
            pose = aligned_cameras[i]

            # Make a camera frustum
            frustum, axes = create_camera_frustum_pyvista(
                pose,
                scale=camera_kwargs.get("scale", 0.02),
                aspect_ratio=camera_kwargs.get("aspect_ratio", 1.33),
                fov=camera_kwargs.get("fov", 60),
                show_axes=camera_kwargs.get("show_axes", False),
            )

            if "color" not in camera_kwargs:
                cmap = plt.cm.get_cmap("viridis")
                camera_kwargs["color"] = cmap(i / len(aligned_cameras))[
                    :3
                ]  # Get RGB from colormap

            # Remove non-pyvista kwargs
            frustum_kwargs = camera_kwargs.copy()
            remove_kwargs = ["scale", "aspect_ratio", "fov", "n_poses", "show_axes"]
            for key in remove_kwargs:
                frustum_kwargs.pop(key, None)

            # Add to plotter with different color for each camera
            plotter.add_mesh(frustum, **frustum_kwargs)

            if axes is not None:
                axis_colors = ["red", "green", "blue"]
                for axis, color in zip(axes, axis_colors):
                    plotter.add_mesh(axis, color=color)

    # Set camera parameters
    for key, value in viz_kwargs.items():
        # Skip non-camera parameters
        if key in ["window_size", "lighting", "show_viz_camera"]:
            continue

        # Direct assignment like your azimuth/elevation example
        if hasattr(plotter.camera, key):
            setattr(plotter.camera, key, value)

    # Set lighting
    for light in viz_kwargs.get("lighting", []):
        plotter.add_light(pv.Light(**light))

    if out_fn is not None:
        plotter.screenshot(
            filename=out_fn,
            window_size=viz_kwargs.get("window_size", [3000, 3000]),
            scale=viz_kwargs.get("scale", 1),
            transparent_background=viz_kwargs.get("transparent_background", True),
            return_img=viz_kwargs.get("return_img", False),
        )

    return plotter


def create_camera_frustum_pyvista(
    pose: np.ndarray,
    scale=0.02,
    aspect_ratio=1.33,
    fov=60,
    show_axes=False,
    show_frustum=True,
):
    """
    Create a camera frustum using PyVista
    """
    # Convert FOV to radians
    fov_rad = np.radians(fov)

    # Calculate frustum dimensions
    near = scale * 0.1
    far = scale * 5.0

    # Near plane dimensions
    near_height = 2 * near * np.tan(fov_rad / 2)
    near_width = near_height * aspect_ratio

    # Far plane dimensions
    far_height = 2 * far * np.tan(fov_rad / 2)
    far_width = far_height * aspect_ratio

    # Define frustum vertices
    vertices = np.array(
        [
            # Camera center (apex)
            [0, 0, 0],
            # Near plane corners
            [-near_width / 2, -near_height / 2, -near],
            [near_width / 2, -near_height / 2, -near],
            [near_width / 2, near_height / 2, -near],
            [-near_width / 2, near_height / 2, -near],
            # Far plane corners
            [-far_width / 2, -far_height / 2, -far],
            [far_width / 2, -far_height / 2, -far],
            [far_width / 2, far_height / 2, -far],
            [-far_width / 2, far_height / 2, -far],
        ]
    )

    # Define lines connecting vertices to form frustum wireframe
    lines = []
    # Lines from camera center to near plane corners
    for i in range(1, 5):
        lines.extend([2, 0, i])

    # Lines from camera center to far plane corners
    for i in range(5, 9):
        lines.extend([2, 0, i])

    # Near plane rectangle
    near_rect = [4, 1, 2, 3, 4]
    lines.extend(near_rect)

    # Far plane rectangle
    far_rect = [4, 5, 6, 7, 8]
    lines.extend(far_rect)

    # Connect near to far plane corners
    for i in range(4):
        lines.extend([2, i + 1, i + 5])

    # Create PyVista polydata for the frustum
    frustum = pv.PolyData(vertices, lines=lines)

    points = frustum.points
    points_homo = np.column_stack([points, np.ones(points.shape[0])])
    transformed_points = (pose @ points_homo.T).T
    frustum.points = transformed_points[:, :3]

    if show_axes:
        # Origin in world space
        origin = pose[:3, 3]

        # Camera basis vectors (normalize to avoid scale issues)
        right = pose[:3, 0] / np.linalg.norm(pose[:3, 0])
        up = pose[:3, 1] / np.linalg.norm(pose[:3, 1])
        forward = -1 * (pose[:3, 2] / np.linalg.norm(pose[:3, 2]))

        scale = 0.25
        tip_length = scale / 2.5
        tip_radius = scale / 5
        shaft_radius = scale / 10

        axes = []
        for dir in [right, up, forward]:
            axes.append(
                pv.Arrow(
                    start=origin,
                    direction=dir,
                    tip_length=tip_length,
                    tip_radius=tip_radius,
                    shaft_radius=shaft_radius,
                    scale=scale,
                )
            )

        return frustum, axes
    else:
        return frustum, None


def format_pyvista_camera_params(camera_params: dict):
    """
    Format camera parameters for PyVista
    """
    # Get camera parameters
    c2w = camera_params["c2w"]
    c2w[:3, 1:3] *= -1  # Flip y axis (OpenGL to PyVista)

    K = camera_params["K"]
    _, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    h, w = camera_params["height"], camera_params["width"]

    # Convert principal point to window center (normalized coordinate system)
    wcx = -2 * (cx - float(w) / 2) / w
    wcy = 2 * (cy - float(h) / 2) / h

    # # Calculate field of view from intrinsics
    fov_y_rad = 2 * np.arctan((h - cy) / fy) if cy > h / 2 else 2 * np.arctan(cy / fy)
    fov_y_deg = np.degrees(fov_y_rad)

    # Extract camera position and orientation from c2w matrix
    camera_position = c2w[:3, 3]
    camera_forward = -c2w[:3, 2]  # -Z axis in camera coordinates

    # Extrinsics matrix
    w2c = np.linalg.inv(c2w)

    viz_kwargs = {
        "position": camera_position,
        "focal_point": camera_position + camera_forward,
        "view_angle": fov_y_deg,
        "window_center": (wcx, wcy),
        "model_transform": w2c,
        "window_size": (w, h),
        "lighting": VIZ_KWARGS.get("lighting", None),
    }

    return viz_kwargs


######################################################################
################## Open3D Visualization Functions ####################
######################################################################


def render_camera(mesh, width, height, intrinsic=None, extrinsic=None):
    # Create renderer
    renderer = o3d.visualization.rendering.OffscreenRenderer(width, height)
    renderer.scene.set_background([1, 1, 1, 1])  # white bg

    # Set material to use vertex colors
    material = o3d.visualization.rendering.MaterialRecord()
    material.shader = "defaultLit" if mesh.has_vertex_colors() else "defaultUnlit"

    # Add mesh with material
    if isinstance(mesh, o3d.geometry.TriangleMesh):
        renderer.scene.add_geometry("mesh", mesh, material)
    else:
        renderer.scene.add_geometry("pcd", mesh, material)

    # Set up camera
    renderer.setup_camera(intrinsic, extrinsic, width, height)

    # Render
    img = renderer.render_to_image()
    depth_image = renderer.render_to_depth_image(z_in_view_space=True)

    del renderer
    return np.asarray(img), np.asarray(depth_image)


##################################################################
############### Visualizing tracks on mesh #######################
##################################################################


def plot_tracks_on_image(
    df_tracks: pd.DataFrame,
    image: np.ndarray,
    colors: Optional[dict] = None,
    line_kwargs: dict = {"linestyle": "-", "linewidth": 4, "label": None, "alpha": 1.0},
):
    # Get image dimensions
    height, width = image.shape[:2]
    aspect_ratio = width / height

    # Create figure and axis with correct aspect ratio
    fig_size = 10
    fig = plt.figure(figsize=(fig_size * aspect_ratio, fig_size))
    ax = fig.add_subplot(111)
    ax.imshow(image)

    unique_track_ids = df_tracks["track_id"].unique()

    if colors is None:
        colors = {track_id: np.random.rand(3) for track_id in unique_track_ids}

    # Plot tracks for each unique track_id
    for track_id in unique_track_ids:
        track_data = df_tracks[df_tracks["track_id"] == track_id]
        ax.plot(track_data["u"], track_data["v"], color=colors[track_id], **line_kwargs)

    ax.axis("off")
    fig.tight_layout()
    plt.close(fig)

    return fig


def add_tracks_to_mesh(
    df_tracks: pd.DataFrame,
    plotter: pv.Plotter,
    colors: Optional[dict] = None,
    line_kwargs: dict = {},
    plot_points: bool = False,
):
    """
    Add tracks as connected line segments to a PyVista plotter.

    Args:
        df_tracks: DataFrame with columns ['track_id', 'x', 'y', 'z']
        plotter: PyVista plotter object
        colors: Optional dict mapping track_ids to colors. If None, colors are auto-generated.
        line_kwargs: Optional dict of kwargs passed to `plotter.add_mesh` for lines.
    """

    # If no colors provided, generate evenly spaced colors
    if colors is None:
        unique_track_ids = df_tracks.track_id.unique()
        num_tracks = len(unique_track_ids)
        golden_ratio = (1 + 5**0.5) / 2
        hues = [(i * golden_ratio) % 1 for i in range(num_tracks)]
        colors = {
            track_id: plt.cm.get_cmap("hsv")(hue)
            for track_id, hue in zip(unique_track_ids, hues)
        }

    for track_id, df_id in df_tracks.groupby("track_id"):
        # Ensure temporal order (important for smooth tracks)
        if "frame_id" in df_id.columns:
            df_id = df_id.sort_values("frame_id")
        elif "t" in df_id.columns:
            df_id = df_id.sort_values("t")

        # Get valid points
        valid_mask = ~df_id[["x", "y", "z"]].isna().any(axis=1)
        track_points = df_id.loc[valid_mask, ["x", "y", "z"]].values

        if len(track_points) < 2:
            continue

        if plot_points:
            # Plot points
            mesh = pv.PolyData(track_points)
            plotter.add_mesh(mesh, color=colors[track_id], **line_kwargs)
        else:
            # Plot as connected line
            n_pts = len(track_points)
            cells = np.concatenate(([n_pts], np.arange(n_pts)))
            line = pv.PolyData()
            line.points = track_points
            line.lines = cells
            plotter.add_mesh(line, color=colors[track_id], **line_kwargs)
    
    return plotter
