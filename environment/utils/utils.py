from pyntcloud import PyntCloud
import numpy as np
import open3d as o3d
import torch
from typing import Generator, List, Any
import gc
from huggingface_hub import hf_hub_download
import cv2
import random

########################################################
############ General Utility Functions #################
########################################################

def load_hf_weights(repo_id: str, filename: str):
    """
    Download a file from Hugging Face.
    """
    return hf_hub_download(repo_id=repo_id, filename=filename)

def pytorch_gc():
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    gc.collect()

def batch_iterator(batch_size: int, *args) -> Generator[List[Any], None, None]:
    """
    Batch iterator for MobileSAM --> helps with memory usage

    Inputs:
        - batch_size: int
        - *args: List[Any]

    Returns:
        - Generator[List[Any], None, None]

    Taken from feature-splatting
    """
    assert len(args) > 0 and all(
        len(a) == len(args[0]) for a in args
    ), "Batched iteration must have inputs of all the same size."
    n_batches = len(args[0]) // batch_size + int(len(args[0]) % batch_size != 0)
    for b in range(n_batches):
        yield [arg[b * batch_size : (b + 1) * batch_size] for arg in args]

########################################################
############ Video Utility Functions ###################
########################################################

def sample_random_frames(video_path, num_frames=10, seed=None):
    """
    Sample random frames from a video and return as numpy array.
    
    Args:
        video_path (str): Path to the video file
        num_frames (int): Number of random frames to sample
        seed (int, optional): Random seed for reproducibility
        
    Returns:
        numpy.ndarray: Array of shape (num_frames, height, width, channels)
        dict: Metadata containing frame indices and video info
    """
    if seed is not None:
        random.seed(seed)
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")
    
    # Get video properties
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"Video info: {total_frames} frames, {fps:.2f} FPS, {width}x{height}")
    
    if num_frames > total_frames:
        print(f"Warning: Requested {num_frames} frames but video only has {total_frames}")
        num_frames = total_frames
    
    # Generate random frame indices
    frame_indices = sorted(random.sample(range(total_frames), num_frames))
    
    # Sample frames
    frames = []
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        
        if ret:
            # Convert BGR to RGB (OpenCV uses BGR by default)
            frames.append(frame)
        else:
            print(f"Warning: Could not read frame at index {idx}")
    
    cap.release()
    
    # Convert to numpy array
    frames_array = np.array(frames)
    
    # Metadata
    metadata = {
        'frame_indices': frame_indices,
        'total_frames': total_frames,
        'fps': fps,
        'resolution': (width, height),
        'sampled_frames': len(frames)
    }
    
    return frames_array, metadata

def cut_video(video_path, start_frame, num_frames, output_filename=None):
    """
    Extracts a specified number of frames starting from a given frame number and saves as video.
    If output_file is provided, saves as a video file.
    Otherwise returns frames as a numpy array.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Initialize video writer if output file specified
    if output_filename:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_filename, fourcc, fps, (width, height))
    else:
        frames = []

    frame_count = 0
    extracted_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count >= start_frame and extracted_count < num_frames:
            if output_filename:
                out.write(frame)
            else:
                frames.append(frame)
            extracted_count += 1
        
        if extracted_count >= num_frames:
            break

        frame_count += 1

    cap.release()
    if output_filename:
        out.release()
        print(f"Extracted {extracted_count} frames to video: {output_filename}")
    else:
        return np.stack(frames)

def read_ply(filename, view_direction=[0.0, 0.0, 1.0]):
    """
    Use PyntCloud to read a PLY file and return a DataFrame with the point data.
    """

    ply_info = PyntCloud.from_file(filename.as_posix())

    n_points = ply_info.points.shape[0]
    n_fields = ply_info.points.shape[-1]

    print (f"Loaded {n_points} points")
    print (f"Found {n_fields} fields")

    # Extract spherical harmonics coefficients
    # splatfacto typically stores SH coefficients as f_dc_* and f_rest_*
    sh_coeffs = []
    
    # DC component (f_dc_0, f_dc_1, f_dc_2 for RGB)
    if all(f'f_dc_{i}' in ply_info.points.columns for i in range(3)):
        dc_coeffs = np.vstack([ply_info.points[f'f_dc_{i}'] for i in range(3)])
        sh_coeffs.append(dc_coeffs)
    
    # Higher order SH coefficients (f_rest_*)
    rest_coeffs = []
    i = 0
    while f'f_rest_{i}' in ply_info.points.columns:
        rest_coeffs.append(ply_info.points.loc[:, f'f_rest_{i}'])
        i += 1
    
    if rest_coeffs:
        rest_coeffs = np.vstack(rest_coeffs)
        sh_coeffs.append(rest_coeffs)
    
    # return sh_coeffs
    if sh_coeffs:
        sh_coeffs = np.concatenate(sh_coeffs, axis=0).T
    else:
        sh_coeffs = np.zeros((len(points), 3))  # Fallback to DC only
    # # Try to extract colors - first check for spherical harmonics
    #     print("Found spherical harmonic DC coefficients")
    #     colors = np.column_stack([ply_info.points[f'f_dc_{i}'] for i in range(3)])
    #     colors = np.clip(colors + 0.5, 0, 1)  # Convert from SH to RGB space

    # ply_info.points.loc[:, ['red', 'green', 'blue']] = colors

    # Set default viewing direction if not provided
    # if view_direction is None:
    # view_direction = np.array([0.0, 0.0, 1.0])

    view_direction = np.array(view_direction)
    
    # Evaluate spherical harmonics to get colors
    colors = eval_spherical_harmonics(sh_coeffs, view_direction)
    ply_info.points.loc[:, ['red', 'green', 'blue']] = colors

    # Convert opacity logits to opacity
    opacities = ply_info.points.opacity.to_numpy()
    opacities = torch.sigmoid(torch.tensor(opacities))
    ply_info.points.loc[:, 'opacity'] = opacities.numpy()

    return ply_info


def eval_spherical_harmonics(sh_coeffs, directions):
    """
    Evaluate spherical harmonics for given directions.
    
    Args:
        sh_coeffs: (N, sh_dim) SH coefficients 
        directions: (N, 3) or (3,) viewing directions (normalized)
        
    Returns:
        colors: (N, 3) RGB colors
    """
    if directions.ndim == 1:
        directions = directions.reshape(1, -1)
    if directions.shape[0] == 1 and sh_coeffs.shape[0] > 1:
        directions = np.tile(directions, (sh_coeffs.shape[0], 1))
    
    # Normalize directions
    directions = directions / (np.linalg.norm(directions, axis=1, keepdims=True) + 1e-8)
    
    x, y, z = directions[:, 0], directions[:, 1], directions[:, 2]
    
    # Number of SH coefficients determines the degree
    sh_dim = sh_coeffs.shape[1] // 3  # 3 color channels
    
    if sh_dim == 1:  # Only DC component
        colors = sh_coeffs + 0.5  # DC offset for splatfacto
    elif sh_dim == 4:  # Up to degree 1
        # SH basis functions for degree 0 and 1
        sh_basis = np.stack([
            np.ones_like(x),  # Y_0^0
            y,                # Y_1^{-1}
            z,                # Y_1^0
            x,                # Y_1^1
        ], axis=1)
        
        # Reshape coefficients and evaluate
        sh_coeffs_reshaped = sh_coeffs.reshape(-1, 3, sh_dim)
        colors = np.sum(sh_coeffs_reshaped * sh_basis[:, None, :], axis=2)
        colors = colors + 0.5  # Add DC offset
        
    elif sh_dim == 9:  # Up to degree 2
        # SH basis functions for degree 0, 1, and 2
        sh_basis = np.stack([
            np.ones_like(x),           # Y_0^0
            y,                         # Y_1^{-1}
            z,                         # Y_1^0
            x,                         # Y_1^1
            x * y,                     # Y_2^{-2}
            y * z,                     # Y_2^{-1}
            3 * z * z - 1,            # Y_2^0
            x * z,                     # Y_2^1
            x * x - y * y,            # Y_2^2
        ], axis=1)
        
        # Apply SH coefficients
        sh_coeffs_reshaped = sh_coeffs.reshape(-1, 3, sh_dim)
        colors = np.sum(sh_coeffs_reshaped * sh_basis[:, None, :], axis=2)
        colors = colors + 0.5  # Add DC offset
        
    else:
        # Fallback to DC component only
        colors = sh_coeffs[:, :3] + 0.5
    
    # Clamp colors to valid range
    colors = np.clip(colors, 0, 1)
    return colors


def create_pcd(ply_info):
    # Create and populate Open3D point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(ply_info.points[['x', 'y', 'z']].to_numpy())

    if all(col in ply_info.points.columns for col in ['red', 'green', 'blue']):
        pcd.colors = o3d.utility.Vector3dVector(ply_info.points[['red', 'green', 'blue']].to_numpy())
        print("Added colors to point cloud")

    return pcd