import os
import json
import numpy as np
import torch
import shutil
import pickle
import cv2
from typing import Optional

from pathlib import Path
from tqdm import tqdm

import pycolmap
from hloc import extract_features, match_features, pairs_from_exhaustive
from hloc.localize_sfm import QueryLocalizer, pose_from_cluster

import lpips
from skimage.metrics import structural_similarity as ssim

#########################################################
########### Extract frames for alignment ################
#########################################################


def extract_video_frames(video_path, output_dir, n_frames=1, method="random"):
    """
    Extract frames from a video. Realistically, you only need one frame but
    build a verbose function for now.
    """

    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    os.makedirs(output_dir, exist_ok=True)

    if method == "random":
        frame_indices = np.random.randint(0, frame_count, n_frames)
    elif method == "evenly_spaced":
        frame_indices = np.linspace(0, frame_count - 1, n_frames).astype(int)
    else:
        raise ValueError(f"Invalid method: {method}")

    # Extract frames
    frame_fns = []

    for i in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()

        # Make sure we read the frame
        if not ret:
            break

        # Save frame
        frame_fn = output_dir / f"{i:06d}.png"
        cv2.imwrite(str(frame_fn), frame)
        frame_fns.append(frame_fn)

    cap.release()
    return frame_fns


#########################################################
######## Registration of new camera to COLMAP ###########
#########################################################


def align_to_colmap(
    preproc_dir: Path,
    query_dir: Path,
    output_dir: Path,
    localizer_conf: dict = {
        "estimation": {"ransac": {"max_error": 50}},
        "refinement": {"refine_focal_length": True, "refine_extra_params": True},
    },
    return_logs: bool = False,
):
    """
    Align a new image to a fit COLMAP model.

    Args:
        preproc_dir: Path to the source directory containing the SfM model.
        query_dir: Path to the query directory containing the new images.
        output_dir: Path to the output directory.
        return_logs: Whether to return the logs from the localization + model for visualization.
    """

    #########################################################
    ############ Grab original model information ############
    #########################################################

    # Contains the points3D.bin and images.bin
    colmap_path = preproc_dir / "colmap/"

    # Path to the original training information
    model_path = colmap_path / "sparse/0/"
    model_features = colmap_path / "features.h5"
    model_matches = colmap_path / "matches.h5"

    # Load the COLMAP model and find the images used for its training
    model = pycolmap.Reconstruction(str(model_path))
    references_registered = [model.images[i].name for i in model.reg_image_ids()]

    ##########################################
    ############ Query setup #################
    ##########################################

    print("Creating output directory...")
    output_dir.mkdir(exist_ok=True, parents=True)

    # Define paths for queries
    query_features = output_dir / "features.h5"
    query_matches = output_dir / "matches.h5"
    query_pairs = output_dir / "pairs-loc.txt"
    query_images_list = [image.name for image in query_dir.glob("*")]

    # Copy over original files as to not overwrite / alter
    shutil.copy(src=model_features, dst=query_features)
    shutil.copy(src=model_matches, dst=query_matches)

    ######################################################
    ###### Feature extraction and keypoint matching ######
    ######################################################

    print("Extracting features from queries and matching...")

    # Feature and matcher config
    feature_conf = extract_features.confs["superpoint_aachen"]
    matcher_conf = match_features.confs["superglue"]

    # Extracts features and writes the to the query_features file
    extract_features.main(
        conf=feature_conf,
        image_dir=query_dir,
        feature_path=query_features,
        image_list=query_images_list,
        overwrite=True,
    )

    # Find every set of pairs of reference images and query images (writes to file)
    pairs_from_exhaustive.main(
        query_pairs, image_list=query_images_list, ref_list=references_registered
    )

    # Adds query information into the matches file
    match_features.main(
        matcher_conf,
        query_pairs,
        features=query_features,
        matches=query_matches,
        overwrite=True,
    )

    ######################################################
    ################## Localization #####################
    ######################################################

    localizer = QueryLocalizer(model, localizer_conf)

    poses = []
    logs = []

    # Go through each query image and localize it
    for image in tqdm(query_images_list, desc="Localizing query images..."):
        # Create a camera from the image (i.e., FOV, principal point, etc.)
        camera = pycolmap.infer_camera_from_image(query_dir / image)
        ref_ids = [
            model.find_image_with_name(n).image_id for n in references_registered
        ]

        ret, loc = pose_from_cluster(
            localizer, image, camera, ref_ids, query_features, query_matches
        )

        poses.append(ret)

        logs.append(
            {
                "query": query_dir / image,
                "loc": loc,
            }
        )

    viz_info = {
        "reconstruction": model,
        "db_image_dir": preproc_dir / "images",
    }

    if return_logs:
        return poses, logs, viz_info
    else:
        return poses


def extract_camera_params(pose, alpha=0.5):
    """
    Extract camera parameters from a COLMAP pose (OpenCV convention) and
    convert to OpenGL convention.

    OpenCV convention:
    - x right
    - y down
    - z forward

    OpenGL convention:
    - x right
    - y up
    - z backward
    """

    # Convert COLMAP pose to c2w
    c2w = pose["cam_from_world"].inverse().matrix()
    c2w = np.concatenate([c2w, np.array([0, 0, 0, 1])[np.newaxis, :]], axis=0)

    # Convert from COLMAP's camera coordinate system (OpenCV) to ours (OpenGL)
    c2w = c2w[np.array([0, 2, 1, 3]), :]
    c2w[2, :] *= -1

    # Get intrinsics
    camera_model = pose["camera"].model.name
    params = pose["camera"].params

    if camera_model in ["SIMPLE_PINHOLE"]:
        fx = fy = params[0]
        cx, cy = params[1], params[2]
        dist_coeffs = np.zeros(5)

    elif camera_model in ["SIMPLE_RADIAL"]:
        fx = fy = params[0]
        cx, cy = params[1], params[2]
        dist_coeffs = np.zeros(5)
        dist_coeffs[0] = params[3]  # k1

    elif camera_model in ["PINHOLE"]:
        fx, fy, cx, cy = params[:4]
        dist_coeffs = np.zeros(5)

    elif camera_model in ["RADIAL"]:
        fx = fy = params[0]
        cx, cy = params[1], params[2]
        dist_coeffs = np.zeros(5)
        dist_coeffs[:3] = params[3:6]  # k1, k2, k3

    elif camera_model in ["OPENCV", "FULL_OPENCV"]:
        fx, fy, cx, cy = params[:4]
        raw_dist = np.array(params[4:], dtype=np.float32)
        dist_coeffs = np.zeros(8, dtype=np.float32)
        dist_coeffs[: len(raw_dist)] = raw_dist
        # No reordering needed; COLMAP already uses OpenCV order: [k1, k2, p1, p2, k3, k4, k5, k6]

    else:
        raise ValueError(f"Camera model {camera_model} not supported.")

    height, width = pose["camera"].height, pose["camera"].width

    K = np.array(
        [
            [fx, 0, cx],
            [0, fy, cy],
            [0, 0, 1],
        ]
    )

    K, _ = cv2.getOptimalNewCameraMatrix(K, dist_coeffs, (width, height), alpha)

    return {
        "fx": fx,
        "fy": fy,
        "cx": cx,
        "cy": cy,
        "K": K,
        "height": height,
        "width": width,
        "c2w": c2w,  # 4x4 matrix
    }


#########################################################
######## COLMAP / Nerfstudio Alignment Functions ########
#########################################################


def load_colmap_cameras(preproc_dir, downscale_factor=2.0):
    """
    Load fit COLMAP cameras from nerfstudio. This contains the
    camera parameters and poses.

    Args:
        preproc_dir (Path): Path to the nerfstudio preprocessed directory.
        downscale_factor (float): Factor to downscale the cameras by.

    Returns:
    """
    with open(preproc_dir / "transforms.json", "r") as f:
        cameras = json.load(f)

    # Applies to all cameras
    items = ["fl_x", "fl_y", "cx", "cy", "h", "w"]
    downscale_factor = 1.0 / downscale_factor
    fx, fy, cx, cy, height, width = [cameras[item] * downscale_factor for item in items]

    # Extract distortion parameters
    param_keys = ["k1", "k2", "p1", "p2", "k3", "k4", "k5", "k6"]
    params = [cameras[key] if key in cameras else 0 for key in param_keys]
    camera_model = cameras["camera_model"]

    K = np.array(
        [
            [fx, 0, cx],
            [0, fy, cy][0, 0, 1],
        ]
    )

    poses = np.stack([camera["transform_matrix"] for camera in cameras["frames"]])

    camera_params = {
        "fx": fx,
        "fy": fy,
        "cx": cx,
        "cy": cy,
        "camera_model": camera_model,
        "params": params,
        "height": int(height),
        "width": int(width),
        "K": K,
        "poses": poses,
    }

    return camera_params


def load_nerfstudio_transforms(mesh_dir):
    """
    Transforms applied by nerfstudio dataparser to COLMAP coordinate system.

    Provides a transform that 1) centers the cameras and 2) points them in the Z-up direction.

    """
    nerfstudio_transforms_fn = list(mesh_dir.parent.rglob("dataparser_transforms*"))[0]

    with open(nerfstudio_transforms_fn, "r") as f:
        nerfstudio_transforms = json.load(f)

    # Apply weird nerfstudio transform conventions
    transform = np.stack(nerfstudio_transforms["transform"])
    transform = np.concatenate([transform, np.array([[0, 0, 0, 1]])], axis=0)

    # Go backwards from nerfstudio's convention to COLMAP
    transform = transform[:, [0, 2, 1, 3]]
    transform[:, 2] *= -1

    nerfstudio_transforms["transform"] = transform

    return nerfstudio_transforms


def align_to_splat(camera_params, mesh_dir, out_fn: Optional[str] = None):
    """
    Align camera poses to the Nerfstudio coordinate system and apply scaling.

    Args:
        poses (np.ndarray): Input poses, either shape (3,4) or (N,3,4).
        mesh_dir (str): Directory containing Nerfstudio transforms file.

    Returns:
        np.ndarray: Transformed poses with homogeneous coordinates, shape
                    (4,4) or (N,4,4).
    """

    # Grab the nerfstudio transforms file
    nerfstudio_transforms = load_nerfstudio_transforms(mesh_dir)

    # Grab the nerfstudio transforms
    nerfstudio_transform = np.stack(nerfstudio_transforms["transform"])
    nerfstudio_scale = nerfstudio_transforms["scale"]

    # Apply nerfstudio transforms to our new camera pose
    c2w = nerfstudio_transform @ camera_params["c2w"]
    c2w[..., :3, 3] *= nerfstudio_scale
    camera_params["c2w"] = c2w

    if out_fn is not None:
        with open(out_fn, "wb") as f:
            pickle.dump(camera_params, f)

    return camera_params


def align_to_mesh(camera_params, mesh_dir, out_fn: Optional[str] = None):
    """
    Align camera poses to the mesh coordinate system.

    Args:
        poses (np.ndarray): Input poses, either shape (3,4) or (N,3,4). (c2w)
        mesh_dir (str): Directory containing mesh.

    Returns:
    """

    # Align to splat first --> returns camera params aligned to the splat
    camera_params = align_to_splat(camera_params, mesh_dir)  # Wait to flip z

    with open(mesh_dir / "transforms.pkl", "rb") as f:
        transform = pickle.load(f)

    # Apply the mesh transform to the camera params
    camera_params["c2w"] = transform["mesh_transform"] @ camera_params["c2w"]

    if out_fn is not None:
        with open(out_fn, "wb") as f:
            pickle.dump(camera_params, f)

    return camera_params


#########################################################
################ Image Quality Metrics ##################
#########################################################


def compute_image_similarity(
    view_image: np.ndarray,
    frame: np.ndarray,
    mask_method="brightness",
    background_threshold=240,
    edge_threshold=50,
    morphology_kernel_size=3,
    lpips_net="alex",
):
    """
    Compare two RGB images with different backgrounds: view_image (PyVista render) and frame (camera).
    Uses the PyVista render to identify mesh regions (non-white pixels) for focused comparison.

    Args:
        view_image (np.ndarray): HxWx3 RGB PyVista render (mesh=dark, background=white)
        frame (np.ndarray): HxWx3 RGB camera frame
        mask_method (str): 'brightness_based', 'edge_based', or 'combined'
        background_threshold (int): pixels darker than this are considered mesh (default=240)
        edge_threshold (int): threshold for edge detection
        morphology_kernel_size (int): kernel size for morphological operations
        lpips_net (str): LPIPS network type

    Returns:
        dict: Contains similarity scores and additional info
    """

    view_image = view_image.copy()
    frame = frame.copy()

    # Convert to uint8 if needed
    if view_image.dtype != np.uint8:
        view_image = np.clip(view_image * 255, 0, 255).astype(np.uint8)
    if frame.dtype != np.uint8:
        frame = np.clip(frame * 255, 0, 255).astype(np.uint8)

    # Create mesh mask from PyVista render (dark pixels = mesh, white pixels = background)
    mask = create_foreground_mask(
        view_image,
        method=mask_method,
        background_threshold=background_threshold,
        edge_threshold=edge_threshold,
        kernel_size=morphology_kernel_size,
    )

    # Apply mask to both images for fair comparison
    masked_view, masked_frame = apply_mask_for_comparison(view_image, frame, mask)

    # Calculate SSIM on masked grayscale images
    gray_view = cv2.cvtColor(masked_view, cv2.COLOR_RGB2GRAY)
    gray_frame = cv2.cvtColor(masked_frame, cv2.COLOR_RGB2GRAY)

    # Only compute SSIM on valid (non-masked) pixels
    valid_pixels = mask > 0
    if np.sum(valid_pixels) > 100:  # Ensure enough pixels for meaningful comparison
        ssim_score = ssim(
            gray_view,
            gray_frame,
            data_range=255,
            gaussian_weights=True,
            sigma=1.5,
            use_sample_covariance=False,
        )
    else:
        ssim_score = 0.0

    # Calculate LPIPS on masked images
    lpips_score = compute_masked_lpips(masked_view, masked_frame, mask, lpips_net)

    # Additional metrics
    mesh_coverage = np.sum(mask > 0) / mask.size  # How much of the image contains mesh

    return {
        "ssim": ssim_score,
        "lpips": lpips_score,
        "mesh_coverage": mesh_coverage,
        "mesh_mask": mask,
        "masked_view": masked_view,
        "masked_frame": masked_frame,
    }


def create_foreground_mask(
    view_image,
    method="brightness",
    background_threshold=240,
    edge_threshold=50,
    kernel_size=3,
):
    """
    Create a foreground mask to identify mesh regions (non-white pixels) in the PyVista render.
    In PyVista renders, mesh = dark pixels, background = white pixels.
    """
    gray = cv2.cvtColor(view_image, cv2.COLOR_RGB2GRAY)

    if method == "brightness":
        # Mesh pixels are darker than the white background
        # Use < threshold to identify non-white (mesh) pixels
        mask = gray < background_threshold

    elif method == "edge":
        # First get brightness-based mask as base
        brightness_mask = gray < background_threshold

        # Use edge detection to refine boundaries
        edges = cv2.Canny(gray, edge_threshold, edge_threshold * 2)

        # Dilate edges slightly to capture boundary regions
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        dilated_edges = cv2.dilate(edges, kernel, iterations=1)

        # Combine brightness mask with edge information
        mask = np.logical_or(brightness_mask, dilated_edges).astype(np.uint8)

        # Clean up with morphological operations
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)

    elif method == "combined":
        # Start with brightness thresholding (main approach for PyVista)
        brightness_mask = gray < background_threshold

        # Add edge detection for refinement
        edges = cv2.Canny(gray, edge_threshold, edge_threshold * 2)
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        edge_mask = cv2.dilate(edges, kernel, iterations=1)

        # Combine: brightness mask OR edge mask
        mask = np.logical_or(brightness_mask, edge_mask).astype(np.uint8)

        # Clean up the combined mask
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

    else:
        raise ValueError(f"Unknown mask method: {method}")

    return mask.astype(np.uint8)


def apply_mask_for_comparison(view_image, frame, mask):
    """
    Apply mask to both images for fair comparison.
    Set background pixels to a neutral value.
    """
    masked_view = view_image.copy()
    masked_frame = frame.copy()

    # Set masked regions to neutral gray (128) for both images
    neutral_color = 128
    masked_view[mask == 0] = neutral_color
    masked_frame[mask == 0] = neutral_color

    return masked_view, masked_frame


def compute_masked_lpips(view_image, frame, mask, lpips_net="alex"):
    """
    Compute LPIPS on masked regions with proper normalization.
    """

    def to_tensor(img):
        img = img.astype(np.float32) / 255.0
        # LPIPS expects values in [-1, 1]
        img = img * 2.0 - 1.0
        return torch.tensor(img).permute(2, 0, 1).unsqueeze(0).float()

    tensor_view = to_tensor(view_image)
    tensor_frame = to_tensor(frame)

    # Calculate perceptual similarity
    lpips_fn = lpips.LPIPS(net=lpips_net)
    with torch.no_grad():
        lpips_score = lpips_fn(tensor_view, tensor_frame).item()

    return lpips_score


def visualize_comparison(view_image, frame, result):
    """
    Helper function to visualize the comparison results.
    """
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Original images
    axes[0, 0].imshow(view_image)
    axes[0, 0].set_title("View Image (Render)")
    axes[0, 0].axis("off")

    axes[0, 1].imshow(frame)
    axes[0, 1].set_title("Frame (Original)")
    axes[0, 1].axis("off")

    axes[0, 2].imshow(result["mesh_mask"], cmap="gray")
    axes[0, 2].set_title("Mesh Mask (White=Mesh)")
    axes[0, 2].axis("off")

    # Masked images
    axes[1, 0].imshow(result["masked_view"])
    axes[1, 0].set_title("Masked View")
    axes[1, 0].axis("off")

    axes[1, 1].imshow(result["masked_frame"])
    axes[1, 1].set_title("Masked Frame")
    axes[1, 1].axis("off")

    # Metrics
    axes[1, 2].text(
        0.1,
        0.8,
        f"SSIM: {result['ssim']:.4f}",
        transform=axes[1, 2].transAxes,
        fontsize=12,
    )
    axes[1, 2].text(
        0.1,
        0.6,
        f"LPIPS: {result['lpips']:.4f}",
        transform=axes[1, 2].transAxes,
        fontsize=12,
    )
    axes[1, 2].text(
        0.1,
        0.4,
        f"Mesh Coverage: {result['mesh_coverage']:.2%}",
        transform=axes[1, 2].transAxes,
        fontsize=12,
    )
    axes[1, 2].set_title("Metrics")
    axes[1, 2].axis("off")

    plt.tight_layout()
    plt.show()
