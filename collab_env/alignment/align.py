import json
import numpy as np
from tqdm import tqdm
import shutil
from pathlib import Path
import pyvista as pv
import matplotlib.pyplot as plt

import pycolmap
from hloc import extract_features, match_features, pairs_from_exhaustive
from hloc.localize_sfm import QueryLocalizer, pose_from_cluster

#########################################################
######## Registration of new camera to COLMAP ###########
#########################################################

def align_to_colmap(
    preproc_dir: Path,
    query_dir: Path,
    output_dir: Path,
    localizer_conf: dict = {
        'estimation': {'ransac': {'max_error': 50}},
        'refinement': {'refine_focal_length': True, 'refine_extra_params': True},
    },
):
    """
    Align a new image to a fit COLMAP model.

    Args:
        preproc_dir: Path to the source directory containing the SfM model.
        query_dir: Path to the query directory containing the new images.
        output_dir: Path to the output directory.
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
    model = pycolmap.Reconstruction(model_path)
    references_registered = [model.images[i].name for i in model.reg_image_ids()]

    ##########################################
    ############ Query setup #################
    ##########################################

    print (f"Creating output directory...")
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

    print (f"Extracting features from queries and matching...")

    # Feature and matcher config
    feature_conf = extract_features.confs['superpoint_aachen']
    matcher_conf = match_features.confs['superglue']

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
        query_pairs,
        image_list=query_images_list,
        ref_list=references_registered
    )

    # Adds query information into the matches file
    match_features.main(
        matcher_conf,
        query_pairs,
        features=query_features,
        matches=query_matches,
        overwrite=True
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
        ref_ids = [model.find_image_with_name(n).image_id for n in references_registered]

        ret, loc = pose_from_cluster(localizer, image, camera, ref_ids, query_features, query_matches)
        poses.append(ret)
        logs.append(loc)

    return poses, logs

def align_to_splat(preproc_dir, pose):

    # Load COLMAP cameras and align them to nerfstudio format
    camera_params, transforms = colmap2nerfstudio(preproc_dir)
    camera_params = extract_camera_params(pose)

    # Apply nerfstudio transforms to our new camera pose
    camera_params['pose'] = transforms['transform_matrix'] @ camera_params['pose'] # Align to splat
    camera_params['pose'][:3, 3] *= transforms['scale'] # Scale to splat

    return camera_params
    

def align_to_mesh(preproc_dir, mesh_dir, pose):

    # Align to splat first --> returns camera params aligned to the splat
    camera_params = align_to_splat(preproc_dir, pose)

    # Load the mesh transform and compose into matrix
    transform = np.load(mesh_dir / 'alignment.npz')
    transform = np.concatenate([
        transform['R'], 
        (transform['translation'][..., np.newaxis])
    ], axis=1)

    # Turn to homogeneous coordinates
    transform = np.concatenate([
        transform, 
        np.array([[0, 0, 0, 1]])
    ], axis=0)

    # Apply the mesh transform to the camera params
    camera_params['pose'] = transform @ camera_params['pose']
    
    return camera_params

def extract_camera_params(pose):
    """
    Transform a localized pose in COLMAP to match the nerfstudio coordinate system.
    """

    # Convert COLMAP pose to c2w
    c2w = pose['cam_from_world'].inverse().matrix()
    c2w = np.concatenate([
        c2w, 
        np.array([0, 0, 0, 1])[np.newaxis, :]
    ], axis=0)

    # Convert from COLMAP's camera coordinate system (OpenCV) to ours (OpenGL)
    c2w[0:3, 1:3] *= -1
    c2w = c2w[np.array([0, 2, 1, 3]), :]
    c2w[2, :] *= -1

    # Get intrinsics
    f, cx, cy, _ = pose['camera'].params
    fx = fy = f
    height, width = pose['camera'].height, pose['camera'].width

    K = np.array([
        [fx, 0, cx],
        [0, fy, cy],
        [0,  0,  1],
    ])

    return {
        'fx': fx,
        'fy': fy,
        'cx': cx,
        'cy': cy,
        'K': K,
        'height': height,
        'width': width,
        'pose': c2w,  # 4x4 matrix
    }

#########################################################
######## Mapping of COLMAP to Nerfstudio camera #########
#########################################################

def colmap2nerfstudio(preproc_dir, downscale_factor=2.0):
    """
    Maps the COLMAP camera parameters to the Nerfstudio camera parameters.
    Annoying but necessary as splats are created after processing the cameras
    within nerfstudio.

    This extrapolates those parameters to the colmap cameras and aligns them
    to the splat.
    """

    with open(preproc_dir / "transforms.json", "r", encoding="utf-8") as f:
        meta = json.load(f)

    # Applies to all cameras
    items = ['fl_x', 'fl_y', 'cx', 'cy', 'h', 'w']
    downscale_factor = 1.0 / downscale_factor
    fx, fy, cx, cy, height, width = [meta[item] * downscale_factor for item in items]

    K = np.array([
        [fx, 0, cx],
        [0, fy, cy],
        [0,  0,  1],
    ])

    poses = np.stack([cam['transform_matrix'] for cam in meta['frames']])
    origins = poses[..., :3, 3]

    mean_origin = np.mean(origins, axis=0)
    translation_diff = origins - mean_origin
    translation = mean_origin

    # Orient upwards
    up = np.mean(poses[:, :3, 1], axis=0)
    up = up / np.linalg.norm(up)

    rotation = rotation_matrix_between(up, np.array([0, 0, 1]))
    transform = np.concatenate([rotation, rotation @ -translation[..., None]], axis=-1)
    poses = transform @ poses

    scale_factor = 1.0
    scale_factor /= float(np.max(np.abs(poses[:, :3, 3])))
    poses[:, :3, 3] *= scale_factor
    
    # Orient upwards
    poses[:, :3, 1:3] *= -1
    
    # Add homogeneous coordinates to make them 4x4 matrices
    bottom_row = np.tile([0, 0, 0, 1], (poses.shape[0], 1, 1))
    poses = np.concatenate([poses, bottom_row], axis=1)

    camera_params = {
        'fx': fx,
        'fy': fy,
        'cx': cx,
        'cy': cy,
        'K': K,
        'height': int(height),
        'width': int(width),
        'poses': poses,
    }

    transform = np.concatenate([
        transform,
        np.array([0, 0, 0, 1])[np.newaxis, :],
    ], axis=0)

    transforms = {
        'transform_matrix': transform,
        'scale': scale_factor,
    }

    return camera_params, transforms


def project_image(K, c2w, points, colors):
    """
    Project 3D points to 2D image coordinates.

    Args:
        K: Camera intrinsics matrix (3x3)
        c2w: Camera-to-world transformation matrix (4x4)
        points: (N, 3) array of 3D points
        colors: (N, 3) array of colors

    Returns:
        points_proj: (N, 2) array of 2D points
        colors: (N, 3) array of colors
    """

    # Convert from camera-to-world to world-to-camera
    w2c = np.linalg.inv(c2w)

    # Get rotation and translation
    R = w2c[:3, :3]
    t = w2c[:3, 3][:, np.newaxis]

    # Apply mapping transforming points in world to camera space
    points_cam = (R @ points.T + t).T  # (N, 3)
    points_cam = np.asarray(points_cam)

    # Filter out points behind the camera
    in_front = points_cam[:, 2] > 0
    points_cam = points_cam[in_front]
    colors = colors[in_front]

    # Project to 2D
    points_proj = (K @ points_cam.T).T  # (N, 3)
    points_proj = points_proj[:, :2] / points_proj[:, 2:3]  # normalize

    return points_proj, colors

def plot_projection(H, W, points_proj, colors):
    image = np.ones((H, W, 3), dtype=np.float32)  # white background
    # Draw points
    for pt, color in zip(points_proj, colors):
        x, y = int(pt[0]), int(pt[1])
        if 0 <= x < W and 0 <= y < H:
            image[y, x] = color  # (y, x) because row-major

    return image

def nerfstudio_project_image(camera, points, colors):

    H, W = camera.height, camera.width
    K = camera.get_intrinsics_matrices().clone().cpu().numpy()
    c2w = camera.camera_to_worlds.clone().cpu().numpy()
    c2w = c2w.squeeze(0)
    c2w = np.asarray(c2w)
    c2w[:3, 1:3] *= -1
    c2w = np.concatenate([c2w, np.asarray([0, 0, 0, 1])[np.newaxis, :]], axis=0)
    w2c = np.linalg.inv(c2w)

    points_proj, colors = project_image(K, w2c, points, colors)
    proj_image = plot_projection(H, W, points_proj, colors)

    camera_params = {
        'K': K,
        'w2c': w2c,
        'H': H,
        'W': W,
    }

    return proj_image, camera_params

#########################################################
################# Helper Functions ######################
#########################################################

def rotation_matrix_between(a, b):
    """Compute the rotation matrix that rotates vector a to vector b.

    Args:
        a: The vector to rotate.
        b: The vector to rotate to.
    Returns:
        The rotation matrix.

    Taken from nerfstudio
    """
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    v = np.cross(a, b)  # Axis of rotation.

    # Handle cases where `a` and `b` are parallel.
    eps = 1e-6
    if np.sum(np.abs(v)) < eps:
        x = np.array([1.0, 0, 0]) if abs(a[0]) < eps else np.array([0, 1.0, 0])
        v = np.cross(a, x)

    v = v / np.linalg.norm(v)
    skew_sym_mat = np.array(
        [
            [0, -v[2], v[1]],
            [v[2], 0, -v[0]],
            [-v[1], v[0], 0],
        ]
    )
    theta = np.arccos(np.clip(np.dot(a, b), -1, 1))

    # Rodrigues rotation formula. https://en.wikipedia.org/wiki/Rodrigues%27_rotation_formula
    return np.eye(3) + np.sin(theta) * skew_sym_mat + (1 - np.cos(theta)) * (skew_sym_mat @ skew_sym_mat)

def qvec2rotmat(qvec):
    """
    Convert a quaternion to a rotation matrix.

    Args:
        qvec: (4,) array of quaternion
    """
    return np.array(
        [
            [
                1 - 2 * qvec[2] ** 2 - 2 * qvec[3] ** 2,
                2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
                2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2],
            ],
            [
                2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
                1 - 2 * qvec[1] ** 2 - 2 * qvec[3] ** 2,
                2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1],
            ],
            [
                2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],
                2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
                1 - 2 * qvec[1] ** 2 - 2 * qvec[2] ** 2,
            ],
        ]
    )

# def transform_localized(ret, transforms):
#     """
#     Transform a localized pose in COLMAP to match the nerfstudio coordinate system.
#     """
#     # Convert COLMAP (qvec + tvec) to c2w
#     rotation = qvec2rotmat(ret['qvec'])
#     translation = ret['tvec'].reshape(3, 1)
    
#     w2c = np.concatenate([rotation, translation], axis=1)  # 3x4
#     w2c = np.vstack([w2c, np.array([0, 0, 0, 1])])         # 4x4
#     c2w = np.linalg.inv(w2c)                               # world-to-camera → camera-to-world

#     # Convert from COLMAP's camera coordinate system (OpenCV) to ours (OpenGL)
#     c2w[0:3, 1:3] *= -1
#     c2w = c2w[np.array([0, 2, 1, 3]), :]
#     c2w[2, :] *= -1

#     # Apply transform matrix (global alignment from colmap2nerfstudio)
#     c2w = transforms['transform_matrix'] @ c2w

#     # Apply scene scaling
#     c2w[:3, 3] *= transforms['scale']

#     # Flip y and z axes (OpenCV → OpenGL)
#     c2w[:3, 1:3] *= -1
#     c2w = np.concatenate([c2w, np.asarray([0, 0, 0, 1])[np.newaxis, :]], axis=0)

#     # Get intrinsics
#     f, cx, cy, _ = ret['camera']['params']
#     fx = fy = f
#     height, width = ret['camera']['height'], ret['camera']['width']

#     K = np.array([
#         [fx, 0, cx],
#         [0, fy, cy],
#         [0,  0,  1],
#     ])

#     return {
#         'fx': fx,
#         'fy': fy,
#         'cx': cx,
#         'cy': cy,
#         'K': K,
#         'height': height,
#         'width': width,
#         'pose': c2w,  # 4x4 matrix
#     }

#########################################################
######### Visualization of alignment ###################
#########################################################

def pyvista_camera_view_with_intrinsics(mesh_fn, camera_params, 
                                      save_path=None, show_info=True):
    """
    PyVista visualization that properly uses camera intrinsics to set up the view.
    
    Args:
        mesh_fn: Path to mesh PLY file
        mesh_alignment_fn: Path to alignment NPZ file
        localized_params: Dict with 'pose', 'K', 'height', 'width'
        save_path: Optional path to save the image
        show_info: Whether to print camera info
    """

    # Load mesh with PyVista
    mesh_pv = pv.read(str(mesh_fn))

    # Get camera parameters
    c2w = camera_params['pose']
    
    K = camera_params['K']
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    h, w = camera_params['height'], camera_params['width']
    
    # Calculate field of view from intrinsics
    fov_y_rad = 2 * np.arctan(h / (2 * fy))
    fov_x_rad = 2 * np.arctan(w / (2 * fx))
    fov_y_deg = np.degrees(fov_y_rad)
    fov_x_deg = np.degrees(fov_x_rad)
    aspect_ratio = w/h
    
    if show_info:
        print(f"Camera Intrinsics:")
        print(f"  Resolution: {w} x {h}")
        print(f"  Focal lengths: fx={fx:.1f}, fy={fy:.1f}")
        print(f"  Principal point: cx={cx:.1f}, cy={cy:.1f}")
        print(f"  Field of view: {fov_x_deg:.1f}° (horizontal), {fov_y_deg:.1f}° (vertical)")
        print(f"  Aspect ratio: {aspect_ratio:.3f}")
        
        # Check if principal point is centered
        if abs(cx - w/2) > 5 or abs(cy - h/2) > 5:
            print(f"  WARNING: Principal point is off-center by ({cx-w/2:.1f}, {cy-h/2:.1f}) pixels")
        
        # Check if fx and fy are similar (square pixels)
        if abs(fx - fy) > 1:
            print(f"  WARNING: Non-square pixels detected (fx/fy = {fx/fy:.3f})")
    
    # Create plotter with exact resolution
    plotter = pv.Plotter(
        off_screen=True, 
        window_size=(w, h)
    )
    
    # Add mesh
    if 'RGB' in mesh_pv.point_data:
        plotter.add_mesh(mesh_pv, scalars='RGB', rgb=True)
    else:
        plotter.add_mesh(mesh_pv, color='lightblue')
    
    # Extract camera position and orientation from c2w matrix
    camera_position = c2w[:3, 3]
    camera_forward = -c2w[:3, 2]  # -Z axis in camera coordinates
    camera_up = c2w[:3, 1]        # Y axis in camera coordinates
    
    # Set camera position and orientation
    plotter.camera.position = camera_position
    plotter.camera.focal_point = camera_position + camera_forward
    plotter.camera.up = camera_up
    
    # Set field of view - PyVista uses vertical FOV
    plotter.camera.view_angle = fov_x_deg #fov_y_deg * aspect_ratio
    
    # Handle non-square pixels by adjusting the viewport
    if abs(fx - fy) > 1:
        # For non-square pixels, we need to adjust the aspect ratio
        print(f"Adjusting for non-square pixels (fx={fx:.1f}, fy={fy:.1f})")
        # This might require additional viewport manipulation
    
    # Handle off-center principal point
    if abs(cx - w/2) > 1 or abs(cy - h/2) > 1:
        # For off-center principal point, we might need to adjust the camera
        print(f"Note: Principal point offset may not be perfectly handled in PyVista")
    
    try:
        # Render the image
        image = plotter.screenshot(return_img=True)
        plotter.close()
        
        # Display the result
        plt.figure(figsize=(7, 5))
        plt.imshow(image)
        plt.axis('off')
        plt.title(f'PyVista Camera View ({w}x{h}, FOV: {fov_y_deg:.1f}°)')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=150)
            print(f"Saved PyVista render to: {save_path}")
        
        plt.show()
        
        return image
        
    except Exception as e:
        print(f"PyVista rendering failed: {e}")
        print("Falling back to projection method...")
        return None