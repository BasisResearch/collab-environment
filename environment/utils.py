from pyntcloud import PyntCloud
import numpy as np
import open3d as o3d

def reconstruct_features(model, features):
    """
    Recover full dimensional features from distilled features
    """

    features = features.permute(1, 0).unsqueeze(0).unsqueeze(-1)  # [1, 13, N, 1]

    # Pass through 1d conv to upsample 
    hidden_features = model.feature_mlp.hidden_conv(features)

    # Pass through branch in relation to the current feature
    recovered_features = {}
    for branch_name, branch_layer in model.feature_mlp.feature_branch_dict.items():
        recovered_features[branch_name] = branch_layer(hidden_features)

    return recovered_features


def read_ply(filename):
    """
    Use PyntCloud to read a PLY file and return a DataFrame with the point data.
    """

    ply_info = PyntCloud.from_file(filename.as_posix())

    n_points = ply_info.points.shape[0]
    n_fields = ply_info.points.shape[-1]

    print (f"Loaded {n_points} points")
    print (f"Found {n_fields} fields")


    # Try to extract colors - first check for spherical harmonics
    if all(f'f_dc_{i}' in ply_info.points.columns for i in range(3)):
        print("Found spherical harmonic DC coefficients")
        colors = np.column_stack([ply_info.points[f'f_dc_{i}'] for i in range(3)])
        colors = np.clip(colors + 0.5, 0, 1)  # Convert from SH to RGB space

    ply_info.points.loc[:, ['red', 'green', 'blue']] = colors

    return ply_info

def create_pcd(ply_info):
    # Create and populate Open3D point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(ply_info.points[['x', 'y', 'z']].to_numpy())

    if all(col in ply_info.points.columns for col in ['red', 'green', 'blue']):
        pcd.colors = o3d.utility.Vector3dVector(ply_info.points[['red', 'green', 'blue']].to_numpy())
        print("Added colors to point cloud")

    return pcd