import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.optimize import fsolve
import matplotlib.patches as patches

class SphericalProjection:
    def __init__(self, sphere_center, sphere_radius, camera_pos, camera_target, camera_up):
        self.sphere_center = np.array(sphere_center)
        self.sphere_radius = sphere_radius
        self.camera_pos = np.array(camera_pos)
        self.camera_target = np.array(camera_target)
        self.camera_up = np.array(camera_up)
        
        # Compute camera coordinate system (OpenGL/OpenCV convention)
        # Forward points from camera towards target
        self.forward = self._normalize(self.camera_target - self.camera_pos)
        self.right = self._normalize(np.cross(self.forward, self.camera_up))
        self.up = np.cross(self.right, self.forward)
        
        # Create view matrix
        self.view_matrix = self._create_view_matrix()
    
    def _normalize(self, v):
        return v / np.linalg.norm(v)
    
    def _create_view_matrix(self):
        """Create view matrix to transform world coords to camera coords"""
        # In camera coordinates: +X = right, +Y = up, +Z = towards camera (negative forward)
        R = np.array([
            self.right,
            self.up,
            -self.forward  # Negative because +Z points towards camera
        ])
        t = -R @ self.camera_pos
        view = np.eye(4)
        view[:3, :3] = R
        view[:3, 3] = t
        return view
    
    def world_to_camera(self, points):
        """Transform world coordinates to camera coordinates"""
        if points.ndim == 1:
            points = points.reshape(1, -1)
        
        # Add homogeneous coordinate
        points_h = np.column_stack([points, np.ones(points.shape[0])])
        
        # Transform
        camera_coords = (self.view_matrix @ points_h.T).T
        return camera_coords[:, :3]
    
    def compute_sphere_silhouette_exact(self):
        """
        Compute the exact silhouette of the sphere as seen from the camera.
        This is the intersection of the sphere with the cone of rays from camera
        that are tangent to the sphere.
        """
        # Transform sphere center to camera coordinates
        sphere_center_cam = self.world_to_camera(self.sphere_center.reshape(1, -1))[0]
        
        # Distance from camera to sphere center
        d = np.linalg.norm(sphere_center_cam)
        
        # Check if sphere is in front of camera
        if sphere_center_cam[2] >= -self.sphere_radius:
            print(f"Warning: Sphere is behind or intersecting camera plane! Z = {sphere_center_cam[2]:.3f}")
            return None, "Sphere behind camera"
        
        if d <= self.sphere_radius:
            print("Warning: Camera is inside the sphere!")
            return None, "Camera inside sphere"
        
        # Angle of the tangent cone (half-angle)
        sin_alpha = self.sphere_radius / d
        cos_alpha = np.sqrt(1 - sin_alpha**2)
        
        # The silhouette is a circle on the sphere, perpendicular to the line
        # from camera to sphere center
        
        # Center of silhouette circle (on the line from camera to sphere center)
        silhouette_center_cam = sphere_center_cam * cos_alpha**2
        
        # Radius of silhouette circle
        silhouette_radius = self.sphere_radius * sin_alpha * cos_alpha
        
        # Create the silhouette circle
        # The circle lies in a plane perpendicular to the camera-sphere direction
        sphere_direction = self._normalize(sphere_center_cam)
        
        # Create two orthogonal vectors in the silhouette plane
        if abs(sphere_direction[0]) < 0.9:
            v1 = np.array([1, 0, 0])
        else:
            v1 = np.array([0, 1, 0])
        
        v1 = v1 - np.dot(v1, sphere_direction) * sphere_direction
        v1 = self._normalize(v1)
        v2 = np.cross(sphere_direction, v1)
        
        # Generate points on the silhouette circle
        theta = np.linspace(0, 2*np.pi, 100)
        silhouette_points_cam = (
            silhouette_center_cam.reshape(1, -1) +
            silhouette_radius * np.outer(np.cos(theta), v1) +
            silhouette_radius * np.outer(np.sin(theta), v2)
        )
        
        return silhouette_points_cam, silhouette_center_cam, silhouette_radius
    
    def project_to_image_plane(self, points_3d, focal_length=1.0):
        """
        Project 3D points in camera coordinates to 2D image plane.
        Using perspective projection: x' = f * x/(-z), y' = f * y/(-z)
        Note: In camera coords, objects in front have negative Z values
        """
        if points_3d.ndim == 1:
            points_3d = points_3d.reshape(1, -1)
        
        # In camera coordinates, points in front of camera have negative Z
        z = points_3d[:, 2]
        
        # Filter out points behind camera (positive Z) or too close to camera
        valid_mask = (z < -1e-6)  # Must be in front of camera with negative Z
        
        if not np.any(valid_mask):
            return np.array([]), valid_mask
        
        valid_points = points_3d[valid_mask]
        # Perspective projection with negative Z (points in front have negative Z)
        x_proj = focal_length * valid_points[:, 0] / (-valid_points[:, 2])
        y_proj = focal_length * valid_points[:, 1] / (-valid_points[:, 2])
        
        return np.column_stack([x_proj, y_proj]), valid_mask
    
    def analyze_projected_conic(self, focal_length=1.0):
        """
        Analyze the projected silhouette to determine if it's a circle or ellipse.
        Returns conic parameters and classification.
        """
        # Get silhouette points
        silhouette_result = self.compute_sphere_silhouette_exact()
        
        if silhouette_result is None or len(silhouette_result) != 3:
            return None, "Failed to compute silhouette"
            
        silhouette_3d, center_3d, radius_3d = silhouette_result
        
        # Project to image plane
        projected_points, valid_mask = self.project_to_image_plane(silhouette_3d, focal_length)
        
        if len(projected_points) == 0:
            return None, "No valid projection"
        
        # Fit conic section to projected points
        conic_params = self._fit_conic_section(projected_points)
        
        # Classify the conic
        conic_type, ellipse_params = self._classify_conic(conic_params)
        
        return {
            'silhouette_3d': silhouette_3d,
            'projected_points': projected_points,
            'conic_params': conic_params,
            'conic_type': conic_type,
            'ellipse_params': ellipse_params,
            'center_3d': center_3d,
            'radius_3d': radius_3d
        }, conic_type
    
    def _fit_conic_section(self, points):
        """
        Fit a conic section Ax^2 + Bxy + Cy^2 + Dx + Ey + F = 0 to points.
        Using least squares method.
        """
        x, y = points[:, 0], points[:, 1]
        
        # Design matrix for conic: [x^2, xy, y^2, x, y, 1]
        A_matrix = np.column_stack([x**2, x*y, y**2, x, y, np.ones(len(x))])
        
        # Solve using SVD (homogeneous least squares)
        U, s, Vt = np.linalg.svd(A_matrix)
        conic_coeffs = Vt[-1, :]  # Last row of V^T corresponds to smallest singular value
        
        return conic_coeffs
    
    def _classify_conic(self, coeffs):
        """
        Classify conic section and extract ellipse parameters if applicable.
        Conic: Ax^2 + Bxy + Cy^2 + Dx + Ey + F = 0
        """
        A, B, C, D, E, F = coeffs
        
        # Discriminant
        discriminant = B**2 - 4*A*C
        
        if abs(discriminant) < 1e-10:
            conic_type = "parabola"
        elif discriminant < 0:
            conic_type = "ellipse"
        else:
            conic_type = "hyperbola"
        
        ellipse_params = None
        
        if conic_type == "ellipse":
            # Extract ellipse parameters
            ellipse_params = self._extract_ellipse_parameters(coeffs)
        
        return conic_type, ellipse_params
    
    def _extract_ellipse_parameters(self, coeffs):
        """Extract center, semi-axes, and rotation angle from ellipse equation."""
        A, B, C, D, E, F = coeffs
        
        # Center coordinates
        det = B**2 - 4*A*C
        if abs(det) < 1e-10:
            return None
            
        cx = (2*C*D - B*E) / det
        cy = (2*A*E - B*D) / det
        
        # Rotation angle
        if abs(B) < 1e-10:
            theta = 0 if A < C else np.pi/2
        else:
            theta = 0.5 * np.arctan(B / (A - C))
        
        # Semi-axes lengths
        # Transform to canonical form
        cos_t, sin_t = np.cos(theta), np.sin(theta)
        
        A_rot = A * cos_t**2 + B * cos_t * sin_t + C * sin_t**2
        C_rot = A * sin_t**2 - B * cos_t * sin_t + C * cos_t**2
        
        # Constant term after translation
        F_trans = A*cx**2 + B*cx*cy + C*cy**2 + D*cx + E*cy + F
        
        if F_trans >= 0:
            return None  # Not a real ellipse
        
        a = np.sqrt(-F_trans / A_rot)
        b = np.sqrt(-F_trans / C_rot)
        
        # Ensure a >= b (semi-major >= semi-minor)
        if a < b:
            a, b = b, a
            theta += np.pi/2
        
        return {
            'center': (cx, cy),
            'semi_major': a,
            'semi_minor': b,
            'rotation_angle': theta,
            'eccentricity': np.sqrt(1 - (b/a)**2) if a > 0 else 0
        }
    
    def compute_theoretical_analysis(self):
        """
        Theoretical analysis of when projection is circle vs ellipse.
        """
        # Transform sphere center to camera coordinates
        sphere_center_cam = self.world_to_camera(self.sphere_center.reshape(1, -1))[0]
        
        # In camera coordinates, forward direction is -Z (towards negative Z)
        # Check if sphere is in front of camera
        if sphere_center_cam[2] >= 0:
            print(f"Warning: Sphere may be behind camera! Z = {sphere_center_cam[2]:.3f}")
        
        # Camera looking direction in camera coordinates is (0, 0, -1)
        camera_forward = np.array([0, 0, -1])
        sphere_direction = self._normalize(sphere_center_cam)
        
        # Angle between camera forward direction and camera-to-sphere direction
        alignment_angle = np.arccos(np.clip(np.dot(camera_forward, sphere_direction), -1, 1))
        alignment_degrees = np.degrees(alignment_angle)
        
        # Distance analysis
        distance_to_sphere = np.linalg.norm(sphere_center_cam)
        
        if distance_to_sphere <= self.sphere_radius:
            viewing_angle = 90.0  # Camera inside sphere
        else:
            viewing_angle = np.degrees(np.arcsin(self.sphere_radius / distance_to_sphere))
        
        return {
            'sphere_center_camera': sphere_center_cam,
            'distance_to_sphere': distance_to_sphere,
            'viewing_angle_degrees': viewing_angle,
            'alignment_angle_degrees': alignment_degrees,
            'is_orthogonal_view': abs(alignment_degrees) < 5.0,  # Nearly orthogonal
            'theoretical_shape': 'circle' if abs(alignment_degrees) < 5.0 else 'ellipse'
        }

def debug_projection_geometry():
    """Debug function to understand the projection geometry better"""
    
    print("\n" + "="*60)
    print("DEBUGGING: Why are all projections circles?")
    print("="*60)
    
    sphere_center = [0, 0, 0]
    sphere_radius = 2.0
    
    # Test with very extreme camera position
    camera_pos = [20, 0, 0.5]  # Very close to sphere plane, far to the side
    
    proj = SphericalProjection(
        sphere_center, sphere_radius,
        camera_pos, [0, 0, 0], [0, 1, 0]
    )
    
    # Get the 3D silhouette circle
    silhouette_result = proj.compute_sphere_silhouette_exact()
    if silhouette_result is None:
        print("Failed to compute silhouette")
        return
        
    silhouette_3d, center_3d, radius_3d = silhouette_result
    
    print(f"Camera position: {camera_pos}")
    print(f"Sphere center in camera coords: {center_3d}")
    print(f"Silhouette center in camera coords: {center_3d}")
    print(f"Silhouette radius in 3D: {radius_3d:.3f}")
    
    # Check the 3D silhouette circle orientation
    camera_to_sphere = proj._normalize(center_3d)
    print(f"Direction from camera to sphere center: {camera_to_sphere}")
    
    # Project a few points and see their Z values
    sample_points = silhouette_3d[::10]  # Every 10th point
    projected_points, valid_mask = proj.project_to_image_plane(sample_points, focal_length=5.0)
    
    if len(projected_points) > 0:
        z_values = sample_points[valid_mask][:, 2]
        print(f"Z values of projected points: min={z_values.min():.3f}, max={z_values.max():.3f}")
        print(f"Z range: {z_values.max() - z_values.min():.3f}")
        
        # Check if the Z variation is causing the elliptical effect
        z_relative_variation = (z_values.max() - z_values.min()) / abs(z_values.mean())
        print(f"Relative Z variation: {z_relative_variation:.3f}")
        
        if z_relative_variation < 0.01:
            print("→ Very small Z variation - projection should be nearly circular")
        else:
            print("→ Significant Z variation - projection should be elliptical")
    
    # Let's manually create a tilted circle to force an ellipse
    print("\n" + "-"*40)
    print("MANUAL TEST: Creating a tilted circle")
    print("-"*40)
    
    # Create a circle tilted 45 degrees in camera space
    t = np.linspace(0, 2*np.pi, 100)
    circle_radius = 1.0
    
    # Circle in a tilted plane: mix XY and XZ components
    tilt_angle = np.pi/3  # 60 degrees
    tilted_circle = np.column_stack([
        circle_radius * np.cos(t),
        circle_radius * np.sin(t) * np.cos(tilt_angle),
        circle_radius * np.sin(t) * np.sin(tilt_angle) - 5  # Move in front of camera
    ])
    
    projected_tilted, valid = proj.project_to_image_plane(tilted_circle, focal_length=5.0)
    
    if len(projected_tilted) > 0:
        # Fit ellipse to this known tilted circle
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        
        ax.scatter(projected_tilted[:, 0], projected_tilted[:, 1], alpha=0.6, s=20)
        
        # Quick ellipse fitting
        conic_params = proj._fit_conic_section(projected_tilted)
        conic_type, ellipse_params = proj._classify_conic(conic_params)
        
        print(f"Tilted circle projects as: {conic_type}")
        if ellipse_params:
            print(f"Eccentricity: {ellipse_params['eccentricity']:.3f}")
            print(f"Axis ratio: {ellipse_params['semi_minor']/ellipse_params['semi_major']:.3f}")
        
        ax.set_aspect('equal')
        ax.grid(True)
        ax.set_title("Projection of manually tilted circle")
        plt.show()

def run_analysis_examples():
    """Run several examples to demonstrate circle vs ellipse projections."""
    
    sphere_center = [0, 0, 0]
    sphere_radius = 2.0
    
    examples = [
        {
            'name': 'Orthogonal View (looking at center)',
            'camera_pos': [0, 0, 8],
            'camera_target': [0, 0, 0],  # Looking at sphere center
            'camera_up': [0, 1, 0]
        },
        {
            'name': 'Oblique View (looking past sphere)',
            'camera_pos': [6, 4, 6],
            'camera_target': [2, 1, -2],  # Looking past sphere center
            'camera_up': [0, 1, 0]
        },
        {
            'name': 'Side View (tangential)',
            'camera_pos': [8, 0, 2],
            'camera_target': [0, 0, -3],  # Looking parallel to sphere
            'camera_up': [0, 1, 0]
        },
        {
            'name': 'Extreme Grazing (almost tangent)',
            'camera_pos': [12, 0, 1],
            'camera_target': [-5, 0, -2],  # Looking way off to the side
            'camera_up': [0, 1, 0]
        }
    ]
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.flatten()
    
    for i, example in enumerate(examples):
        print(f"\n{'='*50}")
        print(f"Example {i+1}: {example['name']}")
        print('='*50)
        
        proj = SphericalProjection(
            sphere_center, sphere_radius,
            example['camera_pos'], example['camera_target'], example['camera_up']
        )
        
        # Theoretical analysis
        theory = proj.compute_theoretical_analysis()
        print(f"Distance to sphere: {theory['distance_to_sphere']:.2f}")
        print(f"Viewing angle: {theory['viewing_angle_degrees']:.1f}°")
        print(f"Alignment angle: {theory['alignment_angle_degrees']:.1f}°")
        print(f"Theoretical shape: {theory['theoretical_shape']}")
        
        # Compute projection
        result, conic_type = proj.analyze_projected_conic(focal_length=5.0)
        
        if result is None:
            print("No valid projection found")
            continue
            
        print(f"Actual projected shape: {conic_type}")
        
        # Plot results
        ax = axes[i]
        
        # Plot projected points
        points = result['projected_points']
        ax.scatter(points[:, 0], points[:, 1], alpha=0.6, s=2, label='Silhouette points')
        
        # Plot fitted conic
        if result['ellipse_params'] is not None:
            params = result['ellipse_params']
            ellipse = patches.Ellipse(
                params['center'], 
                2*params['semi_major'], 
                2*params['semi_minor'],
                angle=np.degrees(params['rotation_angle']),
                fill=False, color='red', linewidth=2,
                label=f'Fitted ellipse (e={params["eccentricity"]:.3f})'
            )
            ax.add_patch(ellipse)
            
            print(f"Semi-major axis: {params['semi_major']:.3f}")
            print(f"Semi-minor axis: {params['semi_minor']:.3f}")
            print(f"Eccentricity: {params['eccentricity']:.3f}")
            print(f"Rotation angle: {np.degrees(params['rotation_angle']):.1f}°")
            
            # Check if it's approximately a circle
            axis_ratio = params['semi_minor'] / params['semi_major']
            print(f"Axis ratio (minor/major): {axis_ratio:.3f}")
            if axis_ratio > 0.98:
                print("→ This is approximately a CIRCLE")
            elif axis_ratio > 0.85:
                print("→ This is a slightly flattened ELLIPSE")
            elif axis_ratio > 0.5:
                print("→ This is a moderately elongated ELLIPSE") 
            else:
                print("→ This is a very elongated ELLIPSE")
        else:
            # Try to plot the convex hull or best-fit circle
            if len(points) > 3:
                # Fit a circle for comparison
                center_x, center_y = np.mean(points, axis=0)
                distances = np.sqrt((points[:, 0] - center_x)**2 + (points[:, 1] - center_y)**2)
                radius = np.mean(distances)
                circle = patches.Circle((center_x, center_y), radius, 
                                      fill=False, color='blue', linewidth=2,
                                      label=f'Best-fit circle (r={radius:.3f})')
                ax.add_patch(circle)
                print(f"Fitted as circle with radius: {radius:.3f}")
                print(f"Standard deviation of radii: {np.std(distances):.3f}")
                
                # Check if points really form a circle
                relative_std = np.std(distances) / radius
                print(f"Relative std of radii: {relative_std:.3f}")
                if relative_std < 0.02:
                    print("→ This is indeed a very good CIRCLE")
                else:
                    print("→ This deviates from a perfect circle")
        
        ax.set_aspect('equal', adjustable='box')
        ax.grid(True, alpha=0.3)
        ax.set_title(f"{example['name']}\n{conic_type.capitalize()}")
        ax.legend()
        
        # Ensure equal axis limits for proper circle/ellipse visualization
        x_range = points[:, 0].max() - points[:, 0].min()
        y_range = points[:, 1].max() - points[:, 1].min()
        max_range = max(x_range, y_range)
        center_x, center_y = np.mean(points, axis=0)
        margin = max_range * 0.1
        ax.set_xlim(center_x - max_range/2 - margin, center_x + max_range/2 + margin)
        ax.set_ylim(center_y - max_range/2 - margin, center_y + max_range/2 + margin)
    
    plt.tight_layout()
    plt.savefig('sphere_projections.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    return examples

def run_extreme_ellipse_test():
    """Create cases that will definitely produce ellipses"""
    
    print("\n" + "="*60)
    print("EXTREME ELLIPSE TEST: Force elliptical projections")
    print("="*60)
    
    sphere_center = [0, 0, 0]
    sphere_radius = 2.0
    
    # Test cases where sphere is off-center in camera view
    extreme_cases = [
        {
            'name': 'Sphere at edge of view (should be strong ellipse)',
            'camera_pos': [0, 0, 10],
            'camera_target': [5, 0, 0],  # Looking 5 units to the right of sphere
            'camera_up': [0, 1, 0]
        },
        {
            'name': 'Very oblique peripheral view',
            'camera_pos': [8, 8, 8],
            'camera_target': [10, 10, 0],  # Looking way off to the side
            'camera_up': [0, 0, 1]  # Different up vector too
        },
        {
            'name': 'Near-tangent view',
            'camera_pos': [2.5, 0, 0],  # Very close to sphere surface
            'camera_target': [0, 5, 0],  # Looking perpendicular
            'camera_up': [0, 0, 1]
        }
    ]
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    for i, case in enumerate(extreme_cases):
        print(f"\n{'-'*40}")
        print(f"Extreme Case {i+1}: {case['name']}")
        print('-'*40)
        
        try:
            proj = SphericalProjection(
                sphere_center, sphere_radius,
                case['camera_pos'], case['camera_target'], case['camera_up']
            )
            
            # Theoretical analysis
            theory = proj.compute_theoretical_analysis()
            print(f"Distance to sphere: {theory['distance_to_sphere']:.2f}")
            print(f"Viewing angle: {theory['viewing_angle_degrees']:.1f}°")
            print(f"Alignment angle: {theory['alignment_angle_degrees']:.1f}°")
            
            # Compute projection
            result, conic_type = proj.analyze_projected_conic(focal_length=3.0)
            
            if result is None:
                print("No valid projection found")
                continue
                
            print(f"Projected shape: {conic_type}")
            
            # Plot results
            ax = axes[i]
            points = result['projected_points']
            ax.scatter(points[:, 0], points[:, 1], alpha=0.7, s=15, label='Silhouette')
            
            if result['ellipse_params'] is not None:
                params = result['ellipse_params']
                ellipse = patches.Ellipse(
                    params['center'], 
                    2*params['semi_major'], 
                    2*params['semi_minor'],
                    angle=np.degrees(params['rotation_angle']),
                    fill=False, color='red', linewidth=3,
                    label=f'e={params["eccentricity"]:.2f}'
                )
                ax.add_patch(ellipse)
                
                axis_ratio = params['semi_minor'] / params['semi_major']
                print(f"Eccentricity: {params['eccentricity']:.3f}")
                print(f"Axis ratio: {axis_ratio:.3f}")
                
                if axis_ratio < 0.7:
                    print("→ SUCCESS: Clear ELLIPSE!")
                elif axis_ratio < 0.9:
                    print("→ Moderate ellipse")
                else:
                    print("→ Nearly circular")
            
            ax.set_aspect('equal', adjustable='box')
            ax.grid(True, alpha=0.3)
            ax.set_title(f"{case['name']}\nAlign: {theory['alignment_angle_degrees']:.1f}°")
            ax.legend()
            
            # Force equal axis scaling
            if len(points) > 0:
                x_range = points[:, 0].max() - points[:, 0].min()
                y_range = points[:, 1].max() - points[:, 1].min()
                max_range = max(x_range, y_range)
                center_x, center_y = np.mean(points, axis=0)
                margin = max_range * 0.15
                ax.set_xlim(center_x - max_range/2 - margin, center_x + max_range/2 + margin)
                ax.set_ylim(center_y - max_range/2 - margin, center_y + max_range/2 + margin)
            
        except Exception as e:
            print(f"Error: {e}")
            continue
    
    plt.tight_layout()
    plt.show()

# Run the analysis
if __name__ == "__main__":
    examples = run_analysis_examples()
    
    # Run extreme test cases
    run_extreme_ellipse_test()
    
    # Run debugging to understand why we get circles
    debug_projection_geometry()
    
    print(f"\n{'='*60}")
    print("SUMMARY: When is sphere projection a circle vs ellipse?")
    print('='*60)
    print("• CIRCLE: When camera looks directly at sphere center")
    print("• ELLIPSE: When sphere appears off-center in camera view")
    print("• KEY INSIGHT: Camera target direction matters more than position!")
    print("• The alignment angle between camera direction and")
    print("  camera-to-sphere direction determines the shape")
    print("• Eccentricity = 0 for perfect circle, approaches 1 for very flat ellipse")