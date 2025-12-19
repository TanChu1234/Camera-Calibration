import cv2
import numpy as np
import yaml
from scipy.optimize import least_squares

class CameraAlignmentCalibrator:
    """
    Aligns multiple cameras to a common reference frame using shared reference points.
    """
    
    def __init__(self, mapper, reference_camera):
        """
        Args:
            mapper: MultiCameraMapper instance with all cameras added
            reference_camera: str "cam3"
        """
        self.mapper = mapper
        self.reference_camera = reference_camera
        self.transformations = {}  # camera_id -> (tx, ty, theta)
        
    def calibrate_with_reference_points(self, reference_pairs):
        """
        Calculate transformations to align all cameras using reference point pairs.
        
        Args:
            reference_pairs: List of dictionaries, each containing:
                {
                    'cam1': (u1, v1),  # pixel in camera 1
                    'cam2': (u2, v2),  # pixel in camera 2
                    'cam3': (u3, v3),  # pixel in camera 3 (optional)
                    ...
                }
        """
        if not reference_pairs:
            raise ValueError("Need at least 1 reference point pair")
        
        camera_ids = list(reference_pairs[0].keys())
        # self.reference_camera = camera_ids[0]
        self.transformations[self.reference_camera] = (0, 0, 0)
        
        print(f"\n{'='*60}")
        print(f"CALIBRATION: Using '{self.reference_camera}' as reference camera")
        print(f"{'='*60}\n")
        
        # Get reference points in reference camera's world frame
        ref_world_points = []
        for pair in reference_pairs:
            cam_u, cam_v = pair[self.reference_camera]
            calib = self.mapper.cameras[self.reference_camera]
            wx, wy, wz = calib.pixel_to_world(cam_u, cam_v, z_world=0)
            ref_world_points.append([wx, wy])
        ref_world_points = np.array(ref_world_points)
        
        # For each other camera, find transformation
        for cam_id in camera_ids:
            if cam_id == self.reference_camera:
                continue
            
            print(f"\nCalibrating '{cam_id}' relative to '{self.reference_camera}':")
            print("-" * 50)
            
            cam_world_points = []
            for pair in reference_pairs:
                if cam_id not in pair:
                    continue
                cam_u, cam_v = pair[cam_id]
                calib = self.mapper.cameras[cam_id]
                wx, wy, wz = calib.pixel_to_world(cam_u, cam_v, z_world=0)
                cam_world_points.append([wx, wy])
            cam_world_points = np.array(cam_world_points)
            
            if len(cam_world_points) < 2:
                print(f"  ⚠ Warning: Only {len(cam_world_points)} point(s). Need ≥2 for rotation.")
            
            tx, ty, theta = self.rigid_transform_2d(cam_world_points, ref_world_points)
            self.transformations[cam_id] = (tx, ty, theta)
            
            print(f"  ✓ Transformation found:")
            print(f"    Translation: ({tx:.2f}, {ty:.2f}) mm")
            print(f"    Rotation:    {np.degrees(theta):.2f}°")
            
            error = self._compute_alignment_error(cam_world_points, ref_world_points, tx, ty, theta)
            print(f"    RMS Error:   {error:.2f} mm")
    
    def _solve_transformation(self, source_points, target_points):
        """Solve for [tx, ty, theta] that transforms source to target."""
        def residuals(params):
            tx, ty, theta = params
            R = np.array([
                [np.cos(theta), -np.sin(theta)],
                [np.sin(theta),  np.cos(theta)]
            ])
            transformed = (R @ source_points.T).T + np.array([tx, ty])
            return (transformed - target_points).flatten()
        
        x0 = [0, 0, 0]
        result = least_squares(residuals, x0, method='lm')
        tx, ty, theta = result.x
        return tx, ty, theta

    def rigid_transform_2d(self, source_points, target_points):
        """
        Compute 2D rigid transform (tx, ty, theta) that maps
        source_points -> target_points

        source_points: (N, 2) ndarray
        target_points: (N, 2) ndarray
        """

        # Centroids
        centroid_src = source_points.mean(axis=0)
        centroid_tgt = target_points.mean(axis=0)

        # Center the points
        src_centered = source_points - centroid_src
        tgt_centered = target_points - centroid_tgt

        # Cross-covariance matrix
        H = src_centered.T @ tgt_centered

        # SVD
        U, _, Vt = np.linalg.svd(H)

        # Rotation
        R = Vt.T @ U.T

        # Ensure a proper rotation (no reflection)
        if np.linalg.det(R) < 0:
            Vt[1, :] *= -1
            R = Vt.T @ U.T

        # Translation
        t = centroid_tgt - R @ centroid_src

        # Rotation angle (radians)
        theta = np.arctan2(R[1, 0], R[0, 0])

        return t[0], t[1], theta
    
    def _compute_alignment_error(self, source_points, target_points, tx, ty, theta):
        """Compute RMS alignment error after transformation."""
        R = np.array([
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta),  np.cos(theta)]
        ])
        transformed = (R @ source_points.T).T + np.array([tx, ty])
        errors = np.linalg.norm(transformed - target_points, axis=1)
        return np.sqrt(np.mean(errors**2))
    
    def transform_to_reference(self, camera_id, world_x, world_y):
        """
        Transform a point from camera's world frame to reference frame.
        """
        if camera_id not in self.transformations:
            raise ValueError(f"Camera '{camera_id}' not calibrated")
        
        if camera_id == self.reference_camera:
            return world_x, world_y
        
        tx, ty, theta = self.transformations[camera_id]
        R = np.array([
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta),  np.cos(theta)]
        ])
        point = np.array([world_x, world_y])
        transformed = R @ point + np.array([tx, ty])
        return transformed[0], transformed[1]
    
    def project_aligned_point_to_map(self, camera_id, cam_u, cam_v, z_world=0, 
                                     color=(0, 255, 0), label=None):
        """
        Project a camera pixel to the map using the aligned coordinate system.
        """
        calib = self.mapper.cameras[camera_id]
        wx, wy, wz = calib.pixel_to_world(cam_u, cam_v, z_world=z_world)
        ref_x, ref_y = self.transform_to_reference(camera_id, wx, wy)
        map_u, map_v = self.mapper.world_to_map(self.reference_camera, ref_x, ref_y)
        
        if label:
            print(f"\n{label}:")
        else:
            print(f"\n{camera_id} → Aligned:")
        print(f"  Camera pixel:      ({cam_u}, {cam_v})")
        print(f"  Camera world:      ({wx:.2f}, {wy:.2f}) mm")
        print(f"  Reference world:   ({ref_x:.2f}, {ref_y:.2f}) mm")
        print(f"  Map pixel:         ({map_u}, {map_v})")
        
        self.mapper.draw_point((map_u, map_v), color=color, radius=5)
        if label:
            self.mapper.draw_text(label, (map_u + 10, map_v), color=color)
        
        return map_u, map_v, ref_x, ref_y
    
    def save_aligned_extrinsics(self, output_dir="output/aligned_extrinsics"):
        """
        Save aligned extrinsic parameters for each camera.
        
        This creates new extrinsic files where all cameras share the same world origin
        (the reference camera's origin).
        
        Args:
            output_dir: Directory to save aligned extrinsic files
        """
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"\n{'='*60}")
        print("SAVING ALIGNED EXTRINSIC PARAMETERS")
        print(f"{'='*60}\n")
        
        for cam_id in self.transformations.keys():
            calib = self.mapper.cameras[cam_id]
            
            # Get original extrinsic parameters
            rvec = calib.rvec
            tvec = calib.tvec
            
            if cam_id == self.reference_camera:
                # Reference camera - no change needed
                aligned_rvec = rvec.copy()
                aligned_tvec = tvec.copy()
                print(f"✓ {cam_id} (reference): No transformation needed")
            else:
                # The transformation we found is: P_ref = R(theta) * P_cam + [tx, ty]
                # We need to modify the extrinsics so that pixel_to_world() gives P_ref directly
                
                tx, ty, theta = self.transformations[cam_id]
                
                # Create 2D rotation matrix for alignment in XY plane
                R_align_2d = np.array([
                    [np.cos(theta), -np.sin(theta)],
                    [np.sin(theta),  np.cos(theta)]
                ], dtype=np.float32)
                
                # Convert to 3D (rotation around Z-axis)
                R_align_3d = np.eye(3, dtype=np.float32)
                R_align_3d[:2, :2] = R_align_2d
                
                # Get original rotation matrix (camera to world)
                R_orig, _ = cv2.Rodrigues(rvec)
                
                # The pixel_to_world transformation is:
                # P_world = R_inv * (s * ray_cam - tvec)
                # where R_inv = R_orig.T (since R is orthogonal)
                
                # We want: P_ref = R_align * P_world + t_align
                # So: P_ref = R_align * R_orig.T * (s * ray_cam - tvec) + t_align
                #           = (R_align * R_orig.T) * (s * ray_cam) - (R_align * R_orig.T * tvec) + t_align
                
                # New effective inverse rotation: R_new_inv = R_align * R_orig.T
                R_new_inv = R_align_3d @ R_orig.T
                
                # New rotation matrix (world to camera)
                R_new = R_new_inv.T
                aligned_rvec, _ = cv2.Rodrigues(R_new)
                
                # New translation: -(R_new @ (R_align * (-R_orig.T @ tvec) + t_align))
                # Simplified: we need to adjust tvec such that the origin shift is correct
                t_align_3d = np.array([[tx], [ty], [0]], dtype=np.float32)
                
                # The new tvec should account for both the rotation and translation
                # New world origin in camera frame: tvec_new = R_new @ t_align + tvec
                # But actually we want: -R_new.T @ tvec_new = -R_align @ R_orig.T @ tvec + t_align
                # So: tvec_new = -R_new @ (R_align @ (-R_orig.T @ tvec) + t_align)
                
                cam_origin_in_world = -R_orig.T @ tvec
                cam_origin_in_ref = R_align_3d @ cam_origin_in_world + t_align_3d
                aligned_tvec = -R_new @ cam_origin_in_ref
                
                print(f"✓ {cam_id}: Applied transformation")
                print(f"    Δtranslation: ({tx:.2f}, {ty:.2f}, 0) mm")
                print(f"    Δrotation:    {np.degrees(theta):.2f}° around Z-axis")
            
            # Save aligned extrinsic parameters
            output_path = os.path.join(output_dir, f"{cam_id}_extrinsic_aligned.yaml")
            
            # Use flatten().tolist() to match your save_extrinsic format
            data = {
                "CameraExt": {
                    "rvec": aligned_rvec.flatten().tolist(),
                    "tvec": aligned_tvec.flatten().tolist()
                },
                "reference_camera": self.reference_camera
            }
            
            with open(output_path, "w") as f:
                yaml.dump(data, f, sort_keys=False)
            
            print(f"    Saved to: {output_path}")
        
        print(f"\n✓ All aligned extrinsic parameters saved to '{output_dir}/'")
        print("\nTo use these aligned parameters:")
        print("  1. Load them with CameraCalibration.load_extrinsic()")
        print("  2. All cameras will now share the same world coordinate origin")
        print("  3. Use the SAME origin_px as the reference camera for all aligned cameras")
    
    def save_transformation_summary(self, output_path="output/alignment_summary.yaml"):
        """
        Save a summary of all transformations applied.
        
        This is useful for documentation and debugging.
        """
        import os
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        summary = {
            'reference_camera': self.reference_camera,
            'transformations': {}
        }
        
        for cam_id, (tx, ty, theta) in self.transformations.items():
            summary['transformations'][cam_id] = {
                'translation_mm': {'x': float(tx), 'y': float(ty)},
                'rotation_deg': float(np.degrees(theta)),
                'is_reference': (cam_id == self.reference_camera)
            }
        
        with open(output_path, 'w') as f:
            yaml.dump(summary, f, default_flow_style=False)
        
        print(f"\n✓ Transformation summary saved to: {output_path}")


# ============================================================
# USAGE EXAMPLE
# ============================================================

"""
UPDATED WORKFLOW WITH SAVING:
------------------------------

1. Calibrate alignment (as before):
   calibrator = CameraAlignmentCalibrator(mapper)
   calibrator.calibrate_with_reference_points(reference_pairs)

2. Verify alignment by projecting points:
   calibrator.project_aligned_point_to_map("cam2", 447, 617)
   calibrator.project_aligned_point_to_map("cam3", 830, 894)

3. Save the aligned extrinsic parameters:
   calibrator.save_aligned_extrinsics("output/aligned_extrinsics")
   calibrator.save_transformation_summary("output/alignment_summary.yaml")

4. In future sessions, load the aligned extrinsics directly:
   mapper = MultiCameraMapper(...)
   mapper.add_camera(
       camera_id="cam2",
       intrinsic_path="output/cam2/intrinsic.yaml",
       extrinsic_path="output/aligned_extrinsics/cam2_extrinsic_aligned.yaml",  # Use aligned!
       origin_px=(817, 296)
   )
   
   # Now all cameras share the same world origin - no need for calibrator!
   calib = mapper.cameras["cam2"]
   wx, wy, wz = calib.pixel_to_world(447, 617)
   map_u, map_v = mapper.world_to_map("cam2", wx, wy)  # Direct mapping!

WHAT GETS SAVED:
----------------
1. Aligned extrinsic files:
   - output/aligned_extrinsics/cam2_extrinsic_aligned.yaml
   - output/aligned_extrinsics/cam3_extrinsic_aligned.yaml
   - output/aligned_extrinsics/cam4_extrinsic_aligned.yaml
   
   Each contains:
   - rvec: Rotation vector (adjusted for alignment)
   - tvec: Translation vector (adjusted for alignment)
   - transformation_applied: Metadata about what was changed

2. Transformation summary:
   - output/alignment_summary.yaml
   
   Contains human-readable summary of all transformations

IMPORTANT NOTES:
----------------
- Intrinsic parameters (focal length, distortion, etc.) are NEVER changed
- Only extrinsic parameters (camera position/orientation) are adjusted
- The reference camera's extrinsic stays the same
- Other cameras' extrinsics are modified to align with reference
"""