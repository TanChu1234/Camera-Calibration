from map.map import Map
from calib_func.calib_multicamera import CameraAlignmentCalibrator
import numpy as np

if __name__ == "__main__":
    # ============================================================
    # STEP 1: Initialize Map and Add All Cameras
    # ============================================================
    mapper = Map(
        map_image_path="C:/Users/admin/project/calibcam/map1/3d_layout.png",
        mm_per_pixel_x=23.8,
        mm_per_pixel_y=23.3
    )
    
    # Add all cameras with their ORIGINAL extrinsics
    mapper.add_camera(
        camera_id="cam1",
        intrinsic_path="output/cam1/intrinsic.yaml",
        extrinsic_path="output/cam1/extrinsic.yaml",
        origin_px=(234, 389)
    )
    
    mapper.add_camera(
        camera_id="cam2",
        intrinsic_path="output/cam2/intrinsic.yaml",
        extrinsic_path="output/cam2/extrinsic.yaml",
        origin_px=(817, 296)
    )
    
    mapper.add_camera(
        camera_id="cam3",
        intrinsic_path="output/cam3/intrinsic.yaml",
        extrinsic_path="output/cam3/extrinsic.yaml",
        origin_px=(678, 382)
    )
    
    mapper.add_camera(
        camera_id="cam4",
        intrinsic_path="output/cam4/intrinsic.yaml",
        extrinsic_path="output/cam4/extrinsic.yaml",
        origin_px=(914, 173)
    )
    
    # ============================================================
    # STEP 2: Define Reference Points
    # ============================================================
    reference_pairs = [
        {
            'cam2': (447, 617),
            'cam3': (830, 894),
            'cam4': (783, 430)
        },
        {
            'cam2': (816, 641),
            'cam3': (1427, 1020),
            'cam4': (811, 509)
        }
    ]
    
    # ============================================================
    # STEP 3: Calibrate Alignment
    # ============================================================
    print("\n" + "="*70)
    print("CALIBRATING CAMERA ALIGNMENT")
    print("="*70)
    
    calibrator = CameraAlignmentCalibrator(mapper, "cam3")
    calibrator.calibrate_with_reference_points(reference_pairs)
    
    # ============================================================
    # STEP 4: Verify Alignment (Optional but Recommended)
    # ============================================================
    print("\n" + "="*70)
    print("VERIFYING REFERENCE POINT ALIGNMENT")
    print("="*70)
    
    print("\n--- Reference Point A ---")
    calibrator.project_aligned_point_to_map(
        "cam2", 447, 617, color=(0, 255, 0), label="Ref_A"
    )
    calibrator.project_aligned_point_to_map(
        "cam3", 830, 894, color=(0, 255, 0), label="Ref_A"
    )
    calibrator.project_aligned_point_to_map(
        "cam4", 783, 430, color=(0, 255, 0), label="Ref_A"
    )
    
    # ============================================================
    # STEP 5: SAVE ALIGNED EXTRINSIC PARAMETERS
    # ============================================================
    print("\n" + "="*70)
    print("SAVING ALIGNED EXTRINSIC PARAMETERS")
    print("="*70)
    
    # Save aligned extrinsics
    calibrator.save_aligned_extrinsics("aligned_extrinsics")
    
    # Save transformation summary for documentation
    calibrator.save_transformation_summary("output/alignment_summary.yaml")
    
    # ============================================================
    # STEP 6: Project All Points (Optional - for visualization)
    # ============================================================
    print("\n" + "="*70)
    print("PROJECTING ALL POINTS WITH ALIGNMENT")
    print("="*70)
    
    # Camera 2 points
    cam2_pixels = [(447, 617), (816, 641)]
    for i, (u, v) in enumerate(cam2_pixels):
        calibrator.project_aligned_point_to_map(
            "cam2", u, v, color=(0, 255, 0), label=f"C2_P{i+1}"
        )
    
    # Camera 3 points
    cam3_pixels = [(830, 894), (1427, 1020)]
    for i, (u, v) in enumerate(cam3_pixels):
        calibrator.project_aligned_point_to_map(
            "cam3", u, v, color=(0, 0, 255), label=f"C3_P{i+1}"
        )
    
    # Camera 4 points
    cam4_pixels = [(783, 430), (811, 509)]
    for i, (u, v) in enumerate(cam4_pixels):
        calibrator.project_aligned_point_to_map(
            "cam4", u, v, color=(255, 255, 0), label=f"C4_P{i+1}"
        )
    
    # ============================================================
    # STEP 7: Save Visualization
    # ============================================================
    mapper.save_and_show("output/aligned_merged_map.png")
    
    # ============================================================
    # NEXT STEPS
    # ============================================================
    print("\n" + "="*70)
    print("✓ ALIGNMENT COMPLETE!")
    print("="*70)
    print("\nFiles saved:")
    print("  1. output/aligned_extrinsics/cam2_extrinsic_aligned.yaml")
    print("  2. output/aligned_extrinsics/cam3_extrinsic_aligned.yaml")
    print("  3. output/aligned_extrinsics/cam4_extrinsic_aligned.yaml")
    print("  4. output/alignment_summary.yaml")
    print("  5. output/aligned_merged_map.png")
    
    print("\nTo use aligned cameras in future (without calibrator):")
    print("  mapper.add_camera(")
    print("      camera_id='cam2',")
    print("      intrinsic_path='output/cam2/intrinsic.yaml',")
    print("      extrinsic_path='output/aligned_extrinsics/cam2_extrinsic_aligned.yaml',")
    print("      origin_px=(817, 296)")
    print("  )")
    print("\n  # Now all cameras share the same world origin!")
    print("  wx, wy, wz = mapper.cameras['cam2'].pixel_to_world(u, v)")
    print("  map_u, map_v = mapper.world_to_map('cam2', wx, wy)")