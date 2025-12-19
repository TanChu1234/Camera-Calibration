from map.map import Map

if __name__ == "__main__":
    # Initialize unified map (no single origin - each camera has its own)
    mapper = Map(
        map_image_path="C:/Users/admin/project/calibcam/map1/3d_layout.png",
        mm_per_pixel_x=23.8,
        mm_per_pixel_y=23.3
    )
    
    # Add Camera 1 with its origin at (234, 389)
    mapper.add_camera(
        camera_id="cam1",
        intrinsic_path="output/cam1/intrinsic.yaml",
        extrinsic_path="output/cam1/extrinsic.yaml",
        origin_px=(234, 389)
    )
    
    # Add Camera 2 with its origin at (822, 295)
    mapper.add_camera(
        camera_id="cam2",
        intrinsic_path="output/cam2/intrinsic.yaml",
        extrinsic_path="aligned_extrinsics/cam2_extrinsic_aligned.yaml",
        origin_px=(817, 296)
    )
     # Add Camera 3 with its origin at (822, 295)
    mapper.add_camera(
        camera_id="cam3",
        intrinsic_path="output/cam3/intrinsic.yaml",
        extrinsic_path="aligned_extrinsics/cam3_extrinsic_aligned.yaml",
        origin_px=(817, 296)
    )
    
    # Add Camera 4 with its origin at (822, 295)
    mapper.add_camera(
        camera_id="cam4",
        intrinsic_path="output/cam4/intrinsic.yaml",
        extrinsic_path="aligned_extrinsics/cam4_extrinsic_aligned.yaml",
        origin_px=(817, 296)
    )
    # Define points from Camera 1
    cam1_pixels = [
        (1292, 567),
        (585, 1275),
        (665, 1511),
        (1001, 1393),
    ]
    
    # Define points from Camera 2
    cam2_pixels = [
        (447, 617),
        (816, 641),
    ]
    
    # Define points from Camera 3
    cam3_pixels = [
        # (1328, 828),
        (830, 894), 
        (1427, 1020), 
    ]
    
    # Define points from Camera 4
    cam4_pixels = [
        # (851, 444),
        (783, 430), 
        (811, 509), 
    ]
    
    print("PROJECTING CAMERA 1 POINTS")
    cam1_points = mapper.project_camera_points_to_map(
        camera_id="cam1",
        camera_pixels=cam1_pixels,
        z_world=0,
        color=(255, 0, 0),  # Blue
    )
    
    print("PROJECTING CAMERA 2 POINTS")
    mapper.project_camera_points_to_map(
        camera_id="cam2",
        camera_pixels=cam2_pixels,
        z_world=0,
        color=(0, 255, 0),  # Green

    )
    print("PROJECTING CAMERA 3 POINTS")
    mapper.project_camera_points_to_map(
        camera_id="cam3",
        camera_pixels=cam3_pixels,
        z_world=0,
        color=(0, 0, 255),  # Red

    )
    print("PROJECTING CAMERA 4 POINTS")
    mapper.project_camera_points_to_map(
        camera_id="cam4",
        camera_pixels=cam4_pixels,
        z_world=0,
        color=(255, 255, 0),  #Cyan

    )
    # Save and display merged map
    mapper.save_and_show("output/merged_map.png")