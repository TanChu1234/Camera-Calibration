from map.map import Map

if __name__ == "__main__":
    # Initialize unified map
    mapper = Map(
        map_image_path="3d_layout.png",
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
    
    # Add Camera 2 with its origin at (817, 296)
    mapper.add_camera(
        camera_id="cam2",
        intrinsic_path="output/cam2/intrinsic.yaml",
        extrinsic_path="aligned_extrinsics/cam2_extrinsic_aligned.yaml",
        origin_px=(678, 382)
    )
    
    # Add Camera 3
    mapper.add_camera(
        camera_id="cam3",
        intrinsic_path="output/cam3/intrinsic.yaml",
        extrinsic_path="aligned_extrinsics/cam3_extrinsic_aligned.yaml",
        origin_px=(678, 382)
    )
    
    # Add Camera 4
    mapper.add_camera(
        camera_id="cam4",
        intrinsic_path="output/cam4/intrinsic.yaml",
        extrinsic_path="aligned_extrinsics/cam4_extrinsic_aligned.yaml",
        origin_px=(678, 382)
    )

    camera_points = {
        "cam1": [(1, 1292, 567), (2, 585, 1275), (3, 665, 1511), (4, 1001, 1393)],
        "cam2": [(5, 447, 617), (6, 816, 641)],
        "cam3": [(5, 830, 894), (6, 1427, 1020)],
        "cam4": [(5, 783, 430), (6, 811, 509)],
        }


    mapper.add_camera_points(camera_points)
    # Project all points to the map (store_for_merge=True by default)
    
    merged = mapper.merge_points(distance_thresh_mm=130)
    mapper.draw_merged_points(merged)
    mapper.save_and_show("output/merged_map.png")