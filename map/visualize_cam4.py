from calib_func.camera_calibration import CameraCalibration
from map_single import Map2D
import numpy as np
import cv2

calib = CameraCalibration()
calib.load_intrinsic("intrinsic.yaml")
calib.load_extrinsic("extrinsic.yaml")

map2d = Map2D(
    map_image_path="3d_layout.png",
    origin_px=(914, 173),
    mm_per_pixel_x=23.75,
    mm_per_pixel_y=23.3,
)

map2d.draw_origin()
map2d.draw_axes(axis_length_mm=2000)  

# Example: Test with the pixel you had (517, 1109)
camera_pixels = [
    # (783, 430),
    (677, 472), 
    (811, 509),
    (783, 430),
    (844, 1423), 
    (522, 1090), 
]

# Load distorted image
img = cv2.imread( "C:/Users/admin/project/calibcam/images/camera_4/raw_2/camera4_v3_00014.jpg")

h, w = img.shape[:2]
# Get camera matrix 
newcameramtx, roi = cv2.getOptimalNewCameraMatrix(calib.mtx, calib.dist, (w, h), 1, (w, h))

# Undistort image
undistorted_image = cv2.undistort(img, calib.mtx, calib.dist, None, newcameramtx)

# Convert pixel distorted → pixel undistorted
points = np.array(camera_pixels, dtype=np.float32).reshape(-1, 1, 2)
undistorted_points = cv2.undistortPoints(points, calib.mtx, calib.dist, P=newcameramtx)
undistorted_points = undistorted_points.reshape(-1, 2)

for ux, uy in undistorted_points:
    cv2.circle(undistorted_image, (int(ux), int(uy)), 6, (0,255,0), -1)  # xanh lá
cv2.imwrite("output/undistorted_points.png", undistorted_image)
    
# Vẽ pixel gốc (đỏ) và khử distortion (xanh lá)
for (x, y), (ux, uy) in zip(camera_pixels, undistorted_points):
    cv2.circle(img, (int(x), int(y)), 8, (0,0,255), -1)    # red: distorted

cv2.imwrite("output/distorted_points.png", img)

# Store map pixel coordinates for distance calculation
map_points = []

for i, (cam_u, cam_v) in enumerate(camera_pixels):
    # Convert camera pixel to world coordinates
    world_x, world_y, world_z = calib.pixel_to_world(cam_u, cam_v, z_world=0)
    print(f"\nPoint {i+1}:")
    print(f"  Camera pixel: ({cam_u}, {cam_v})")
    print(f"  World coord:  ({world_x:.2f}, {world_y:.2f}, {world_z:.2f}) mm")
    
    # Convert world coordinates to map pixel
    map_u, map_v = map2d.world_to_map(world_x, world_y)
    print(f"  Map pixel:    ({map_u}, {map_v})")
    
    # Store for distance calculation
    map_points.append((map_u, map_v))
    
    # Draw on map
    map2d.draw_point((map_u, map_v), color=(0, 0, 255), radius=5)
    map2d.draw_text(f"P{i+1}", (map_u + 15, map_v), color=(0, 255, 0))

# Draw distances between consecutive points
print("\n=== Drawing Distances ===")
for i in range(len(map_points) - 1):
    pt1 = map_points[i]
    pt2 = map_points[i + 1]
    
    # Draw line between points
    map2d.draw_line(pt1, pt2, color=(255, 165, 0), thickness=2)  # Orange line
    
    # Draw distance annotation
    map2d.draw_distance(pt1, pt2)
    
    print(f"Distance P{i+1} → P{i+2} drawn")

# Optional: Draw distance from first to last point
if len(map_points) > 2:
    map2d.draw_line(map_points[0], map_points[-1], color=(255, 0, 255), thickness=2)  # Magenta
    map2d.draw_distance(map_points[0], map_points[-1])
    print(f"Distance P1 → P{len(map_points)} drawn")
    
# Step 4: Save and display result
print("\n=== Saving Result ===")
map2d.save("output/map_with_camera_points.png")
map2d.show()