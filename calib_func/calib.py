import cv2
from calib_func.camera_calibration import CameraCalibration

# -----------Cam 1----------
INTRINSIC_IMAGES = "C:/Users/admin/project/calibcam/images/camera_1/calib_v3.0/*.jpeg"
EXTRINSIC_IMAGE = "C:/Users/admin/project/calibcam/images/camera_1/4k-test/IMG_20251206_125324.jpeg"

# -----------Cam 2----------
# INTRINSIC_IMAGES = "C:/Users/admin/project/calibcam/images/camera_2/raw_2/*.jpg"
# EXTRINSIC_IMAGE = "C:/Users/admin/project/calibcam/images/camera_2/raw_2/camera2_v7_00007.jpg"

# -----------Cam 3----------
# INTRINSIC_IMAGES = "C:/Users/admin/project/calibcam/images/camera_3/raw_2/*.jpg"
# EXTRINSIC_IMAGE = "C:/Users/admin/project/calibcam/images/camera_3/raw_2/camera3_v5_00004.jpg"

# -----------Cam 4----------
# INTRINSIC_IMAGES = "C:/Users/admin/project/calibcam/images/camera_4/raw_2/*.jpg"
# EXTRINSIC_IMAGE = "C:/Users/admin/project/calibcam/images/camera_4/raw_2/camera4_v3_00014.jpg"

CHESS_SIZE = (9, 6)
SQUARE_SIZE = 24

INTRINSIC_FILE = "intrinsic.yaml"
EXTRINSIC_FILE = "extrinsic.yaml"
AXIS_OUTPUT = "output/axis_visualization.jpg"


if __name__ == "__main__":
    calib = CameraCalibration()

    print("\n=== Intrinsic Calibration ===")
    calib.calibrate_intrinsic(
        image_folder=INTRINSIC_IMAGES,
        chess_size=CHESS_SIZE,
        save_results=True,
        results_dir="intrinsic_results",
        output_file=INTRINSIC_FILE,
    )

    print("\n=== Extrinsic Calibration ===")
    calib.calibrate_extrinsic(
        image_path=EXTRINSIC_IMAGE,
        chess_size=CHESS_SIZE,
        square_size=SQUARE_SIZE,
    )
    calib.save_extrinsic(EXTRINSIC_FILE)
    # -----------Cam 1----------
    # calib.shift_origin(dz=750)
    calib.shift_origin(dx=-5394.89, dy=-3011.68, dz=750)
    calib.rotate_origin(rz=-89)
    
    # -----------Cam 2----------
    # calib.shift_origin(dz=1825)
    # calib.shift_origin(dx=6384.75, dy=-763.58, dz=1825)
    # calib.rotate_origin(rz=5)
    
    # -----------Cam 3----------
    # calib.shift_origin(dz=1820)
    # calib.shift_origin(dx=2041.16, dy=-892.39, dz=1820)
    # calib.rotate_origin(rz=180)
    
    # -----------Cam 4----------
    # calib.shift_origin(dz=10)
    # calib.shift_origin(dx=1049.18, dy=-169.49, dz=10)
    # calib.rotate_origin(rz=-92)
    # calib.save_extrinsic(EXTRINSIC_FILE)

    print("\n=== Pixel → World Test ===")
    # -----------Cam 1----------
    w = calib.pixel_to_world(1243, 501, z_world=0)
    
    # -----------Cam 2----------
    # w = calib.pixel_to_world(436, 497, z_world=0)
    
    # -----------Cam 3----------
    # w = calib.pixel_to_world(703, 701, z_world=0)
    
    # -----------Cam 4----------
    # w = calib.pixel_to_world(356, 1173, z_world=0)
    p = calib.world_to_pixel(w)
    print("World:", w)
    print("Back to pixel:", p)

    img = calib.draw_axes(EXTRINSIC_IMAGE, axis_length=100)
    cv2.imwrite(AXIS_OUTPUT, img)
    print("Saved:", AXIS_OUTPUT)
