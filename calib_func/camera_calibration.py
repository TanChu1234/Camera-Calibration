import cv2
import numpy as np
import yaml
import glob
import os
import math

class CameraCalibration:
    def __init__(self):
        self.mtx = None       # intrinsic matrix
        self.dist = None      # distortion coeffs
        self.rvec = None      # rotation vector (world→camera)
        self.tvec = None      # translation vector (world→camera)

    # ==============================================================
    #  INTRINSIC CALIBRATION
    # ==============================================================
    
    def calibrate_intrinsic(
        self,
        image_folder,
        chess_size=(9, 6),
        save_results=True,
        results_dir="calib_results",
        output_file="intrinsic.yaml"
    ):

        # 1) World coordinates of corners
        objp = np.zeros((chess_size[0] * chess_size[1], 3), np.float32)
        objp[:, :2] = np.mgrid[0:chess_size[0], 0:chess_size[1]].T.reshape(-1, 2)

        objpoints = []
        imgpoints = []

        # Load images
        extensions = ["*.jpg", "*.jpeg", "*.png", "*.bmp"]
        images = []
        if "*" in image_folder:
            images = glob.glob(image_folder)
        else:
            for ext in extensions:
                images.extend(glob.glob(os.path.join(image_folder, ext)))

        if len(images) == 0:
            raise ValueError("No calibration images found")

        if save_results:
            os.makedirs(results_dir, exist_ok=True)

        # 2) Detect corners
        for fname in images:
            img = cv2.imread(fname)
            if img is None:
                continue

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            ret, corners = cv2.findChessboardCorners(gray, chess_size)

            if not ret:
                continue

            objpoints.append(objp)
            imgpoints.append(corners)

            if save_results:
                drawn = cv2.drawChessboardCorners(img.copy(), chess_size, corners, True)
                cv2.imwrite(os.path.join(results_dir, os.path.basename(fname)), drawn)

        if len(imgpoints) == 0:
            raise ValueError("No valid chessboard detected!")

        # 3) Calibrate
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
            objpoints, imgpoints, gray.shape[::-1], None, None,
            flags=cv2.CALIB_RATIONAL_MODEL
        )

        self.mtx = mtx
        self.dist = dist

        # 4) Compute error
        total_err = 0
        for i in range(len(objpoints)):
            reprojected, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
            err = cv2.norm(imgpoints[i], reprojected, cv2.NORM_L2) / len(reprojected)
            total_err += err
        mean_err = total_err / len(objpoints)

        self.save_intrinsic(output_file)

        print(f"✓ Intrinsic saved to {output_file}")
        print(f"Reprojection error = {mean_err:.4f}")

        return mtx, dist, mean_err
    # --------------------------------------------------------------

    def save_intrinsic(self, output_file="intrinsic_params.yaml"):
        if self.mtx is None or self.dist is None:
            raise ValueError("Intrinsic parameters not available")

        d = self.dist.flatten().tolist()

        data = {
            'CameraInt': {
                'fx': float(self.mtx[0, 0]),
                'fy': float(self.mtx[1, 1]),
                'cx': float(self.mtx[0, 2]),
                'cy': float(self.mtx[1, 2]),
                'dist_coeffs': d
            }
        }

        with open(output_file, "w") as f:
            yaml.dump(data, f, sort_keys=False)

    def load_intrinsic(self, yaml_file):
        with open(yaml_file, "r") as f:
            data = yaml.safe_load(f)

        cam = data["CameraInt"]
        self.mtx = np.array([[cam['fx'], 0, cam['cx']],
                             [0, cam['fy'], cam['cy']],
                             [0, 0, 1]], dtype=np.float32)

        self.dist = np.array(cam['dist_coeffs'], dtype=np.float32).reshape(1, -1)
        print("✓ Intrinsic loaded.")

    # ==============================================================
    #  EXTRINSIC CALIBRATION (ORIGIN = CHESSBOARD CORNER)
    # ==============================================================

    def calibrate_extrinsic(self, image_path, chess_size=(9, 6), square_size=25):
        """Compute rotation/translation from chessboard."""

        if self.mtx is None:
            raise ValueError("Run intrinsic calibration first!")

        img = cv2.imread(image_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        ret, corners = cv2.findChessboardCorners(gray, chess_size)
        if not ret:
            raise ValueError("Chessboard not found in extrinsic image")

        # World coordinates (origin = first corner)
        objp = np.zeros((chess_size[0]*chess_size[1], 3), np.float32)
        objp[:, :2] = (np.mgrid[0:chess_size[0], 0:chess_size[1]]
                       .T.reshape(-1, 2) * square_size)

        # Improve corner accuracy
        corners2 = cv2.cornerSubPix(
            gray, corners, (11, 11), (-1, -1),
            (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))

        # Solve PnP (chessboard → camera)
        retval, rvec, tvec = cv2.solvePnP(objp, corners2, self.mtx, self.dist)


        self.rvec = rvec
        self.tvec = tvec

        print("✓ Extrinsic calibrated (origin at chessboard corner)")
        print("Rvec:", rvec.T)
        print("Tvec:", tvec.T)

        return rvec, tvec

    def save_extrinsic(self, file="extrinsic_params.yaml"):
        if self.rvec is None:
            raise ValueError("Extrinsic not computed yet")

        data = {
            "CameraExt": {
                "rvec": self.rvec.flatten().tolist(),
                "tvec": self.tvec.flatten().tolist()
            }
        }
        with open(file, "w") as f:
            yaml.dump(data, f, sort_keys=False)

    def load_extrinsic(self, file="extrinsic_params.yaml"):
        with open(file, "r") as f:
            data = yaml.safe_load(f)

        ext = data["CameraExt"]
        self.rvec = np.array(ext["rvec"], dtype=np.float32).reshape(3, 1)
        self.tvec = np.array(ext["tvec"], dtype=np.float32).reshape(3, 1)

        print("✓ Extrinsic loaded.")

    # ==============================================================
    #  PROJECTION FUNCTIONS
    # ==============================================================

    def world_to_pixel(self, xyz):
        """Project world coordinate to pixel."""
        xyz = np.array(xyz, dtype=np.float32).reshape(3, 1)
        img_pt, _ = cv2.projectPoints(xyz, self.rvec, self.tvec,
                                      self.mtx, self.dist)
        print(img_pt[0, 0])
        
        return tuple(img_pt[0, 0])
    
    def pixel_to_world(self, u, v, z_world=0):
        """Convert pixel → world coordinate assuming Z = z_world."""
        if self.rvec is None:
            raise ValueError("Extrinsic not loaded")
        
        # KEY FIX: Undistort the pixel first!
        distorted_point = np.array([[[u, v]]], dtype=np.float32)
        normalized_point = cv2.undistortPoints(
            distorted_point,
            self.mtx,
            self.dist,
            P=None  # Return normalized coordinates
        )
        
        # Get normalized camera coordinates
        x_norm = normalized_point[0, 0, 0]
        y_norm = normalized_point[0, 0, 1]
        ray_cam = np.array([[x_norm], [y_norm], [1.0]], dtype=np.float32)
        
        # Transform ray to world coordinates
        R, _ = cv2.Rodrigues(self.rvec)
        R_inv = np.linalg.inv(R)
        cam_pos = -R_inv.dot(self.tvec)
        ray_world = R_inv.dot(ray_cam)
        
        # Intersection with Z = z_world
        if abs(ray_world[2, 0]) < 1e-6:
            raise ValueError("Ray is parallel to Z=0 plane")
        
        s = (z_world - cam_pos[2, 0]) / ray_world[2, 0]
        Pw = cam_pos + s * ray_world
        Pw = Pw.reshape(3)
        
        print(f"Pixel ({u:.2f}, {v:.2f}) → World ({Pw[0]:.2f}, {Pw[1]:.2f}, {Pw[2]:.2f})")
        
        return float(Pw[0]), float(Pw[1]), float(Pw[2])

    # ==============================================================
    # Draw world axes at origin
    # ==============================================================
    def _to_int_tuple(self, p):
        """Convert projected point (float32) to (int, int)."""
        return (int(p[0]), int(p[1]))
    
    def draw_axes(self, image_path, axis_length=400):
        if self.rvec is None:
            raise ValueError("Extrinsic parameters not loaded")

        img = cv2.imread(image_path)
        if img is None:
            raise FileNotFoundError(image_path)

        # World axes endpoints
        axis_3d = np.float32([
            [0, 0, 0],                 # Origin
            [axis_length, 0, 0],       # X axis
            [0, axis_length, 0],       # Y axis
            [0, 0, axis_length]        # Z axis
        ])

        imgpts, _ = cv2.projectPoints(axis_3d, self.rvec, self.tvec, self.mtx, self.dist)

        # Convert to int tuples
        origin = self._to_int_tuple(imgpts[0][0])
        x_axis = self._to_int_tuple(imgpts[1][0])
        y_axis = self._to_int_tuple(imgpts[2][0])
        z_axis = self._to_int_tuple(imgpts[3][0])

        # Draw axes
        cv2.line(img, origin, x_axis, (0, 0, 255), 3)     # X - red
        cv2.line(img, origin, y_axis, (0, 255, 0), 3)     # Y - green
        cv2.line(img, origin, z_axis, (255, 0, 0), 3)     # Z - blue

        return img

    def shift_origin(self, dx=0, dy=0, dz=0):
        """
        Shift world origin by (dx, dy, dz) mm.
        
        +dx → move origin in +X world direction  
        +dy → move origin in +Y world direction  
        +dz → move origin in +Z world direction (toward camera)
        """
        if self.rvec is None:
            raise ValueError("Extrinsic parameters not loaded")

        R, _ = cv2.Rodrigues(self.rvec)

        delta = np.array([[dx], [dy], [dz]], dtype=np.float32)
        self.tvec = self.tvec + R.dot(delta)

        print(f"✓ Origin shifted by (dx={dx}, dy={dy}, dz={dz}) mm")
        print("New Tvec:", self.tvec.T)
    
    # --------------------
    # Rotation helpers
    # --------------------
    def _euler_to_rotmat(self, rx, ry, rz, degrees=True, order='xyz'):
        """Return rotation matrix from Euler angles (rx,ry,rz).
        rx,ry,rz are rotations about X,Y,Z respectively.
        order string defines application order; default 'xyz' means apply Rx then Ry then Rz.
        """
        if degrees:
            rx = math.radians(rx)
            ry = math.radians(ry)
            rz = math.radians(rz)

        # Rotation about X
        Rx = np.array([[1, 0, 0],
                       [0, math.cos(rx), -math.sin(rx)],
                       [0, math.sin(rx),  math.cos(rx)]], dtype=np.float32)

        # Rotation about Y
        Ry = np.array([[ math.cos(ry), 0, math.sin(ry)],
                       [0, 1, 0],
                       [-math.sin(ry), 0, math.cos(ry)]], dtype=np.float32)

        # Rotation about Z
        Rz = np.array([[math.cos(rz), -math.sin(rz), 0],
                       [math.sin(rz),  math.cos(rz), 0],
                       [0, 0, 1]], dtype=np.float32)

        # Compose according to order: if order='xyz' we want R = Rz @ Ry @ Rx
        # (apply Rx, then Ry, then Rz)
        order = order.lower()
        mapping = {'x': Rx, 'y': Ry, 'z': Rz}
        R_total = np.eye(3, dtype=np.float32)
        # apply in sequence: first letter = first rotation applied
        for axis in order:
            R_total = mapping[axis] @ R_total

        # BUT above yields R_total = R_ordered @ I; to get same semantics (apply Rx then Ry then Rz)
        # it's clearer to compute explicitly for common orders:
        if order == 'xyz':
            R_total = Rz @ Ry @ Rx
        elif order == 'xzy':
            R_total = Ry @ Rz @ Rx
        elif order == 'yxz':
            R_total = Rz @ Rx @ Ry
        elif order == 'yzx':
            R_total = Rx @ Rz @ Ry
        elif order == 'zxy':
            R_total = Ry @ Rx @ Rz
        elif order == 'zyx':
            R_total = Rx @ Ry @ Rz
        # otherwise fallback to previous composition
        return R_total

    def rotate_origin(self, rx=0, ry=0, rz=0, degrees=True, order='xyz'):
        """
        Rotate the WORLD FRAME by Euler angles (rx,ry,rz) about the world origin.
        This updates the camera rotation rvec (world->camera). Translation tvec is unchanged.
        Positive angles follow right-hand rule.
        - rx: rotation about World X (degrees if degrees=True)
        - ry: rotation about World Y
        - rz: rotation about World Z
        order defines application sequence (default 'xyz' applies Rx then Ry then Rz).
        """
        if self.rvec is None:
            raise ValueError("Extrinsic parameters not loaded")

        # current R (world -> camera)
        R_curr, _ = cv2.Rodrigues(self.rvec)

        # rotation applied to world frame (world_new = R_delta * world_old)
        R_delta = self._euler_to_rotmat(rx, ry, rz, degrees=degrees, order=order)

        # new rotation: R_new = R_curr * R_delta
        R_new = R_curr @ R_delta

        # convert back to rvec
        self.rvec, _ = cv2.Rodrigues(R_new)
        print(f"✓ World rotated by (rx={rx}, ry={ry}, rz={rz}) [{order}]")
        print("New Rvec:", self.rvec.T)

    def undistort_points_to_image(self, camera_pixels):
        """
        Convert distorted pixels → undistorted image pixels (same size as original)
        """
        points = np.array(camera_pixels, dtype=np.float32).reshape(-1, 1, 2)
        # Undistort points into image coordinates
        undistorted = cv2.undistortPoints(points, self.mtx, self.dist, P=self.mtx)
        undistorted = undistorted.reshape(-1, 2)
        return undistorted
    
    def undistort_image(self, image_path, output_path=None, show_comparison=False):
        """
        Undistort a single image using the calibrated intrinsic parameters.
        
        Args:
            image_path: Path to the input image
            output_path: Path to save undistorted image (optional)
            show_comparison: If True, return side-by-side comparison
        
        Returns:
            Undistorted image (numpy array)
        """
        if self.mtx is None or self.dist is None:
            raise ValueError("Intrinsic parameters not loaded. Run calibrate_intrinsic() first.")
        
        img = cv2.imread(image_path)
        if img is None:
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        h, w = img.shape[:2]
        
        # Get optimal new camera matrix
        newcameramtx, roi = cv2.getOptimalNewCameraMatrix(
            self.mtx, self.dist, (w, h), 1, (w, h))
        
        # Undistort
        dst = cv2.undistort(img, self.mtx, self.dist, None, newcameramtx)
        
        # Crop the image (optional)
        x, y, w, h = roi
        dst = dst[y:y+h, x:x+w]
        
        # Save if output path provided
        if output_path:
            cv2.imwrite(output_path, dst)
            print(f"✓ Undistorted image saved to {output_path}")
        
        return dst

    