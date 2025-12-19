import cv2
import numpy as np
from calib_func.camera_calibration import CameraCalibration


class Map:
    def __init__(self, map_image_path, mm_per_pixel_x, mm_per_pixel_y):
        self.map_image_path = map_image_path
        self.mm_per_pixel_x = mm_per_pixel_x
        self.mm_per_pixel_y = mm_per_pixel_y

        self.cameras = {}
        self.camera_origins = {}
        self.map_img = None
        self.all_projected_points = []

    # ---------- Basic ----------
    def _initialize_map(self):
        if self.map_img is None:
            self.map_img = cv2.imread(self.map_image_path)
            if self.map_img is None:
                raise ValueError("Cannot load map image")

    def add_camera(self, camera_id, intrinsic_path, extrinsic_path, origin_px):
        calib = CameraCalibration()
        calib.load_intrinsic(intrinsic_path)
        calib.load_extrinsic(extrinsic_path)

        self.cameras[camera_id] = calib
        self.camera_origins[camera_id] = origin_px

        print(f"✓ Camera '{camera_id}' added at {origin_px}")
        
    def add_camera_points(self, camera_points_dict, z_world=0):
        for camera_id, points in camera_points_dict.items():
            if camera_id not in self.cameras:
                raise ValueError(f"Camera '{camera_id}' not registered")

            calib: CameraCalibration = self.cameras[camera_id]

            for point_id, u, v in points:
                wx, wy, wz = calib.pixel_to_world(u, v, z_world)
                map_u, map_v = self.world_to_map(camera_id, wx, wy)

                self.all_projected_points.append({
                    "camera": camera_id,
                    "point_id": int(point_id),  
                    "map_px": (map_u, map_v),
                    "world_mm": (wx, wy, wz)
                })
    # ---------- Coordinate ----------
    def world_to_map(self, camera_id, X_mm, Y_mm):
        u0, v0 = self.camera_origins[camera_id]
        u = int(u0 + X_mm / self.mm_per_pixel_x)
        v = int(v0 + Y_mm / self.mm_per_pixel_y)
        return (u, v)

    def map_to_world(self, camera_id, u, v):
        u0, v0 = self.camera_origins[camera_id]
        X = (u - u0) * self.mm_per_pixel_x
        Y = (v - v0) * self.mm_per_pixel_y
        return X, Y

    # ---------- Drawing ----------
    def draw_point(self, pt_px, color, radius=3):
        self._initialize_map()
        cv2.circle(self.map_img, pt_px, radius, color, -1)

    def draw_text(self, text, pt_px, color):
        self._initialize_map()
        cv2.putText(
            self.map_img,
            text,
            pt_px,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2
        )

    # ---------- Projection ----------
    def project_camera_points_to_map(
        self,
        camera_id,
        camera_pixels,
        z_world=0,
        color=(0, 255, 0),
        store_for_merge=True
    ):
        calib = self.cameras[camera_id]
        results = []

        for i, (u, v) in enumerate(camera_pixels):
            wx, wy, wz = calib.pixel_to_world(u, v, z_world)
            map_u, map_v = self.world_to_map(camera_id, wx, wy)

            self.draw_point((map_u, map_v), color)
            self.draw_text(f"{camera_id}{i+1}", (map_u + 8, map_v), color)

            results.append((map_u, map_v, wx, wy, wz))

            if store_for_merge:
                self.all_projected_points.append({
                    "camera": camera_id,
                    "map_px": (map_u, map_v),
                    "world_mm": (wx, wy, wz)
                })

        return results

    # ---------- Merge ----------
    def merge_points(self, distance_thresh_mm=130, reference_camera=None):
        if not self.all_projected_points:
            return []

        # mm → pixel threshold
        thresh_px = np.sqrt(
            (distance_thresh_mm / self.mm_per_pixel_x) ** 2 +
            (distance_thresh_mm / self.mm_per_pixel_y) ** 2
        )

        clusters = []

        for p in self.all_projected_points:
            pu, pv = p["map_px"]
            pid = p["point_id"]          # ⭐ numeric ID

            added = False
            for c in clusters:
                # ❌ DO NOT merge if ID is different
                if c["point_id"] != pid:
                    continue

                cu, cv = c["center"]
                if np.hypot(pu - cu, pv - cv) < thresh_px:
                    c["points"].append(p)

                    # update cluster center
                    us = [pt["map_px"][0] for pt in c["points"]]
                    vs = [pt["map_px"][1] for pt in c["points"]]
                    c["center"] = (np.mean(us), np.mean(vs))

                    added = True
                    break

            if not added:
                clusters.append({
                    "point_id": pid,      # ⭐ store ID in cluster
                    "center": (pu, pv),
                    "points": [p]
                })

        if reference_camera is None:
            reference_camera = list(self.camera_origins.keys())[0]

        merged = []
        for c in clusters:
            cu, cv = c["center"]
            wx, wy = self.map_to_world(
                reference_camera,
                int(round(cu)),
                int(round(cv))
            )

            merged.append({
                "point_id": c["point_id"],
                "map_px": (int(round(cu)), int(round(cv))),
                "world_mm": (wx, wy),
                "sources": [p["camera"] for p in c["points"]],
                "num_sources": len(set(p["camera"] for p in c["points"])),
                "source_points": c["points"]
            })

        return merged

    # ---------- Output ----------
    def draw_merged_points(self, merged_points, color=(255, 0, 255)):
        self._initialize_map()
        for i, p in enumerate(merged_points):
            u, v = p["map_px"]
            cv2.circle(self.map_img, (u, v), 3, color, 2)
            self.draw_text(f"M{i+1}", (u + 10, v - 5), color)

    def save_and_show(self, output_path):
        import matplotlib.pyplot as plt

        self._initialize_map()
        cv2.imwrite(output_path, self.map_img)

        plt.figure(figsize=(12, 10))
        plt.imshow(cv2.cvtColor(self.map_img, cv2.COLOR_BGR2RGB))
        plt.axis("off")
        plt.show()
