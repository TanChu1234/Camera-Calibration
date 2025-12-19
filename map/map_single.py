import cv2
import math
import matplotlib.pyplot as plt

class Map2D:
    def __init__(self, map_image_path, origin_px, mm_per_pixel_x, mm_per_pixel_y):
        """
        map_image_path: path to map image
        origin_px: (u0, v0) pixel coordinates of origin
        mm_per_pixel_x: millimeters per pixel along X
        mm_per_pixel_y: millimeters per pixel along Y
        """
        self.map_img = cv2.imread(map_image_path)
        if self.map_img is None:
            raise ValueError("Cannot load map image")

        self.origin_px = origin_px
        self.mm_per_pixel_x = mm_per_pixel_x
        self.mm_per_pixel_y = mm_per_pixel_y

    # -------------------------------------------------------
    # Coordinate Transformations
    # -------------------------------------------------------
    def world_to_map(self, X_mm, Y_mm):
        """Convert real-world mm → pixel coordinates (+X→right, +Y→down)"""
        u0, v0 = self.origin_px
        u = int(u0 + X_mm / self.mm_per_pixel_x)
        v = int(v0 + Y_mm / self.mm_per_pixel_y)
        return (u, v)

    def map_to_world(self, u, v):
        """Convert pixel coordinates → real-world mm (+X→right, +Y→down)"""
        u0, v0 = self.origin_px
        X_mm = (u - u0) * self.mm_per_pixel_x
        Y_mm = (v - v0) * self.mm_per_pixel_y
        return (X_mm, Y_mm)

    # -------------------------------------------------------
    # Drawing utilities
    # -------------------------------------------------------
    def draw_point(self, pt_px, color=(0, 0, 255), radius=4, thickness=-1):
        cv2.circle(self.map_img, pt_px, radius, color, thickness)

    def draw_line(self, pt1_px, pt2_px, color=(255, 255, 0), thickness=2):
        cv2.line(self.map_img, pt1_px, pt2_px, color, thickness)

    def draw_distance(self, pt1_px, pt2_px):
        dx_px = pt2_px[0] - pt1_px[0]
        dy_px = pt2_px[1] - pt1_px[1]

        dx_mm = dx_px * self.mm_per_pixel_x
        dy_mm = dy_px * self.mm_per_pixel_y
        dist_mm = math.sqrt(dx_mm**2 + dy_mm**2)
        print(dist_mm)
        # mid_x = (pt1_px[0] + pt2_px[0]) // 2
        # mid_y = (pt1_px[1] + pt2_px[1]) // 2

        # info = f"dx={dx_mm:.1f}mm  dy={dy_mm:.1f}mm  dist={dist_mm:.1f}mm"
        # cv2.putText(self.map_img, info, (mid_x, mid_y),
        #             cv2.FONT_HERSHEY_SIMPLEX, 0.5,
        #             (255, 255, 255), 2)

    def draw_text(self, text, pt_px, color=(0, 0, 0)):
        cv2.putText(self.map_img, text, pt_px,
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    color, 2)
            
    # -------------------------------------------------------
    # Origin + Axes
    # -------------------------------------------------------
    def draw_origin(self):
        cv2.circle(self.map_img, self.origin_px, 5, (0, 0, 255), -1)
        self.draw_text("Origin (0,0)", (self.origin_px[0] + 15, self.origin_px[1] - 10))

    def draw_axes(self, axis_length_mm=200):
        ox, oy = self.origin_px

        # +X → right
        x_end = self.world_to_map(axis_length_mm, 0)

        # +Y → down
        y_end = self.world_to_map(0, axis_length_mm)

        cv2.arrowedLine(self.map_img, (ox, oy), x_end, (0, 0, 255), 3, tipLength=0.03)
        cv2.arrowedLine(self.map_img, (ox, oy), y_end, (0, 255, 0), 3, tipLength=0.03)

        self.draw_text("+X", (x_end[0] + 5, x_end[1]))
        self.draw_text("+Y", (y_end[0], y_end[1] - 5))

    # -------------------------------------------------------
    # Display / Save
    # -------------------------------------------------------
   
    def show(self, window_name="Map2D"):
        plt.figure(figsize=(8, 8))
        plt.title(window_name)

        # Convert BGR → RGB for matplotlib
        img_rgb = cv2.cvtColor(self.map_img, cv2.COLOR_BGR2RGB)

        plt.imshow(img_rgb)
        plt.axis("off")   # hide axes
        plt.tight_layout()
        plt.show()

    def save(self, out_path="map_output.png"):
        cv2.imwrite(out_path, self.map_img)
        print(f"Saved: {out_path}")
