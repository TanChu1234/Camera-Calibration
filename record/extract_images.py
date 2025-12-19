import cv2
import os

video_path = "recordings/camera_3_20251217_092140.avi"      # path to your AVI file
output_dir = "camera_4/camera4_v4"
os.makedirs(output_dir, exist_ok=True)

cap = cv2.VideoCapture(video_path)

frame_id = 0
save_every = 15   # ⬅️ increase this number = fewer images
saved = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    if frame_id % save_every == 0:
        cv2.imwrite(
            os.path.join(output_dir, f"frame_{saved:05d}.jpg"),
            frame
        )
        saved += 1

    frame_id += 1

cap.release()
print(f"Saved {saved} images")
