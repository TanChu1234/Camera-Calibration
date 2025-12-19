import os
import shutil

# source_folders = [
#     r"camera_2/camera2_v1",
#     r"camera_2/camera2_v2",
#     r"camera_2/camera2_v3",
#     r"camera_2/camera2_v4",
#     r"camera_2/camera2_v5",
#     r"camera_2/camera2_v6",
#     r"camera_2/camera2_v7",
# ]

# source_folders = [
#     r"camera_3/camera3_v1",
#     r"camera_3/camera3_v2",
#     r"camera_3/camera3_v3",
#     r"camera_3/camera3_v4",
#     r"camera_3/camera3_v5",
# ]

source_folders = [
    r"camera_4/camera4_v1",
    r"camera_4/camera4_v2",
    r"camera_4/camera4_v3",
]
output_folder = r"camera_4/camera4"
os.makedirs(output_folder, exist_ok=True)

for folder in source_folders:
    prefix = os.path.basename(folder.rstrip("/\\"))  # folder name

    for idx, filename in enumerate(sorted(os.listdir(folder))):
        src_path = os.path.join(folder, filename)

        if not os.path.isfile(src_path):
            continue

        name, ext = os.path.splitext(filename)
        new_name = f"{prefix}_{idx:05d}{ext}"

        dst_path = os.path.join(output_folder, new_name)
        shutil.copy2(src_path, dst_path)

print("✅ Images merged successfully")
