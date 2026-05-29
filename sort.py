import os
import json

# Path to dataset folders
dataset_dir = "dataset/valid"

# Load class-name mapping
with open("cat_to_name.json", "r") as f:
    cat_to_name = json.load(f)

# Rename folders
for folder in os.listdir(dataset_dir):
    old_path = os.path.join(dataset_dir, folder)

    # Skip non-directories
    if not os.path.isdir(old_path):
        continue

    # Get number of folder
    class_num = folder

    # Get class name
    class_name = cat_to_name[class_num]

    # Clean folder name for filesystem safety
    class_name = class_name.replace("/", "-")

    # New path
    new_path = os.path.join(dataset_dir, class_name)

    # Rename folder
    os.rename(old_path, new_path)

    print(f"Renamed {folder} -> {class_name}")

print("Done.")