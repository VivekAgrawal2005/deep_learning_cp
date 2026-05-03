import os
import json

dataset_path = "known-faces"
fallback_dataset_path = "dataset"
CONFIG_FILE = "config.json"

owners = set()

source_path = dataset_path if os.path.isdir(dataset_path) else fallback_dataset_path

if os.path.isdir(source_path):
    for file in os.listdir(source_path):
        if file.lower().endswith((".jpg", ".jpeg", ".png")):
            name = file.split("_")[0].strip().lower()
            owners.add(name)

print("Owners detected in dataset:")
for owner in sorted(owners):
    print(f"  - {owner}")

# BUG FIX: Save names to a shared config file so face_recognize.py
# and camera_api.py don't need to hardcode them separately.
# Previously "yash" was in the dataset but missing from known_names in both files.
config = {"known_names": sorted(list(owners))}
with open(CONFIG_FILE, "w") as f:
    json.dump(config, f, indent=2)

print(f"\nSaved {len(owners)} owner(s) to {CONFIG_FILE}")
print("Dataset ready for face recognition.")