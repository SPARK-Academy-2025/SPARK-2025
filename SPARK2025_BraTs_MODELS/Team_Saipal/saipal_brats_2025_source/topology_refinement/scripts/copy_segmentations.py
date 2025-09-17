import os
import shutil

def copy_segmentation_files(source_root, destination_dir):
    os.makedirs(destination_dir, exist_ok=True)
    for root, dirs, files in os.walk(source_root):
        for file in files:
            if file.endswith("-seg.nii.gz"):
                source_file_path = os.path.join(root, file)
                destination_file_path = os.path.join(destination_dir, file)

                if os.path.abspath(source_file_path) == os.path.abspath(destination_file_path):
                    print(f"Skipped (same file): {file}")
                    continue

                shutil.copy2(source_file_path, destination_file_path)
                print(f"Copied: {source_file_path} → {destination_file_path}")

if __name__ == "__main__":
    source_root = "data/raw/ssa"
    destination_dir = "data/raw/working/masks"
    copy_segmentation_files(source_root, destination_dir)
