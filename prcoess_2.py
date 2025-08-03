import os

SHAPENET_ROOT = "shapenet_download/ShapeNetCore.v2"
TARGET_FILENAME = "model_normalized.obj"

def clean_directory(root: str):
    for dirpath, _, filenames in os.walk(root):
        for fname in filenames:
            if fname != TARGET_FILENAME:
                file_path = os.path.join(dirpath, fname)
                try:
                    os.remove(file_path)
                    print(f"Deleted: {file_path}")
                except Exception as e:
                    print(f"Error deleting {file_path}: {e}")

if __name__ == "__main__":
    clean_directory(SHAPENET_ROOT)
