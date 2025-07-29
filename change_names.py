import os
from pathlib import Path

def remove_headers(root_dir="data/grid_data/acronym/Mug"):
    # Process all subdirectories
    for subdir in Path(root_dir).glob("*"):
        if not subdir.is_dir():
            continue
            
        csv_file = subdir / "grid_gt.csv"
        if not csv_file.exists():
            continue
            
        # Read the file, skip first line (header)
        with open(csv_file, 'r') as f:
            lines = f.readlines()
        
        # Write back without first line
        with open(csv_file, 'w') as f:
            f.writelines(lines[1:])
        
        print(f"Removed header from: {csv_file}")

if __name__ == "__main__":
    remove_headers()