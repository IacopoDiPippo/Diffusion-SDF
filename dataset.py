from huggingface_hub import snapshot_download

# Download ONLY the .zip file (no repo files)
snapshot_download(
    repo_id="ShapeNet/ShapeNetCore-archive",
    repo_type="dataset",
    allow_patterns="ShapeNetCore.v2.zip",  # Downloads only this file
    local_dir="./shapenet_download",       # Saves to this folder
    token=""                  # Required for LFS files
)