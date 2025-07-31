import os
import numpy as np
import trimesh
import point_cloud_utils as pcu
from tqdm import tqdm

# Constants
SHAPENET_mug_CATEGORY = "02691156"  # Official ShapeNet category ID for mugs
MODEL_FILE_PATH = "models/model_normalized.obj"  # Relative path from object ID directory

def make_watertight_with_pcu(mesh_path):
    """Create watertight mesh using point-cloud-utils"""
    mesh = trimesh.load(mesh_path, force='mesh')
    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.dump(concatenate=True)
    if not isinstance(mesh, trimesh.Trimesh):
        raise ValueError("Input could not be converted to a single mesh")
    verts, faces = pcu.make_mesh_watertight(mesh.vertices, mesh.faces, resolution=20000)
    return verts, faces

def normalize_mesh(verts):
    """Normalize vertices to fit in unit cube centered at origin"""
    verts = verts - np.mean(verts, axis=0)
    scale = 1.0 / np.max(np.ptp(verts, axis=0))
    return verts * scale

def sample_on_surface(mesh, num_points):
    """Sample points on mesh surface"""
    return trimesh.sample.sample_surface(mesh, num_points)[0]

def sample_uniform_grid(num_points):
    """Sample points in uniform 3D grid"""
    return np.random.uniform(-1, 1, size=(num_points, 3))

def compute_signed_distance(verts, faces, points):
    """Compute signed distance for points"""
    return pcu.signed_distance_to_mesh(points, verts, faces)[0]

def save_samples(output_dir, filename, points, distances):
    """Save sampled points with distances to CSV"""
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, filename)
    np.savetxt(path, np.hstack([points, distances.reshape(-1, 1)]), delimiter=',')
    return path

def process_single_model(obj_path, surface_output_dir, grid_output_dir):
    """Process a single model and save both surface and grid samples"""
    # Skip if both outputs exist
    obj_id = os.path.basename(os.path.dirname(os.path.dirname(obj_path)))
    surface_csv = os.path.join(surface_output_dir, "sdf_data.csv")
    grid_csv = os.path.join(grid_output_dir, "grid_gt.csv")
    
    if os.path.exists(surface_csv) and os.path.exists(grid_csv):
        return "skipped"
    
    try:
        # Process mesh
        verts, faces = make_watertight_with_pcu(obj_path)
        verts = normalize_mesh(verts)
        mesh = trimesh.Trimesh(vertices=verts, faces=faces)
        
        # Generate and save surface samples
        surface_points = sample_on_surface(mesh, 7000)
        surface_sdf = 0
        save_samples(surface_output_dir, "sdf_data.csv", surface_points, surface_sdf)
        
        # Generate and save grid samples
        grid_points = sample_uniform_grid(3000)
        grid_sdf = compute_signed_distance(verts, faces, grid_points)
        save_samples(grid_output_dir, "grid_gt.csv", grid_points, grid_sdf)
        
        return "success"
    except Exception as e:
        print(f"Error processing {obj_id}: {str(e)}")
        return "failed"

def process_all_mugs(shapenet_root, acronym_output, grid_output):
    """Process all mug models from ShapeNet"""
    mug_dir = os.path.join(shapenet_root, SHAPENET_mug_CATEGORY)
    
    if not os.path.exists(mug_dir):
        raise FileNotFoundError(f"mug category directory not found at {mug_dir}")
    
    model_ids = [d for d in os.listdir(mug_dir) 
               if os.path.isdir(os.path.join(mug_dir, d))]
    
    stats = {"success": 0, "skipped": 0, "failed": 0}
    
    print(f"Processing {len(model_ids)} mug models...")
    for obj_id in tqdm(model_ids, desc="mug Models"):
        obj_path = os.path.join(mug_dir, obj_id, MODEL_FILE_PATH)
        
        # Skip if model doesn't exist
        if not os.path.exists(obj_path):
            stats["failed"] += 1
            continue
            
        # Set up output directories
        surface_dir = os.path.join(acronym_output, "mug", obj_id)
        grid_dir = os.path.join(grid_output, "acronym", "mug", obj_id)
        
        # Process the model
        result = process_single_model(obj_path, surface_dir, grid_dir)
        stats[result] += 1
    
    print("\nProcessing Results:")
    print(f"Successful: {stats['success']}")
    print(f"Skipped:    {stats['skipped']}")
    print(f"Failed:     {stats['failed']}")

if __name__ == "__main__":
    # Configure paths
    SHAPENET_ROOT = "shapenet_download/ShapeNetCore.v2"  # Root of ShapeNet dataset
    ACRONYM_OUTPUT = "data/acronym"  # For surface samples
    GRID_OUTPUT = "data/grid_data"    # For grid samples
    
    # Create output directories
    os.makedirs(os.path.join(ACRONYM_OUTPUT, "mug"), exist_ok=True)
    os.makedirs(os.path.join(GRID_OUTPUT, "acronym", "mug"), exist_ok=True)
    
    # Run processing
    process_all_mugs(SHAPENET_ROOT, ACRONYM_OUTPUT, GRID_OUTPUT)