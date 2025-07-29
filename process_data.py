import os
import numpy as np
import trimesh
import point_cloud_utils as pcu
from tqdm import tqdm

def make_watertight_with_pcu(mesh_path):
    """
    Create watertight mesh using point-cloud-utils
    Handles both single meshes and scenes
    Returns watertight vertices and faces
    """
    # Load mesh and handle scene objects
    mesh = trimesh.load(mesh_path, force='mesh')  # force loading as mesh
    
    # If we got a scene, combine all meshes into one
    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.dump(concatenate=True)
    
    # Now we should have a Trimesh object
    if not isinstance(mesh, trimesh.Trimesh):
        raise ValueError("Input could not be converted to a single mesh")
    
    # Get vertices and faces
    verts = mesh.vertices
    faces = mesh.faces
    
    # Create watertight mesh - now passing both vertices and faces
    verts, faces = pcu.make_mesh_watertight(verts, faces, resolution=20000)
    return verts, faces

def sample_on_surface(mesh, num_points=100000):
    """
    Sample points directly on the mesh surface using trimesh
    """
    points, _ = trimesh.sample.sample_surface(mesh, num_points)
    return points

def sample_uniform_grid(num_points=100000, bounds=(-1, 1)):
    """
    Sample points uniformly in the 3D grid space
    """
    return np.random.uniform(low=bounds[0], high=bounds[1], 
                           size=(num_points, 3))

def compute_signed_distance(verts, faces, points):
    """
    Compute signed distance using point-cloud-utils
    """
    sdf, _, _ = pcu.signed_distance_to_mesh(points, verts, faces)
    return sdf

def normalize_mesh(verts):
    """
    Normalize vertices to fit in unit cube centered at origin
    """
    # Center the mesh
    verts = verts - np.mean(verts, axis=0)
    
    # Scale to fit in [-1, 1] cube
    scale = 1.0 / np.max(np.ptp(verts, axis=0))
    verts = verts * scale
    return verts

def process_mesh(mesh_path, output_dir, num_surface_points=70000, num_grid_points=30000):
    """
    Process a single mesh and save results in the specified output directory
    """
    # Create output directories
    os.makedirs(output_dir, exist_ok=True)
    
    # Define output paths
    surface_csv_path = os.path.join(output_dir, "sdf_data.csv")
    grid_csv_path = os.path.join(output_dir, "grid_gt.csv")
    
    if os.path.exists(surface_csv_path) and os.path.exists(grid_csv_path):
        return "skipped"
    
    try:
        # Create watertight version using pcu
        verts, faces = make_watertight_with_pcu(mesh_path)
        verts = normalize_mesh(verts)
        
        # Create trimesh object from watertight mesh
        watertight_mesh = trimesh.Trimesh(vertices=verts, faces=faces)

        # Sample points on surface and compute SDF
        surface_points = sample_on_surface(watertight_mesh, num_surface_points)
        surface_distances = compute_signed_distance(verts, faces, surface_points)
        
        # Sample uniform grid points and compute SDF
        grid_points = sample_uniform_grid(num_grid_points)
        grid_distances = compute_signed_distance(verts, faces, grid_points)
        
        # Combine points and distances
        surface_data = np.hstack([surface_points, surface_distances.reshape(-1, 1)])
        grid_data = np.hstack([grid_points, grid_distances.reshape(-1, 1)])
        
        # Save the data
        np.savetxt(surface_csv_path, surface_data, delimiter=',', comments="")
        np.savetxt(grid_csv_path, grid_data, delimiter=',', comments="")
        
        print(f"Successfully processed {mesh_path}")
        print(f"  Output saved to {output_dir}")
        
        return "success"
    except Exception as e:
        print(f"Failed to process {mesh_path}: {str(e)}")
        return "failed"

def process_shapenet_mugs(input_base_dir, output_base_dir):
    """
    Process all Mug models from ShapeNet and organize them in data/acronym/Mug structure
    """
    # Suppress trimesh material warnings
    import logging
    logging.getLogger('trimesh').setLevel(logging.ERROR)
    
    # Define ShapeNet directory structure for Mugs
    mug_category_id = "03797390"  # ShapeNet category ID for Mugs
    shapenet_mug_dir = os.path.join(input_base_dir, mug_category_id)
    
    if not os.path.exists(shapenet_mug_dir):
        raise FileNotFoundError(f"ShapeNet Mug directory not found at {shapenet_mug_dir}")
    
    # Get all Mug models
    mug_models = [d for d in os.listdir(shapenet_mug_dir) 
                 if os.path.isdir(os.path.join(shapenet_mug_dir, d))]
    
    stats = {"success": 0, "skipped": 0, "failed": 0}
    
    print(f"\nProcessing {len(mug_models)} Mug models...")
    
    for model_id in tqdm(mug_models, desc="Processing Mugs", unit="model"):
        # Define input and output paths
        input_obj_path = os.path.join(shapenet_mug_dir, model_id, "model.obj")
        output_dir = os.path.join(output_base_dir, "Mug", model_id)
        
        # Skip if the OBJ file doesn't exist
        if not os.path.exists(input_obj_path):
            print(f"Model.obj not found for {model_id}, skipping")
            stats["failed"] += 1
            continue
        
        # Process the model
        result = process_mesh(input_obj_path, output_dir)
        stats[result] += 1

    print(f"\nResults: {stats['success']} succeeded, {stats['skipped']} skipped, {stats['failed']} failed")

if __name__ == "__main__":
    # Check for pcu installation
    try:
        import point_cloud_utils
    except ImportError:
        raise ImportError(
            "point-cloud-utils required. Install with:\n"
            "pip install point-cloud-utils\n"
            "Note: On Linux you may need to install libomp-dev first"
        )
    
    # Define directories
    shapenet_base_dir = "shapenet_download"  # Base directory where ShapeNet is downloaded
    acronym_output_dir = "data/acronym"      # Output directory for Acronym dataset structure
    
    # Process all Mug models
    process_shapenet_mugs(shapenet_base_dir, acronym_output_dir)