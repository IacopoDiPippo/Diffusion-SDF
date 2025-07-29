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

def process_mesh(mesh_path, output_surface_dir, output_grid_dir, 
                num_surface_points=70000, num_grid_points=30000):
    """
    Process a single mesh with cleaner output naming
    """

    # Skip if both outputs already exist
    obj_id = os.path.splitext(os.path.basename(mesh_path))[0].replace("model_normalized_", "")
    surface_csv_path = os.path.join(output_surface_dir, "sdf_data.csv")
    grid_csv_path = os.path.join(output_grid_dir, "grid_gt.csv")
    
    if os.path.exists(surface_csv_path) and os.path.exists(grid_csv_path):
        return "skipped"
    
    # Create watertight version using pcu
    verts, faces = make_watertight_with_pcu(mesh_path)
    verts = normalize_mesh(verts)
    
    # Create output directories
    os.makedirs(output_surface_dir, exist_ok=True)
    os.makedirs(output_grid_dir, exist_ok=True)
    
    # Get base filename
    obj_id = os.path.splitext(os.path.basename(mesh_path))[0]
    
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
    
    
    np.savetxt(surface_csv_path, surface_data, delimiter=',', 
                header="x,y,z,sdf", comments="")
    np.savetxt(grid_csv_path, grid_data, delimiter=',', 
                header="x,y,z,sdf", comments="")
    
    print(f"Successfully processed {mesh_path}")
    print(f"  Surface samples: {surface_csv_path}")
    print(f"  Grid samples: {grid_csv_path}")
        
    
    return "success"

def process_all_meshes(input_dir, output_surface_base, output_grid_base):
    """
    Cleaner processing with proper output naming and warning suppression
    """
    # Suppress trimesh material warnings
    import logging
    logging.getLogger('trimesh').setLevel(logging.ERROR)
    
    obj_files = [f for f in os.listdir(input_dir) if f.endswith('.obj')]
    stats = {"success": 0, "skipped": 0, "failed": 0}
    category = "Mug"
    
    print(f"\nProcessing {len(obj_files)} meshes...")
    
    for obj_file in tqdm(obj_files, desc="Processing", unit="mesh"):
        obj_id = obj_file.replace("model_normalized_", "").replace(".obj", "")
        mesh_path = os.path.join(input_dir, obj_file)
        print(f"Mesh_path = {mesh_path}")
        output_surface_dir = os.path.join(output_surface_base, category, obj_id)
        output_grid_dir = os.path.join(output_grid_base, category, obj_id)
        
        os.makedirs(output_surface_dir, exist_ok=True)
        os.makedirs(output_grid_dir, exist_ok=True)
        
        result = process_mesh(mesh_path, output_surface_dir, output_grid_dir)
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
    
    input_dir = "all_models_renamed"
    output_surface_base = "data/acronym"
    output_grid_base = "data/grid_data/acronym"
    
    process_all_meshes(input_dir, output_surface_base, output_grid_base)