import os
import numpy as np
import trimesh
import point_cloud_utils as pcu
from tqdm import tqdm

# Constants
SHAPENET_couch_CATEGORY = "04256520"  # Official ShapeNet category ID for couches
MODEL_FILE_PATH = "models/model_normalized.obj"  # Relative path from object ID directory

def make_watertight_with_pcu(mesh_path: str):
    """Create watertight mesh using point-cloud-utils"""
    mesh = trimesh.load(mesh_path, force='mesh')
    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.dump(concatenate=True)
    if not isinstance(mesh, trimesh.Trimesh):
        raise ValueError("Input could not be converted to a single mesh")
    verts, faces = pcu.make_mesh_watertight(mesh.vertices, mesh.faces, resolution=20000)
    return verts, faces

def normalize_mesh(verts: np.ndarray) -> np.ndarray:
    """Center and normalize mesh such that diagonal of bounding box = 1"""
    min_bb = np.min(verts, axis=0)
    max_bb = np.max(verts, axis=0)
    center = (min_bb + max_bb) / 2.0
    diagonal = np.linalg.norm(max_bb - min_bb)
    return (verts - center) / diagonal

def sample_on_surface(mesh: trimesh.Trimesh, num_points: int) -> np.ndarray:
    """Sample points on mesh surface"""
    return trimesh.sample.sample_surface(mesh, num_points)[0]

def sample_uniform_grid(resolution: int = 128) -> np.ndarray:
    """Generate uniform 3D grid of points in [-1, 1]^3"""
    lin = np.linspace(-1, 1, resolution)
    grid_x, grid_y, grid_z = np.meshgrid(lin, lin, lin, indexing='ij')
    grid_points = np.stack([grid_x, grid_y, grid_z], axis=-1).reshape(-1, 3)
    return grid_points

def compute_signed_distance(verts: np.ndarray, faces: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Compute signed distance for points"""
    return pcu.signed_distance_to_mesh(points, verts, faces)[0]

def save_samples(output_dir: str, filename: str, points: np.ndarray, distances: np.ndarray) -> str:
    """Save sampled points with distances to CSV"""
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, filename)
    np.savetxt(path, np.hstack([points, distances.reshape(-1, 1)]), delimiter=',')
    return path

def scale_sdf(sdf, factor_neg=10.0, factor_pos=1.0):
    sdf_scaled = np.where(sdf < 0, sdf * factor_neg, sdf * factor_pos)
    return sdf_scaled

def process_single_model(obj_path: str, surface_output_dir: str, grid_output_dir: str) -> str:
    """Process a single model and save both surface and grid samples"""
    obj_id = os.path.basename(os.path.dirname(os.path.dirname(obj_path)))
    surface_csv = os.path.join(surface_output_dir, "sdf_data.csv")
    grid_csv = os.path.join(grid_output_dir, "grid_gt.csv")
    
    if os.path.exists(surface_csv) and os.path.exists(grid_csv):
        return "skipped"
    
    try:
        # Watertight and normalized
        verts, faces = make_watertight_with_pcu(obj_path)
        verts = normalize_mesh(verts)
        mesh = trimesh.Trimesh(vertices=verts, faces=faces)

        # Surface sampling
        surface_points = sample_on_surface(mesh, 70000)
        surface_sdf = np.zeros(len(surface_points))

        # Gaussian perturbations
        noisy_005 = surface_points + np.random.normal(0, 0.005, surface_points.shape)
        noisy_0005 = surface_points + np.random.normal(0, 0.0005, surface_points.shape)

        sdf_005 = scale_sdf(compute_signed_distance(verts, faces, noisy_005))
        sdf_0005 = scale_sdf(compute_signed_distance(verts, faces, noisy_0005))

        all_points = np.vstack([surface_points, noisy_005, noisy_0005])
        all_sdf = np.concatenate([surface_sdf, sdf_005, sdf_0005])

        save_samples(surface_output_dir, "sdf_data.csv", all_points, all_sdf)

        # Structured grid sampling
        grid_points = sample_uniform_grid(64)
        grid_sdf = scale_sdf(compute_signed_distance(verts, faces, grid_points))
        save_samples(grid_output_dir, "grid_gt.csv", grid_points, grid_sdf)

        return "success"
    except Exception as e:
        print(f"Error processing {obj_id}: {str(e)}")
        return "failed"

def process_all_couches(shapenet_root: str, acronym_output: str, grid_output: str):
    """Process all couch models from ShapeNet"""
    couch_dir = os.path.join(shapenet_root, SHAPENET_couch_CATEGORY)
    
    if not os.path.exists(couch_dir):
        raise FileNotFoundError(f"couch category directory not found at {couch_dir}")
    
    model_ids = [d for d in os.listdir(couch_dir) if os.path.isdir(os.path.join(couch_dir, d))]
    stats = {"success": 0, "skipped": 0, "failed": 0}
    
    print(f"Processing {len(model_ids)} couch models...")
    for obj_id in tqdm(model_ids, desc="couch Models"):
        obj_path = os.path.join(couch_dir, obj_id, MODEL_FILE_PATH)
        
        if not os.path.exists(obj_path):
            stats["failed"] += 1
            continue
        
        surface_dir = os.path.join(acronym_output, "couch", obj_id)
        grid_dir = os.path.join(grid_output, "acronym", "couch", obj_id)
        
        result = process_single_model(obj_path, surface_dir, grid_dir)
        stats[result] += 1
    
    print("\nProcessing Results:")
    print(f"Successful: {stats['success']}")
    print(f"Skipped:    {stats['skipped']}")
    print(f"Failed:     {stats['failed']}")

if __name__ == "__main__":
    SHAPENET_ROOT = "shapenet_download/ShapeNetCore.v2"
    ACRONYM_OUTPUT = "data/acronym"
    GRID_OUTPUT = "data/grid_data"

    os.makedirs(os.path.join(ACRONYM_OUTPUT, "couch"), exist_ok=True)
    os.makedirs(os.path.join(GRID_OUTPUT, "acronym", "couch"), exist_ok=True)

    process_all_couches(SHAPENET_ROOT, ACRONYM_OUTPUT, GRID_OUTPUT)
