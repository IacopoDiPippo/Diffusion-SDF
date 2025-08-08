import os
import numpy as np
import trimesh
import point_cloud_utils as pcu
from tqdm import tqdm
import json

MODEL_FILE_PATH = "models/model_normalized.obj"  # Relative path from object ID directory
TAXONOMY_FILE = "taxonomy.json"  # ShapeNet taxonomy file (maps names to IDs)

def make_watertight_with_pcu(mesh_path: str):
    mesh = trimesh.load(mesh_path, force='mesh')
    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.dump(concatenate=True)
    if not isinstance(mesh, trimesh.Trimesh):
        raise ValueError("Input could not be converted to a single mesh")
    verts, faces = pcu.make_mesh_watertight(mesh.vertices, mesh.faces, resolution=20000)
    return verts, faces

def normalize_mesh(verts: np.ndarray) -> np.ndarray:
    min_bb = np.min(verts, axis=0)
    max_bb = np.max(verts, axis=0)
    center = (min_bb + max_bb) / 2.0
    diagonal = np.linalg.norm(max_bb - min_bb)
    return (verts - center) / diagonal

def sample_on_surface(mesh: trimesh.Trimesh, num_points: int) -> np.ndarray:
    return trimesh.sample.sample_surface(mesh, num_points)[0]

def sample_uniform_grid(resolution: int = 128) -> np.ndarray:
    lin = np.linspace(-1, 1, resolution)
    grid_x, grid_y, grid_z = np.meshgrid(lin, lin, lin, indexing='ij')
    return np.stack([grid_x, grid_y, grid_z], axis=-1).reshape(-1, 3)

def compute_signed_distance(verts: np.ndarray, faces: np.ndarray, points: np.ndarray) -> np.ndarray:
    return pcu.signed_distance_to_mesh(points, verts, faces)[0]

def save_samples(output_dir: str, filename: str, points: np.ndarray, distances: np.ndarray) -> str:
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, filename)
    np.savetxt(path, np.hstack([points, distances.reshape(-1, 1)]), delimiter=',')
    return path

def scale_sdf(sdf, factor_neg=10.0, factor_pos=1.0):
    return np.where(sdf < 0, sdf * factor_neg, sdf * factor_pos)

def process_single_model(obj_path: str, surface_output_dir: str, grid_output_dir: str) -> str:
    obj_id = os.path.basename(os.path.dirname(os.path.dirname(obj_path)))
    surface_csv = os.path.join(surface_output_dir, "sdf_data.csv")
    grid_csv = os.path.join(grid_output_dir, "grid_gt.csv")

    if os.path.exists(surface_csv) and os.path.exists(grid_csv):
        return "skipped"

    try:
        verts, faces = make_watertight_with_pcu(obj_path)
        verts = normalize_mesh(verts)
        mesh = trimesh.Trimesh(vertices=verts, faces=faces)

        surface_points = sample_on_surface(mesh, 70000)
        surface_sdf = np.zeros(len(surface_points))

        noisy_005 = surface_points + np.random.normal(0, 0.005, surface_points.shape)
        noisy_0005 = surface_points + np.random.normal(0, 0.0005, surface_points.shape)

        sdf_005 = scale_sdf(compute_signed_distance(verts, faces, noisy_005))
        sdf_0005 = scale_sdf(compute_signed_distance(verts, faces, noisy_0005))

        all_points = np.vstack([surface_points, noisy_005, noisy_0005])
        all_sdf = np.concatenate([surface_sdf, sdf_005, sdf_0005])
        save_samples(surface_output_dir, "sdf_data.csv", all_points, all_sdf)

        grid_points = sample_uniform_grid(64)
        grid_sdf = scale_sdf(compute_signed_distance(verts, faces, grid_points))
        save_samples(grid_output_dir, "grid_gt.csv", grid_points, grid_sdf)

        return "success"
    except Exception as e:
        print(f"Error processing {obj_id}: {str(e)}")
        return "failed"

def get_category_id(shapenet_root: str, class_name: str) -> str:
    taxonomy_path = os.path.join(shapenet_root, TAXONOMY_FILE)
    if not os.path.exists(taxonomy_path):
        raise FileNotFoundError(f"Taxonomy file not found at {taxonomy_path}")

    with open(taxonomy_path, "r") as f:
        taxonomy = json.load(f)

    for entry in taxonomy:
        if entry["name"].lower() == class_name.lower():
            return entry["synsetId"]
    raise ValueError(f"Class name '{class_name}' not found in taxonomy")

def process_single_class(shapenet_root: str, category: str, acronym_output: str, grid_output: str):
    # category can be either a class name or a ShapeNet synset ID
    if os.path.isdir(os.path.join(shapenet_root, category)):
        cat_id = category  # It's already an ID or folder name
        class_name = category
    else:
        raise ValueError(f"'{category}' is not a valid category folder inside {shapenet_root}")

    cat_dir = os.path.join(shapenet_root, cat_id)
    model_ids = [d for d in os.listdir(cat_dir) if os.path.isdir(os.path.join(cat_dir, d))]
    stats = {"success": 0, "skipped": 0, "failed": 0}

    print(f"\nProcessing category '{class_name}' with {len(model_ids)} models...")
    for obj_id in tqdm(model_ids, desc=f"Category {class_name}"):
        obj_path = os.path.join(cat_dir, obj_id, MODEL_FILE_PATH)
        if not os.path.exists(obj_path):
            stats["failed"] += 1
            continue

        surface_dir = os.path.join(acronym_output, class_name, obj_id)
        grid_dir = os.path.join(grid_output, "acronym", class_name, obj_id)
        result = process_single_model(obj_path, surface_dir, grid_dir)
        stats[result] += 1

    print("\nProcessing Results:")
    print(f"Successful: {stats['success']}")
    print(f"Skipped:    {stats['skipped']}")
    print(f"Failed:     {stats['failed']}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Process ShapeNet class to SDF data")
    parser.add_argument("--class_name", type=str, required=True, help="Class name (e.g., mug, chair)")
    parser.add_argument("--shapenet_root", type=str, default="shapenet_download/ShapeNetCore.v2")
    parser.add_argument("--acronym_output", type=str, default="data/acronym")
    parser.add_argument("--grid_output", type=str, default="data/grid_data")
    args = parser.parse_args()

    os.makedirs(args.acronym_output, exist_ok=True)
    os.makedirs(os.path.join(args.grid_output, "acronym"), exist_ok=True)

    process_single_class(args.shapenet_root, args.class_name, args.acronym_output, args.grid_output)
