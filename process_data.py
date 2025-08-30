import os
import json
import numpy as np
import trimesh
import point_cloud_utils as pcu
from tqdm import tqdm
from typing import Optional, Tuple

# ============================================================
# Constants
# ============================================================
SHAPENET_MUG_CATEGORY = "03797390"  # Official ShapeNet category ID for mugs
MODEL_FILE_PATH = "models/model_normalized.obj"  # Relative path from object ID directory

# ---------------- Random Augmentation config (uniform ranges) ----------------
# The mesh (after normalization) will be uniformly scaled so its AABB fits inside [-h, h]^3
# h is sampled uniformly in [H_RANGE[0], H_RANGE[1]] at each variant.
H_RANGE: Tuple[float, float] = (0.6, 0.9)

# Z-rotation in degrees sampled uniformly in [Z_ROT_RANGE_DEG[0], Z_ROT_RANGE_DEG[1])
Z_ROT_RANGE_DEG: Tuple[float, float] = (0.0, 360.0)

# Per-axis translations sampled uniformly in the given ranges.
# Keep these modest so geometry tends to remain in [-1,1]^3 after scale+rotation+translation.
TRANS_RANGE_X: Tuple[float, float] = (-0.15, 0.15)
TRANS_RANGE_Y: Tuple[float, float] = (-0.15, 0.15)
TRANS_RANGE_Z: Tuple[float, float] = (-0.05, 0.05)

# How many random augmentation variants to generate (in addition to the base)
AUG_NUM_VARIANTS = 30

# Random seed for reproducibility
AUG_RANDOM_SEED = 42

# Sampling config
NUM_SURFACE_POINTS = 70000
GRID_RESOLUTION = 64
GAUSS_NOISE_STD_BIG = 0.005
GAUSS_NOISE_STD_SMALL = 0.0005

# SDF scaling (inside vs outside)
SDF_SCALE_NEG = 10.0
SDF_SCALE_POS = 1.0
# ---------------------------------------------------------------------------


# ============================================================
# Geometry utilities
# ============================================================
def make_watertight_with_pcu(mesh_path: str):
    """Create watertight mesh using point-cloud-utils."""
    mesh = trimesh.load(mesh_path, force='mesh')
    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.dump(concatenate=True)
    if not isinstance(mesh, trimesh.Trimesh):
        raise ValueError("Input could not be converted to a single mesh")
    verts, faces = pcu.make_mesh_watertight(mesh.vertices, mesh.faces, resolution=20000)
    return verts, faces


def normalize_mesh(verts: np.ndarray) -> np.ndarray:
    """
    Center and normalize mesh so that the diagonal of its axis-aligned bounding box equals 1.
    - Translation: center of AABB -> origin.
    - Scaling: divide by AABB diagonal length.
    """
    min_bb = np.min(verts, axis=0)
    max_bb = np.max(verts, axis=0)
    center = (min_bb + max_bb) / 2.0
    diagonal = np.linalg.norm(max_bb - min_bb)
    if diagonal <= 0:
        raise ValueError("Degenerate mesh: zero AABB diagonal")
    return (verts - center) / diagonal


def y_up_to_z_up_swap(verts: np.ndarray) -> np.ndarray:
    """
    Convert Y-up coordinates to Z-up by swapping Y and Z axes.
    Mapping: (x, y, z) -> (x, z, y)
    NOTE: This preserves handedness if the source is right-handed (common for ShapeNet .obj).
    If you need Z to point in the opposite direction, flip the sign after swap: verts[:, 2] *= -1
    """
    v = verts.copy()
    v[:, [1, 2]] = v[:, [2, 1]]
    return v


def sample_on_surface(mesh: trimesh.Trimesh, num_points: int) -> np.ndarray:
    """Sample points on mesh surface (returns only the points)."""
    return trimesh.sample.sample_surface(mesh, num_points)[0]


def sample_uniform_grid(resolution: int = 128) -> np.ndarray:
    """Generate a uniform 3D grid of points in [-1, 1]^3 with shape (resolution^3, 3)."""
    lin = np.linspace(-1, 1, resolution)
    grid_x, grid_y, grid_z = np.meshgrid(lin, lin, lin, indexing='ij')
    grid_points = np.stack([grid_x, grid_y, grid_z], axis=-1).reshape(-1, 3)
    return grid_points


def compute_signed_distance(verts: np.ndarray, faces: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Compute signed distance for query points to the given mesh (verts, faces)."""
    return pcu.signed_distance_to_mesh(points, verts, faces)[0]


def save_samples(output_dir: str, filename: str, points: np.ndarray, distances: np.ndarray) -> str:
    """Save sampled points with distances to CSV (columns: x,y,z,sdf)."""
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, filename)
    np.savetxt(path, np.hstack([points, distances.reshape(-1, 1)]), delimiter=',')
    return path


def scale_sdf(sdf, factor_neg: float = SDF_SCALE_NEG, factor_pos: float = SDF_SCALE_POS):
    """Apply asymmetric scaling: inside distances (negative) get amplified."""
    sdf_scaled = np.where(sdf < 0, sdf * factor_neg, sdf * factor_pos)
    return sdf_scaled


# ============================================================
# Augmentation utilities (RANDOM)
# ============================================================
def aabb_half_extent(verts: np.ndarray) -> float:
    """
    Compute the maximum half-extent along any axis of the mesh AABB.
    i.e., max( (max_x - min_x)/2, (max_y - min_y)/2, (max_z - min_z)/2 )
    """
    min_bb = verts.min(axis=0)
    max_bb = verts.max(axis=0)
    half_extents = 0.5 * (max_bb - min_bb)
    return float(np.max(half_extents))


def rotation_matrix_z(theta_rad: float) -> np.ndarray:
    """Return a 3x3 rotation matrix for rotation around +Z by theta (radians)."""
    c, s = np.cos(theta_rad), np.sin(theta_rad)
    return np.array([[ c, -s, 0.0],
                     [ s,  c, 0.0],
                     [0.0, 0.0, 1.0]], dtype=np.float64)


def transform_verts(verts: np.ndarray, R: np.ndarray, t: np.ndarray, scale: float = 1.0) -> np.ndarray:
    """Apply uniform scale, then rotation R (3x3), then translation t (3,) to verts."""
    return (verts * scale) @ R.T + t


def augment_mesh_random_variants(
    verts: np.ndarray,
    faces: np.ndarray,
    h_range: Tuple[float, float],
    z_rot_range_deg: Tuple[float, float],
    tx_range: Tuple[float, float],
    ty_range: Tuple[float, float],
    tz_range: Tuple[float, float],
    num_variants: int,
    seed: Optional[int] = None
):
    """
    Yield 'num_variants' random augmented (verts, faces) variants by sampling uniformly:
      - h ~ U(h_range[0], h_range[1])  -> fit AABB inside [-h, h]^3
      - z_deg ~ U(z_rot_range_deg[0], z_rot_range_deg[1])  -> rotation around +Z
      - t ~ (U(tx_range), U(ty_range), U(tz_range))  -> translation
    """
    rng = np.random.default_rng(seed)
    base_half_extent = aabb_half_extent(verts)
    if base_half_extent <= 0:
        raise ValueError("Degenerate mesh: zero base half-extent for augmentation")

    h_low, h_high = h_range
    z_low, z_high = z_rot_range_deg
    tx_low, tx_high = tx_range
    ty_low, ty_high = ty_range
    tz_low, tz_high = tz_range

    for _ in range(num_variants):
        h = rng.uniform(h_low, h_high)
        z_deg = rng.uniform(z_low, z_high)
        Rz = rotation_matrix_z(np.deg2rad(z_deg))
        tx = rng.uniform(tx_low, tx_high)
        ty = rng.uniform(ty_low, ty_high)
        tz = rng.uniform(tz_low, tz_high)
        t_vec = np.array([tx, ty, tz], dtype=np.float64)

        scale_to_h = (h / base_half_extent)
        aug_verts = transform_verts(verts, Rz, t_vec, scale=scale_to_h)

        yield {
            "verts": aug_verts,
            "faces": faces,
            "meta": {
                "target_half_extent": float(h),
                "z_rot_deg": float(z_deg),
                "translation": (float(tx), float(ty), float(tz)),
                "scale_applied": float(scale_to_h),
            }
        }


# ============================================================
# IO helpers for directory naming
# ============================================================
def variant_folder_name(obj_id: str, rot_deg: float, scale_applied: float, tx: float, ty: float, tz: float) -> str:
    """
    Build a folder name like:
      <obj_id>_rot{deg}_s{scale}_tx{tx}_ty{ty}_tz{tz}
    with fixed number formatting for reproducibility and stable paths.
    """
    return (
        f"{obj_id}"
        f"_rot{rot_deg:.1f}"
        f"_s{scale_applied:.3f}"
        f"_tx{tx:+.3f}"
        f"_ty{ty:+.3f}"
        f"_tz{tz:+.3f}"
    )


def write_meta_json(out_dir: str, meta: dict):
    """Write a small JSON file with augmentation parameters for traceability."""
    os.makedirs(out_dir, exist_ok=True)
    meta_path = os.path.join(out_dir, "meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)


# ============================================================
# Processing logic
# ============================================================
def process_single_model(obj_path: str, surface_base_dir: str, grid_base_dir: str) -> str:
    """
    Process a single model and save both surface and grid samples for:
      - the normalized + Y->Z swapped base mesh -> saved with rot0/s1/tx0/ty0/tz0
      - AUG_NUM_VARIANTS random augmentation variants

    Output directory pattern:
      surface_base_dir / "<obj_id>_rot{deg}_s{scale}_tx{tx}_ty{ty}_tz{tz}" / "sdf_data.csv"
      grid_base_dir    / "<obj_id>_rot{deg}_s{scale}_tx{tx}_ty{ty}_tz{tz}" / "grid_gt.csv"
    """
    obj_id = os.path.basename(os.path.dirname(os.path.dirname(obj_path)))

    # Base folder (no augmentation)
    base_folder = variant_folder_name(obj_id, rot_deg=0.0, scale_applied=1.000, tx=0.0, ty=0.0, tz=0.0)
    base_surface_csv = os.path.join(surface_base_dir, base_folder, "sdf_data.csv")
    base_grid_csv = os.path.join(grid_base_dir, base_folder, "grid_gt.csv")
    if os.path.exists(base_surface_csv) and os.path.exists(base_grid_csv):
        return "skipped"

    try:
        # 1) Watertight then normalize (diag(AABB)=1)
        raw_verts, faces = make_watertight_with_pcu(obj_path)
        base_verts = normalize_mesh(raw_verts)

        # 2) Enforce Z-up by swapping Y and Z (since ShapeNet mugs are Y-up)
        base_verts = y_up_to_z_up_swap(base_verts)

        # Helper to process and save one (verts, faces) into a named folder
        def _process_and_save_named(v: np.ndarray, folder_name: str, meta: Optional[dict] = None):
            mesh = trimesh.Trimesh(vertices=v, faces=faces, process=False)

            # Surface sampling (clean + noisy)
            surface_points = sample_on_surface(mesh, NUM_SURFACE_POINTS)
            surface_sdf = np.zeros(len(surface_points))

            noisy_big = surface_points + np.random.normal(0, GAUSS_NOISE_STD_BIG, surface_points.shape)
            noisy_small = surface_points + np.random.normal(0, GAUSS_NOISE_STD_SMALL, surface_points.shape)

            sdf_big = scale_sdf(compute_signed_distance(v, faces, noisy_big))
            sdf_small = scale_sdf(compute_signed_distance(v, faces, noisy_small))

            all_points = np.vstack([surface_points, noisy_big, noisy_small])
            all_sdf = np.concatenate([surface_sdf, sdf_big, sdf_small])

            # Grid sampling
            grid_points = sample_uniform_grid(GRID_RESOLUTION)
            grid_sdf = scale_sdf(compute_signed_distance(v, faces, grid_points))

            # Output dirs for this variant
            surf_dir = os.path.join(surface_base_dir, folder_name)
            grid_dir = os.path.join(grid_base_dir, folder_name)

            save_samples(surf_dir, "sdf_data.csv", all_points, all_sdf)
            save_samples(grid_dir, "grid_gt.csv", grid_points, grid_sdf)

            if meta is not None:
                write_meta_json(surf_dir, meta)

        # 3) Base (non-augmented) variant
        _process_and_save_named(
            base_verts,
            base_folder,
            meta={
                "variant": "base_normalized",
                "note": "AABB diagonal normalized to 1; Y-up -> Z-up by swapping Y and Z.",
                "z_rot_deg": 0.0,
                "scale_applied": 1.0,
                "translation": (0.0, 0.0, 0.0),
                "aabb_half_extent_after_norm": aabb_half_extent(base_verts),
            },
        )

        # 4) Random augmented variants
        for variant in augment_mesh_random_variants(
            base_verts,
            faces,
            h_range=H_RANGE,
            z_rot_range_deg=Z_ROT_RANGE_DEG,
            tx_range=TRANS_RANGE_X,
            ty_range=TRANS_RANGE_Y,
            tz_range=TRANS_RANGE_Z,
            num_variants=AUG_NUM_VARIANTS,
            seed=AUG_RANDOM_SEED
        ):
            zdeg = variant["meta"]["z_rot_deg"]
            s = variant["meta"]["scale_applied"]
            tx, ty, tz = variant["meta"]["translation"]
            folder = variant_folder_name(obj_id, rot_deg=zdeg, scale_applied=s, tx=tx, ty=ty, tz=tz)

            _process_and_save_named(
                variant["verts"],
                folder,
                meta=variant["meta"]
            )

        return "success"

    except Exception as e:
        print(f"Error processing {obj_id}: {str(e)}")
        return "failed"


def process_all_mugs(shapenet_root: str, acronym_output: str, grid_output: str):
    """Process all mug models from ShapeNet with random augmentation and path encoding."""
    mug_dir = os.path.join(shapenet_root, SHAPENET_MUG_CATEGORY)

    if not os.path.exists(mug_dir):
        raise FileNotFoundError(f"mug category directory not found at {mug_dir}")

    model_ids = [d for d in os.listdir(mug_dir) if os.path.isdir(os.path.join(mug_dir, d))]
    stats = {"success": 0, "skipped": 0, "failed": 0}

    print(f"Processing {len(model_ids)} mug models...")
    for obj_id in tqdm(model_ids, desc="Mug Models"):
        obj_path = os.path.join(mug_dir, obj_id, MODEL_FILE_PATH)

        if not os.path.exists(obj_path):
            stats["failed"] += 1
            continue

        # base dirs (per-variant folders are named, not nested by obj_id)
        surface_base_dir = os.path.join(acronym_output, "mug")
        grid_base_dir = os.path.join(grid_output, "acronym", "mug")

        result = process_single_model(obj_path, surface_base_dir, grid_base_dir)
        stats[result] += 1

    print("\nProcessing Results:")
    print(f"Successful: {stats['success']}")
    print(f"Skipped:    {stats['skipped']}")
    print(f"Failed:     {stats['failed']}")


# ============================================================
# Entrypoint
# ============================================================
if __name__ == "__main__":
    SHAPENET_ROOT = "shapenet_download/ShapeNetCore.v2"
    ACRONYM_OUTPUT = "data/acronym"
    GRID_OUTPUT = "data/grid_data"

    # Create the base directories (per-variant folders are created on demand)
    os.makedirs(os.path.join(ACRONYM_OUTPUT, "mug"), exist_ok=True)
    os.makedirs(os.path.join(GRID_OUTPUT, "acronym", "mug"), exist_ok=True)

    process_all_mugs(SHAPENET_ROOT, ACRONYM_OUTPUT, GRID_OUTPUT)
