import os
import json
import numpy as np
import trimesh
import point_cloud_utils as pcu
from tqdm import tqdm
from typing import Optional, Tuple, List

# ============================================================
# Constants
# ============================================================
SHAPENET_MUG_CATEGORY = "03797390"  # Official ShapeNet category ID for mugs
MODEL_FILE_PATH = "models/model_normalized.obj"  # Relative path from object ID directory

# ---------------- Augmentation config ----------------
# Il mesh (dopo normalizzazione) viene scalato così che l'AABB stia in [-h,h]^3.
# Per le 10 varianti "translate-only" uso h fisso (media del range) per isolare l'effetto traslazione.
H_RANGE: Tuple[float, float] = (0.6, 0.9)
H_FIXED_FOR_TRANSL = float(np.mean(H_RANGE))  # ### MOD: h fisso per le 10 traslazioni

# Z-rotation in degrees (usata per variare orientamento)
Z_ROT_RANGE_DEG: Tuple[float, float] = (0.0, 360.0)

# Ranges di traslazione (restano uguali)
TRANS_RANGE_X: Tuple[float, float] = (-0.15, 0.15)
TRANS_RANGE_Y: Tuple[float, float] = (-0.15, 0.15)
TRANS_RANGE_Z: Tuple[float, float] = (-0.05, 0.05)

# ### MOD: Numero varianti deterministiche = 16 (10 translate-max + 6 center-scale)
AUG_NUM_VARIANTS = 16

# Random seed per eventuali rotazioni
AUG_RANDOM_SEED = 42

# Sampling config (uguale)
NUM_SURFACE_POINTS = 70000
GRID_RESOLUTION = 64
GAUSS_NOISE_STD_BIG = 0.005
GAUSS_NOISE_STD_SMALL = 0.0005

# SDF scaling (inside vs outside) (uguale)
SDF_SCALE_NEG = 10.0
SDF_SCALE_POS = 1.0
# -----------------------------------------------------


# ============================================================
# Geometry utilities
# ============================================================
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
    if diagonal <= 0:
        raise ValueError("Degenerate mesh: zero AABB diagonal")
    return (verts - center) / diagonal


def y_up_to_z_up_swap(verts: np.ndarray) -> np.ndarray:
    v = verts.copy()
    v[:, [1, 2]] = v[:, [2, 1]]
    return v


def sample_on_surface(mesh: trimesh.Trimesh, num_points: int) -> np.ndarray:
    return trimesh.sample.sample_surface(mesh, num_points)[0]


def sample_uniform_grid(resolution: int = 128) -> np.ndarray:
    lin = np.linspace(-1, 1, resolution)
    grid_x, grid_y, grid_z = np.meshgrid(lin, lin, lin, indexing='ij')
    grid_points = np.stack([grid_x, grid_y, grid_z], axis=-1).reshape(-1, 3)
    return grid_points


def compute_signed_distance(verts: np.ndarray, faces: np.ndarray, points: np.ndarray) -> np.ndarray:
    return pcu.signed_distance_to_mesh(points, verts, faces)[0]


def save_samples(output_dir: str, filename: str, points: np.ndarray, distances: np.ndarray) -> str:
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, filename)
    np.savetxt(path, np.hstack([points, distances.reshape(-1, 1)]), delimiter=',')
    return path


def scale_sdf(sdf, factor_neg: float = SDF_SCALE_NEG, factor_pos: float = SDF_SCALE_POS):
    return np.where(sdf < 0, sdf * factor_neg, sdf * factor_pos)


# ============================================================
# Augmentation utilities
# ============================================================
def aabb_half_extent(verts: np.ndarray) -> float:
    min_bb = verts.min(axis=0)
    max_bb = verts.max(axis=0)
    half_extents = 0.5 * (max_bb - min_bb)
    return float(np.max(half_extents))


def rotation_matrix_z(theta_rad: float) -> np.ndarray:
    c, s = np.cos(theta_rad), np.sin(theta_rad)
    return np.array([[ c, -s, 0.0],
                     [ s,  c, 0.0],
                     [0.0, 0.0, 1.0]], dtype=np.float64)


def transform_verts(verts: np.ndarray, R: np.ndarray, t: np.ndarray, scale: float = 1.0) -> np.ndarray:
    return (verts * scale) @ R.T + t


# ---------- NEW: generator deterministico per 10 traslazioni massime ----------
def generate_max_translation_variants(
    verts: np.ndarray,
    faces: np.ndarray,
    tx_range: Tuple[float, float],
    ty_range: Tuple[float, float],
    tz_range: Tuple[float, float],
    h_fixed: float,
    z_rot_range_deg: Tuple[float, float],
    seed: Optional[int] = None
):
    """
    10 varianti con traslazioni alla massima distanza dal centro
    entro i range consentiti:
      - 8 agli "angoli" del parallelepipedo di traslazione (±x_max, ±y_max, ±z_max)
      - 2 extra con stessa distanza massima ma rotazioni diverse (riuso di 2 angoli)
    La scala è fissa (h_fixed) per isolare l'effetto traslazione.
    """
    rng = np.random.default_rng(seed)

    # estremi
    tx_min, tx_max = tx_range
    ty_min, ty_max = ty_range
    tz_min, tz_max = tz_range

    # 8 angoli (±, ±, ±)
    corners = [
        (tx, ty, tz)
        for tx in (tx_min, tx_max)
        for ty in (ty_min, ty_max)
        for tz in (tz_min, tz_max)
    ]

    # pick 2 corners (casuali) per arrivare a 10 (distanza uguale, rotazione diversa)
    extra_idxs = rng.choice(len(corners), size=2, replace=False).tolist()
    translations: List[Tuple[float, float, float]] = corners + [corners[i] for i in extra_idxs]

    # rotazioni equidistanti
    z_angles = np.linspace(z_rot_range_deg[0], z_rot_range_deg[1], num=len(translations), endpoint=False)

    base_half_extent = aabb_half_extent(verts)
    if base_half_extent <= 0:
        raise ValueError("Degenerate mesh: zero base half-extent for augmentation")
    scale_to_h = (h_fixed / base_half_extent)

    for (tx, ty, tz), zdeg in zip(translations, z_angles):
        Rz = rotation_matrix_z(np.deg2rad(zdeg))
        t_vec = np.array([tx, ty, tz], dtype=np.float64)
        aug_verts = transform_verts(verts, Rz, t_vec, scale=scale_to_h)
        yield {
            "verts": aug_verts,
            "faces": faces,
            "meta": {
                "variant": "translate_max",
                "target_half_extent": float(h_fixed),
                "z_rot_deg": float(zdeg),
                "translation": (float(tx), float(ty), float(tz)),
                "scale_applied": float(scale_to_h),
                "note": "Traslazione alla distanza massima consentita dai range; scala fissata per isolare la traslazione."
            }
        }


# ---------- NEW: generator deterministico per 6 scale al centro ----------
def generate_center_scale_variants(
    verts: np.ndarray,
    faces: np.ndarray,
    h_range: Tuple[float, float],
    num_scales: int = 6
):
    """
    6 varianti con mug al centro (t=0), rotazione z=0, scale diverse.
    Le scale sono ottenute campionando h in linspace sull'intervallo dato.
    """
    base_half_extent = aabb_half_extent(verts)
    if base_half_extent <= 0:
        raise ValueError("Degenerate mesh: zero base half-extent for augmentation")

    h_vals = np.linspace(h_range[0], h_range[1], num_scales)
    for h in h_vals:
        scale_to_h = (h / base_half_extent)
        Rz = rotation_matrix_z(0.0)
        t_vec = np.array([0.0, 0.0, 0.0], dtype=np.float64)
        aug_verts = transform_verts(verts, Rz, t_vec, scale=scale_to_h)
        yield {
            "verts": aug_verts,
            "faces": faces,
            "meta": {
                "variant": "center_scale",
                "target_half_extent": float(h),
                "z_rot_deg": 0.0,
                "translation": (0.0, 0.0, 0.0),
                "scale_applied": float(scale_to_h),
                "note": "Mug al centro, scale differenti (linspace su H_RANGE)."
            }
        }


# ============================================================
# IO helpers for directory naming
# ============================================================
def variant_folder_name(obj_id: str, rot_deg: float, scale_applied: float, tx: float, ty: float, tz: float) -> str:
    return (
        f"{obj_id}"
        f"_rot{rot_deg:.1f}"
        f"_s{scale_applied:.3f}"
        f"_tx{tx:+.3f}"
        f"_ty{ty:+.3f}"
        f"_tz{tz:+.3f}"
    )


def write_meta_json(out_dir: str, meta: dict):
    os.makedirs(out_dir, exist_ok=True)
    meta_path = os.path.join(out_dir, "meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)


# ============================================================
# Processing logic
# ============================================================
def process_single_model(obj_path: str, surface_base_dir: str, grid_base_dir: str) -> str:
    """
    Per un singolo modello:
      - base normalizzata (come prima)
      - 10 varianti con traslazione massima (scala fissa)
      - 6 varianti al centro con scale diverse
    """
    obj_id = os.path.basename(os.path.dirname(os.path.dirname(obj_path)))

    # Base folder (no augmentation)
    base_folder = variant_folder_name(obj_id, rot_deg=0.0, scale_applied=1.000, tx=0.0, ty=0.0, tz=0.0)
    base_surface_csv = os.path.join(surface_base_dir, base_folder, "sdf_data.csv")
    base_grid_csv = os.path.join(grid_base_dir, base_folder, "grid_gt.csv")
    """if os.path.exists(base_surface_csv) and os.path.exists(base_grid_csv):
        return "skipped""""

    try:
        # 1) Watertight + normalize
        raw_verts, faces = make_watertight_with_pcu(obj_path)
        base_verts = normalize_mesh(raw_verts)

        # 2) Y-up -> Z-up
        base_verts = y_up_to_z_up_swap(base_verts)

        # helper di salvataggio (uguale)
        def _process_and_save_named(v: np.ndarray, folder_name: str, meta: Optional[dict] = None):
            mesh = trimesh.Trimesh(vertices=v, faces=faces, process=False)

            surface_points = sample_on_surface(mesh, NUM_SURFACE_POINTS)
            surface_sdf = np.zeros(len(surface_points))

            noisy_big = surface_points + np.random.normal(0, GAUSS_NOISE_STD_BIG, surface_points.shape)
            noisy_small = surface_points + np.random.normal(0, GAUSS_NOISE_STD_SMALL, surface_points.shape)

            sdf_big = scale_sdf(compute_signed_distance(v, faces, noisy_big))
            sdf_small = scale_sdf(compute_signed_distance(v, faces, noisy_small))

            all_points = np.vstack([surface_points, noisy_big, noisy_small])
            all_sdf = np.concatenate([surface_sdf, sdf_big, sdf_small])

            grid_points = sample_uniform_grid(GRID_RESOLUTION)
            grid_sdf = scale_sdf(compute_signed_distance(v, faces, grid_points))

            surf_dir = os.path.join(surface_base_dir, folder_name)
            grid_dir = os.path.join(grid_base_dir, folder_name)

            save_samples(surf_dir, "sdf_data.csv", all_points, all_sdf)
            save_samples(grid_dir, "grid_gt.csv", grid_points, grid_sdf)

            if meta is not None:
                write_meta_json(surf_dir, meta)

        # 3) Base (non-augmented)
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

        # 4) 10 varianti: traslazioni massime
        for variant in generate_max_translation_variants(
            base_verts,
            faces,
            tx_range=TRANS_RANGE_X,
            ty_range=TRANS_RANGE_Y,
            tz_range=TRANS_RANGE_Z,
            h_fixed=H_FIXED_FOR_TRANSL,
            z_rot_range_deg=Z_ROT_RANGE_DEG,
            seed=AUG_RANDOM_SEED
        ):
            zdeg = variant["meta"]["z_rot_deg"]
            s = variant["meta"]["scale_applied"]
            tx, ty, tz = variant["meta"]["translation"]
            folder = variant_folder_name(obj_id, rot_deg=zdeg, scale_applied=s, tx=tx, ty=ty, tz=tz)
            _process_and_save_named(variant["verts"], folder, meta=variant["meta"])

        # 5) 6 varianti: centro + scale diverse
        for variant in generate_center_scale_variants(
            base_verts,
            faces,
            h_range=H_RANGE,
            num_scales=6
        ):
            zdeg = variant["meta"]["z_rot_deg"]
            s = variant["meta"]["scale_applied"]
            tx, ty, tz = variant["meta"]["translation"]
            folder = variant_folder_name(obj_id, rot_deg=zdeg, scale_applied=s, tx=tx, ty=ty, tz=tz)
            _process_and_save_named(variant["verts"], folder, meta=variant["meta"])

        return "success"

    except Exception as e:
        print(f"Error processing {obj_id}: {str(e)}")
        return "failed"


# ============================================================
# Selezione UNA sola mug
# ============================================================
def find_single_mug_path(shapenet_root: str, target_obj_id: Optional[str] = None) -> Optional[str]:
    """
    Restituisce il path dell'unica mug da processare:
      - se target_obj_id è dato e valido, usa quello
      - altrimenti prende il primo modello valido trovato
    """
    mug_dir = os.path.join(shapenet_root, SHAPENET_MUG_CATEGORY)
    if not os.path.exists(mug_dir):
        raise FileNotFoundError(f"mug category directory not found at {mug_dir}")

    if target_obj_id:
        obj_path = os.path.join(mug_dir, target_obj_id, MODEL_FILE_PATH)
        return obj_path if os.path.exists(obj_path) else None

    # altrimenti il primo disponibile
    for obj_id in os.listdir(mug_dir):
        obj_path = os.path.join(mug_dir, obj_id, MODEL_FILE_PATH)
        if os.path.exists(obj_path):
            return obj_path
    return None


def process_one_mug(shapenet_root: str, acronym_output: str, grid_output: str, target_obj_id: Optional[str] = None):
    """Processa una sola mug secondo le nuove regole (16 augmentation deterministiche)."""
    obj_path = find_single_mug_path(shapenet_root, target_obj_id=target_obj_id)
    if obj_path is None:
        raise FileNotFoundError("Nessuna mug valida trovata (o ID specificato non valido).")

    surface_base_dir = os.path.join(acronym_output, "mug")
    grid_base_dir = os.path.join(grid_output, "acronym", "mug")
    os.makedirs(surface_base_dir, exist_ok=True)
    os.makedirs(grid_base_dir, exist_ok=True)

    res = process_single_model(obj_path, surface_base_dir, grid_base_dir)
    print(f"Processing result: {res}")


# ============================================================
# Entrypoint
# ============================================================
if __name__ == "__main__":
    SHAPENET_ROOT = "shapenet_download/ShapeNetCore.v2"
    ACRONYM_OUTPUT = "data/acronym"
    GRID_OUTPUT = "data/grid_data"

    # ### MOD: opzionale, specifica qui un ID di mug ShapeNet se vuoi una mug precisa
    TARGET_OBJ_ID = None  # es. "1a2b3c4d5e..." oppure lascia None per la prima trovata

    process_one_mug(SHAPENET_ROOT, ACRONYM_OUTPUT, GRID_OUTPUT, target_obj_id=TARGET_OBJ_ID)

