import os
import time
import json
import csv
import hashlib
import logging
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.utils.data
from tqdm import tqdm
from bps import bps

from . import base


def _hash_path(path: str) -> str:
    """Hash robusto: path + size + mtime (invalidazione cache se cambia)."""
    try:
        st = os.stat(path)
        meta = f"{path}|{st.st_size}|{int(st.st_mtime)}"
    except OSError:
        meta = path
    return hashlib.sha1(meta.encode("utf-8")).hexdigest()


def _ensure_contig_f32(x: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(x.astype(np.float32, copy=False))


def _clean_numeric_ndarray(arr: np.ndarray) -> np.ndarray:
    """Rimuove righe con NaN/Inf; forza float32 contiguo."""
    if arr.ndim != 2:
        arr = arr.reshape(arr.shape[0], -1)
    # keep only finite rows
    mask = np.isfinite(arr).all(axis=1)
    if mask.sum() < arr.shape[0]:
        print(f"[warn] dropped {arr.shape[0] - mask.sum()} non-finite rows")
    arr = arr[mask]
    return _ensure_contig_f32(arr)


def _safe_read_csv_to_numpy(
    csv_path: str,
    timeout_warn_s: float = 30.0,
    allow_skip_bad_lines: bool = True
) -> np.ndarray:
    """
    Legge un CSV velocemente; se fallisce, riprova con engine='python' e on_bad_lines='skip'.
    Logga quanto impiega e warning se > timeout_warn_s.
    """
    t0 = time.time()
    try:
        df = pd.read_csv(csv_path, sep=",", header=None, dtype=np.float32, engine="c")
        arr = df.values
    except Exception as e_c:
        print(f"[warn] pandas C-engine failed for {csv_path}: {e_c}")
        try:
            df = pd.read_csv(
                csv_path,
                sep=",",
                header=None,
                dtype=np.float32,
                engine="python",
                on_bad_lines="skip" if allow_skip_bad_lines else "error",
            )
            arr = df.values
        except Exception as e_py:
            raise RuntimeError(f"[fatal] both engines failed for {csv_path}: {e_py}") from e_py

    dt = time.time() - t0
    if dt > timeout_warn_s:
        print(f"[warn] slow read: {csv_path} took {dt:.1f}s")

    return _clean_numeric_ndarray(arr)


class SdfLoader(base.Dataset):
    """
    Versione robusta con:
    - cache su disco per CSV e BPS,
    - fallback engine CSV,
    - logging del file corrente,
    - sanitizzazione dei dati,
    - skip opzionale dei file problematici.
    """

    def __init__(
        self,
        data_source: str,
        split_file,
        grid_source: Optional[str] = None,
        samples_per_mesh: int = 16000,
        pc_size: int = 1024,
        modulation_path: Optional[str] = None,
        cache_dir: str = ".sdf_cache",
        skip_bad_files: bool = True,
        csv_slow_warn_s: float = 30.0,
    ):
        self.samples_per_mesh = samples_per_mesh
        self.pc_size = pc_size
        self.grid_source = grid_source
        self.cache_dir = cache_dir
        self.skip_bad_files = skip_bad_files
        self.csv_slow_warn_s = csv_slow_warn_s

        os.makedirs(self.cache_dir, exist_ok=True)

        # --- paths
        self.gt_files = self.get_instance_filenames(
            data_source, split_file, filter_modulation_path=modulation_path
        )
        self.paths = list(self.gt_files)

        if grid_source:
            self.grid_files = self.get_instance_filenames(
                grid_source,
                split_file,
                gt_filename="grid_gt.csv",
                filter_modulation_path=modulation_path,
            )
            self.grid_files = self.grid_files[: len(self.gt_files)]
            assert len(self.grid_files) == len(self.gt_files)
        else:
            self.grid_files = None

        # --- BPS grid
        self.bps_grid = self._create_bps_grid(grid_size=32, radius=1.5).contiguous()
        self._bps_grid_np = self.bps_grid.cpu().numpy()

        # --- cache lists
        self.gt_tensors = []
        self.preprocessed_bps = []
        self.grid_tensors = [] if self.grid_source else None

        # --- bad files (persistente)
        self._bad_file_list_path = os.path.join(self.cache_dir, "bad_files.txt")
        self._bad = set()
        if os.path.exists(self._bad_file_list_path):
            with open(self._bad_file_list_path, "r") as f:
                self._bad |= {ln.strip() for ln in f if ln.strip()}

        print(f"loading all {len(self.gt_files)} files into memory (with cache)…")

        keep_gt_files = []
        keep_grid_files = [] if self.grid_source else None

        for i, f in enumerate(tqdm(self.gt_files, desc="GT+BPS cache")):
            if f in self._bad:
                print(f"[skip] previously bad file: {f}")
                continue

            try:
                # 1) GT tensor via cache
                gt_tensor = self._load_or_cache_csv(f, key="gt")
                # 2) point cloud → BPS via cache
                pc = self.get_pointcloud(gt_tensor, load_from_path=False)  # (N,3)
                bps_tensor = self._load_or_cache_bps(f, pc)
            except Exception as e:
                msg = f"[error] failed on {f}: {e}"
                print(msg)
                if self.skip_bad_files:
                    self._mark_bad_file(f)
                    continue
                else:
                    raise

            self.gt_tensors.append(gt_tensor)
            self.preprocessed_bps.append(bps_tensor)
            keep_gt_files.append(f)

            if self.grid_source:
                gf = self.grid_files[i]
                try:
                    grid_tensor = self._load_or_cache_csv(gf, key="grid")
                except Exception as e:
                    msg = f"[error] failed on GRID {gf} (for {f}): {e}"
                    print(msg)
                    if self.skip_bad_files:
                        self._mark_bad_file(f)
                        continue
                    else:
                        raise
                self.grid_tensors.append(grid_tensor)
                keep_grid_files.append(gf)

        # riduci le liste ai soli “buoni”
        self.paths = keep_gt_files
        if self.grid_source:
            self.grid_files = keep_grid_files

        if len(self.gt_tensors) == 0:
            raise RuntimeError("No valid files found after filtering. Check your data paths.")

        print(f"[ok] kept {len(self.gt_tensors)} / {len(self.gt_files)} items.")

    # ------------------------------------------------------------------ #
    # caching helpers
    # ------------------------------------------------------------------ #

    def _cache_file(self, path: str, kind: str) -> str:
        h = _hash_path(path)
        return os.path.join(self.cache_dir, f"{kind}_{h}.pt")

    def _load_or_cache_csv(self, csv_path: str, key: str = "gt") -> torch.Tensor:
        cache_path = self._cache_file(csv_path, key)
        if os.path.exists(cache_path):
            return torch.load(cache_path, map_location="cpu")

        arr = _safe_read_csv_to_numpy(csv_path, timeout_warn_s=self.csv_slow_warn_s)
        tens = torch.from_numpy(_ensure_contig_f32(arr))  # (N, 3|4)
        torch.save(tens, cache_path)
        return tens

    def _load_or_cache_bps(self, src_path: str, pointcloud_tensor: torch.Tensor) -> torch.Tensor:
        cache_path = self._cache_file(src_path, "bps")
        if os.path.exists(cache_path):
            return torch.load(cache_path, map_location="cpu")

        bps_tensor = self._get_base_points_from_pc(pointcloud_tensor)  # (3,32,32,32)
        torch.save(bps_tensor, cache_path)
        return bps_tensor

    def _mark_bad_file(self, path: str):
        self._bad.add(path)
        try:
            with open(self._bad_file_list_path, "a") as f:
                f.write(path + "\n")
        except Exception:
            pass

    # ------------------------------------------------------------------ #
    # dataset API
    # ------------------------------------------------------------------ #

    def __len__(self):
        return len(self.gt_tensors)

    def __getitem__(self, idx: int):
        near_surface_count = int(self.samples_per_mesh * 0.7) if self.grid_source else self.samples_per_mesh

        pc, sdf_xyz, sdf_gt = self.labeled_sampling(
            self.gt_tensors[idx], near_surface_count, self.pc_size, load_from_path=False
        )

        basis_point = self.preprocessed_bps[idx]  # (3,32,32,32)

        grid = None
        if self.grid_source:
            grid_count = self.samples_per_mesh - near_surface_count
            _, grid_xyz, grid_gt = self.labeled_sampling(
                self.grid_tensors[idx], grid_count, pc_size=grid_count, load_from_path=False
            )
            sdf_xyz = torch.cat((sdf_xyz, grid_xyz), dim=0)
            sdf_gt = torch.cat((sdf_gt, grid_gt), dim=0)
            grid = self.get_grid(self.grid_tensors[idx], load_from_path=False)

        return {
            "xyz": sdf_xyz.float().contiguous(),
            "gt_sdf": sdf_gt.float().contiguous(),
            "basis_point": basis_point.float().contiguous(),
            "grid_point": grid.float().contiguous() if (self.grid_source and grid is not None) else None,
            "point_cloud": pc.float().contiguous(),
            "paths": self.paths[idx],
        }

    # ------------------------------------------------------------------ #
    # utilities
    # ------------------------------------------------------------------ #

    def _create_bps_grid(self, grid_size=32, radius=1.5):
        bps_grid_np = bps.generate_grid_basis(
            grid_size=grid_size, n_dims=3, minv=-radius, maxv=radius
        ).astype(np.float32, copy=False)
        return torch.from_numpy(np.ascontiguousarray(bps_grid_np))

    def _get_base_points_from_pc(self, pointcloud: torch.Tensor) -> torch.Tensor:
        """
        pointcloud: (N,3) float32 CPU
        returns: (3, 32, 32, 32) float32
        """
        pc_np = pointcloud.detach().cpu().numpy().astype(np.float32, copy=False)
        if pc_np.ndim != 2 or pc_np.shape[1] != 3:
            raise ValueError(f"expected pointcloud (N,3), got {pc_np.shape}")

        pc_np = pc_np[np.newaxis, ...]  # (1,N,3)
        pc_norm = bps.normalize(pc_np)

        x_bps = bps.encode(
            pc_norm,
            bps_arrangement="custom",
            custom_basis=self._bps_grid_np,
            bps_cell_type="deltas",
            n_jobs=1,
        )  # (1, 32*32*32, 3)

        x_bps = x_bps.reshape(1, 32, 32, 32, 3).transpose(0, 4, 1, 2, 3)  # (1,3,32,32,32)
        x_bps = _ensure_contig_f32(x_bps)
        return torch.from_numpy(x_bps).float().squeeze(0)
