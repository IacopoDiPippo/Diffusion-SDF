# dataloader/sdf_loader.py

import os, sys, hashlib, contextlib, signal, time
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from bps import bps

from . import base  # la tua base.Dataset

# ---------------- timeout helper ----------------
@contextlib.contextmanager
def time_limit(seconds: int, on_file: str = ""):
    def handler(signum, frame):
        raise TimeoutError(f"Timeout ({seconds}s) while reading: {on_file}")
    old = signal.signal(signal.SIGALRM, handler)
    signal.alarm(int(seconds))
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old)


class SdfLoader(base.Dataset):
    def __init__(self,
                 data_source,
                 split_file,
                 grid_source=None,
                 samples_per_mesh=16000,
                 pc_size=1024,
                 modulation_path=None,
                 cache_dir=".sdf_cache",
                 csv_timeout_s=300,
                 max_retries=0,
                 skip_bad_files=True,
                 verbose_every=1):

        self.samples_per_mesh = int(samples_per_mesh)
        self.pc_size = int(pc_size)
        self.grid_source = grid_source
        self.cache_dir = cache_dir
        self.csv_timeout_s = int(csv_timeout_s)
        self.max_retries = int(max_retries)
        self.skip_bad_files = bool(skip_bad_files)
        self.verbose_every = max(1, int(verbose_every))

        os.makedirs(self.cache_dir, exist_ok=True)

        # --- raccogli lista file ---
        self.gt_files = self.get_instance_filenames(
            data_source, split_file, filter_modulation_path=modulation_path
        )
        self.paths = list(self.gt_files)

        if grid_source:
            self.grid_files = self.get_instance_filenames(
                grid_source, split_file, gt_filename="grid_gt.csv",
                filter_modulation_path=modulation_path
            )
            assert len(self.grid_files) == len(self.gt_files), "grid/gt size mismatch"
        else:
            self.grid_files = None

        # --- griglia BPS fissa (una volta sola) ---
        self.bps_grid = self._create_bps_grid(grid_size=32, radius=1.5).contiguous()
        self._bps_grid_np = self.bps_grid.cpu().numpy()

        kept_gt_tensors = []
        kept_bps = []
        kept_paths = []
        kept_grid_tensors = [] if self.grid_source else None

        print(f"[SdfLoader] caching {len(self.gt_files)} items (CSV→.npy, BPS→.npy, mmap per lettura).")

        for i, f in enumerate(self.gt_files):
            if (i % self.verbose_every) == 0:
                try:
                    fsize = os.path.getsize(f)/1e6
                except Exception:
                    fsize = -1
                print(f"[{i+1}/{len(self.gt_files)}] GT: {f} ({fsize:.2f} MB)")

            # 1) CSV → tensore via cache .npy
            gt_tensor = self._load_or_cache_csv(f, key="gt")
            if gt_tensor is None:
                msg = f"[skip] GT failed: {f}"
                print(msg, flush=True)
                if not self.skip_bad_files:
                    raise RuntimeError(msg)
                continue

            # 2) ricava pc e BPS via cache .npy
            try:
                pc = self.get_pointcloud(gt_tensor, load_from_path=False)  # (N,3)
                bps_tensor = self._load_or_cache_bps(f, pc)                # (3,32,32,32)
                if bps_tensor is None:
                    raise RuntimeError("BPS is None")
            except Exception as e:
                msg = f"[skip] BPS fail for {f}: {e}"
                print(msg, flush=True)
                if not self.skip_bad_files:
                    raise
                continue

            # 3) grid (opzionale) via cache .npy
            if self.grid_source:
                gf = self.grid_files[i]
                grid_tensor = self._load_or_cache_csv(gf, key="grid")
                if grid_tensor is None:
                    msg = f"[skip] GRID failed: {gf}"
                    print(msg, flush=True)
                    if not self.skip_bad_files:
                        raise RuntimeError(msg)
                    continue
                kept_grid_tensors.append(grid_tensor)

            # tieni
            kept_gt_tensors.append(gt_tensor)
            kept_bps.append(bps_tensor)
            kept_paths.append(f)

        self.gt_tensors = kept_gt_tensors
        self.preprocessed_bps = kept_bps          # sempre presente
        self.paths = kept_paths
        self.grid_tensors = kept_grid_tensors if self.grid_source else None

        print(f"[SdfLoader] kept {len(self.gt_tensors)} / {len(self.gt_files)} items.")

    # ---------------- cache helpers (.npy + mmap) ----------------

    def _hash_path(self, path):
        try:
            st = os.stat(path)
            meta = f"{path}|{st.st_size}|{int(st.st_mtime)}"
        except OSError:
            meta = path
        return hashlib.sha1(meta.encode("utf-8")).hexdigest()

    def _cache_file(self, path, kind, ext="npy"):
        h = self._hash_path(path)
        return os.path.join(self.cache_dir, f"{kind}_{h}.{ext}")

    def _fast_read_csv(self, csv_path: str):
        """
        Lettura robusta CSV → torch.Tensor float32
        - timeout
        - engine='c' con fallback 'python'
        - valida col dim=3/4
        """
        def _read(engine):
            with time_limit(self.csv_timeout_s, on_file=csv_path):
                arr = pd.read_csv(csv_path, sep=",", header=None,
                                  dtype=np.float32, engine=engine).values
            return arr

        # prova C-engine poi fallback
        for attempt in range(self.max_retries + 1):
            try:
                try:
                    arr = _read("c")
                except Exception:
                    arr = _read("python")
                if arr.ndim != 2 or arr.shape[1] not in (3, 4):
                    print(f"[invalid CSV shape] {csv_path}: {arr.shape}")
                    return None
                return torch.from_numpy(np.ascontiguousarray(arr))
            except TimeoutError as e:
                print(f"[timeout] {e}", flush=True)
                return None
            except Exception as e:
                print(f"[read_csv warn] {csv_path} attempt {attempt+1}: {e}", flush=True)
                if attempt == self.max_retries:
                    return None
        return None

    def _load_or_cache_csv(self, csv_path, key="gt"):
        """
        Se esiste la cache .npy → np.load(mmap_mode='r') → torch.from_numpy (view, zero-copy)
        Altrimenti legge CSV (con timeout) e salva .npy.
        """
        cache_path = self._cache_file(csv_path, key, ext="npy")
        if os.path.exists(cache_path):
            try:
                arr = np.load(cache_path, mmap_mode='r')
                return torch.from_numpy(np.asarray(arr))
            except Exception as e:
                print(f"[cache load fail] {cache_path}: {e}; re-reading CSV.", flush=True)

        tens = self._fast_read_csv(csv_path)
        if tens is not None:
            try:
                np.save(cache_path, tens.numpy())
            except Exception as e:
                print(f"[cache save fail] {cache_path}: {e}", flush=True)
        return tens

    def _load_or_cache_bps(self, src_path, pointcloud_tensor):
        """
        BPS su .npy con mmap. Se assente, calcola e salva.
        """
        cache_path = self._cache_file(src_path, "bps", ext="npy")
        if os.path.exists(cache_path):
            try:
                arr = np.load(cache_path, mmap_mode='r')
                return torch.from_numpy(np.asarray(arr))
            except Exception as e:
                print(f"[cache load fail] {cache_path}: {e}; recomputing BPS.", flush=True)

        try:
            bps_tensor = self._get_base_points_from_pc(pointcloud_tensor)  # torch.Tensor (3,32,32,32)
            np.save(cache_path, bps_tensor.numpy())
            return bps_tensor
        except Exception as e:
            print(f"[bps fail] {src_path}: {e}", flush=True)
            return None

    # ---------------- Dataset API ----------------

    def __len__(self):
        return len(self.gt_tensors)

    def __getitem__(self, idx):
        near_surface_count = int(self.samples_per_mesh * 0.7) if self.grid_source else self.samples_per_mesh

        pc, sdf_xyz, sdf_gt = self.labeled_sampling(
            self.gt_tensors[idx], near_surface_count, self.pc_size, load_from_path=False
        )

        basis_point = self.preprocessed_bps[idx]  # sempre presente

        grid = None
        if self.grid_source:
            grid_count = self.samples_per_mesh - near_surface_count
            _, grid_xyz, grid_gt = self.labeled_sampling(
                self.grid_tensors[idx], grid_count, pc_size=grid_count, load_from_path=False
            )
            sdf_xyz = torch.cat((sdf_xyz, grid_xyz), dim=0)
            sdf_gt  = torch.cat((sdf_gt,  grid_gt),  dim=0)
            grid = self.get_grid(self.grid_tensors[idx], load_from_path=False)

        return {
            "xyz":         sdf_xyz.float().contiguous().squeeze(),
            "gt_sdf":      sdf_gt.float().contiguous().squeeze(),
            "basis_point": basis_point.float().contiguous().squeeze(),
            "grid_point":  (grid.float().contiguous().squeeze()
                            if (self.grid_source and grid is not None) else None),
            "point_cloud": pc.float().contiguous().squeeze(),
            "paths":       self.paths[idx],
        }

    # ---------------- utilities ----------------

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
        pc_np = pc_np[np.newaxis, ...]  # (1,N,3)
        pc_norm = bps.normalize(pc_np)
        x_bps = bps.encode(
            pc_norm,
            bps_arrangement="custom",
            custom_basis=self._bps_grid_np,
            bps_cell_type="deltas",
            n_jobs=1
        )  # (1, 32*32*32, 3)
        x_bps = x_bps.reshape(1, 32, 32, 32, 3).transpose(0, 4, 1, 2, 3)  # (1,3,32,32,32)
        return torch.from_numpy(np.ascontiguousarray(x_bps)).float().squeeze(0)
