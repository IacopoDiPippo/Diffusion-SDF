import os, hashlib, json, numpy as np, pandas as pd, torch
from tqdm import tqdm
from bps import bps

import time 
import logging
import os
import random
import torch
import torch.utils.data
from . import base 
from bps import bps
import pandas as pd 
import numpy as np
import csv, json

from tqdm import tqdm
class SdfLoader(base.Dataset):
    def __init__(self,
                 data_source,
                 split_file,
                 grid_source=None,
                 samples_per_mesh=16000,
                 pc_size=1024,
                 modulation_path=None,
                 cache_dir=".sdf_cache"):   # <-- NEW: on-disk cache
        self.samples_per_mesh = samples_per_mesh
        self.pc_size = pc_size
        self.grid_source = grid_source
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)

        # ---- collect paths (unchanged)
        self.gt_files = self.get_instance_filenames(
            data_source, split_file, filter_modulation_path=modulation_path
        )
        self.gt_files = self.gt_files[:len(self.gt_files)]
        self.paths = self.gt_files

        if grid_source:
            self.grid_files = self.get_instance_filenames(
                grid_source, split_file, gt_filename="grid_gt.csv",
                filter_modulation_path=modulation_path
            )
            self.grid_files = self.grid_files[:len(self.gt_files)]
            assert len(self.grid_files) == len(self.gt_files)
        else:
            self.grid_files = None

        # ---- precompute BPS grid (once)
        self.bps_grid = self._create_bps_grid(grid_size=32, radius=1.5).contiguous()
        self._bps_grid_np = self.bps_grid.cpu().numpy()

        # ---- EAGER LOAD, but via CACHE for speed
        self.gt_tensors = []
        self.preprocessed_bps = []
        self.grid_tensors = [] if self.grid_source else None

        print(f"loading all {len(self.gt_files)} files into memory (with cache)…")
        for i, f in enumerate(tqdm(self.gt_files, desc="GT+FPS cache")):
            # 1) GT tensor (CSV -> Tensor) with cache
            gt_tensor = self._load_or_cache_csv(f, key="gt")
            self.gt_tensors.append(gt_tensor)

            # 2) point cloud from GT (no cache, it’s fast and small) -> BPS (cached)
            pc = self.get_pointcloud(gt_tensor, load_from_path=False)  # (N,3) float32
            bps_tensor = self._load_or_cache_bps(f, pc)
            self.preprocessed_bps.append(bps_tensor)

            # 3) grid tensor (optional) via cache
            if self.grid_source:
                gf = self.grid_files[i]
                grid_tensor = self._load_or_cache_csv(gf, key="grid")
                self.grid_tensors.append(grid_tensor)

    # ============================ caching helpers ============================

    def _hash_path(self, path):
        # include file size + mtime to invalidate cache on change
        try:
            st = os.stat(path)
            meta = f"{path}|{st.st_size}|{int(st.st_mtime)}"
        except OSError:
            meta = path
        return hashlib.sha1(meta.encode("utf-8")).hexdigest()

    def _cache_file(self, path, kind):
        h = self._hash_path(path)
        return os.path.join(self.cache_dir, f"{kind}_{h}.pt")

    def _load_or_cache_csv(self, csv_path, key="gt"):
        cache_path = self._cache_file(csv_path, key)
        if os.path.exists(cache_path):
            return torch.load(cache_path, map_location="cpu")

        # Fast CSV → float32 tensor
        arr = pd.read_csv(csv_path, sep=",", header=None, dtype=np.float32, engine="c").values
        tens = torch.from_numpy(np.ascontiguousarray(arr))  # (N, 4) or (N,3)
        torch.save(tens, cache_path)
        return tens

    def _load_or_cache_bps(self, src_path, pointcloud_tensor):
        cache_path = self._cache_file(src_path, "bps")
        if os.path.exists(cache_path):
            return torch.load(cache_path, map_location="cpu")

        # Compute BPS once, save
        bps_tensor = self._get_base_points_from_pc(pointcloud_tensor)  # (3,32,32,32) float32
        torch.save(bps_tensor, cache_path)
        return bps_tensor

    # ============================ dataset API ============================

    def __len__(self):
        return len(self.gt_tensors)

    def __getitem__(self, idx):
        near_surface_count = int(self.samples_per_mesh * 0.7) if self.grid_source else self.samples_per_mesh

        pc, sdf_xyz, sdf_gt = self.labeled_sampling(
            self.gt_tensors[idx], near_surface_count, self.pc_size, load_from_path=False
        )

        basis_point = self.preprocessed_bps[idx]  # always present

        grid = None
        if self.grid_source:
            grid_count = self.samples_per_mesh - near_surface_count
            # labeled_sampling expects same API; we pass cached tensor here
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
            "grid_point":  grid.float().contiguous().squeeze() if (self.grid_source and grid is not None) else None,
            "point_cloud": pc.float().contiguous().squeeze(),
            "paths":       self.paths[idx],
        }

    # ============================ utilities ============================

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
            custom_basis=self._bps_grid_np,  # prebuilt numpy grid
            bps_cell_type="deltas",
            n_jobs=1
        )  # (1, 32*32*32, 3)
        x_bps = x_bps.reshape(1, 32, 32, 32, 3).transpose(0, 4, 1, 2, 3)  # (1,3,32,32,32)
        return torch.from_numpy(np.ascontiguousarray(x_bps)).float().squeeze(0)
