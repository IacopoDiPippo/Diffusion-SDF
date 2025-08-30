#!/usr/bin/env python3

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

    def __init__(
        self,
        data_source,
        split_file,
        grid_source=None,
        samples_per_mesh=16000,
        pc_size=1024,
        modulation_path=None
    ):
        self.samples_per_mesh = samples_per_mesh
        self.pc_size = pc_size

        # ---- solo liste di path, niente CSV in RAM
        self.paths = self.get_instance_filenames(
            data_source, split_file, filter_modulation_path=modulation_path
        )
        subsample = len(self.paths)
        self.paths = self.paths[:subsample]

        self.grid_source = grid_source
        if grid_source:
            self.grid_paths = self.get_instance_filenames(
                grid_source, split_file, gt_filename="grid_gt.csv",
                filter_modulation_path=modulation_path
            )
            self.grid_paths = self.grid_paths[:subsample]
            assert len(self.grid_paths) == len(self.paths)

        # ---- BPS: precompute UNA VOLTA (e tieni in RAM solo questi)
        self.bps_grid = self._create_bps_grid(grid_size=32, radius=1.5)

        self.preprocessed_bps = []
        print(f"precomputing BPS for {len(self.paths)} files...")
        for f in tqdm(self.paths):
            # carica CSV -> ricava pc -> BPS -> salva SOLO BPS
            data = torch.from_numpy(pd.read_csv(f, sep=',', header=None).values)
            pc = self.get_pointcloud(data, load_from_path=False)   # (N,3)
            pc_bps = self.get_base_points(pc)                      # (3,32,32,32)
            self.preprocessed_bps.append(pc_bps)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        # ---- lazy: carica GT solo qui
        gt_path = self.paths[idx]
        data = torch.from_numpy(pd.read_csv(gt_path, sep=',', header=None).values)

        near_surface_count = int(self.samples_per_mesh * 0.7) if self.grid_source else self.samples_per_mesh

        pc, sdf_xyz, sdf_gt = self.labeled_sampling(
            data, near_surface_count, self.pc_size, load_from_path=False
        )

        basis_point = self.preprocessed_bps[idx]  # già precalcolato

        grid = None
        if self.grid_source is not None:
            # lazy: carica grid solo qui
            grid_count = self.samples_per_mesh - near_surface_count
            grid_path = self.grid_paths[idx]
            # usa load_from_path=True per far leggere direttamente dentro labeled_sampling
            _, grid_xyz, grid_gt = self.labeled_sampling(
                grid_path, grid_count, pc_size=grid_count, load_from_path=True
            )
            sdf_xyz = torch.cat((sdf_xyz, grid_xyz), dim=0)
            sdf_gt  = torch.cat((sdf_gt,  grid_gt),  dim=0)

            # se ti serve anche la griglia densa:
            grid_data = torch.from_numpy(pd.read_csv(grid_path, sep=',', header=None).values)
            grid = self.get_grid(grid_data, load_from_path=False)

        data_dict = {
            "xyz":         sdf_xyz.float().squeeze(),
            "gt_sdf":      sdf_gt.float().squeeze(),
            "basis_point": basis_point.float().squeeze(),
            "grid_point":  grid.float().squeeze() if (self.grid_source and grid is not None) else None,
            "point_cloud": pc.float().squeeze(),
            "paths":       gt_path,
        }
        return data_dict

    def _create_bps_grid(self, grid_size=32, radius=1.5):
        bps_grid_np = bps.generate_grid_basis(
            grid_size=grid_size, n_dims=3, minv=-radius, maxv=radius
        )
        return torch.from_numpy(bps_grid_np).float()

    def get_base_points(self, pointcloud: torch.Tensor) -> torch.Tensor:
        pointcloud_np = pointcloud.detach().cpu().numpy()[np.newaxis, ...]  # (1,N,3)
        pc_normalized = bps.normalize(pointcloud_np)
        current_grid = self.bps_grid.cpu().numpy()
        x_bps_grid = bps.encode(
            pc_normalized,
            bps_arrangement='custom',
            custom_basis=current_grid,
            bps_cell_type='deltas',
            n_jobs=1
        )  # (1, 32*32*32, 3)
        x_bps_grid = x_bps_grid.reshape(1, 32, 32, 32, 3).transpose(0, 4, 1, 2, 3)  # (1,3,32,32,32)
        return torch.from_numpy(x_bps_grid).float().squeeze(0)
