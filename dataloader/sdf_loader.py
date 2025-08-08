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
        data_source, # path to points sampled around surface
        split_file, # json filepath which contains train/test classes and meshes 
        grid_source=None, # path to grid points; grid refers to sampling throughout the unit cube instead of only around the surface; necessary for preventing artifacts in empty space
        samples_per_mesh=16000,
        pc_size=1024,
        modulation_path=None # used for third stage of training; needs to be set in config file when some modulation training had been filtered
    ):
 
        self.samples_per_mesh = samples_per_mesh
        self.pc_size = pc_size
        self.gt_files = self.get_instance_filenames(data_source, split_file, filter_modulation_path=modulation_path)

        subsample = len(self.gt_files) 
        self.gt_files = self.gt_files[0:subsample]

        self.grid_source = grid_source
        #print("grid source: ", grid_source)

        self.bps_grid = self._create_bps_grid(grid_size=32, radius=1.5)

        if grid_source:
            self.grid_files = self.get_instance_filenames(grid_source, split_file, gt_filename="grid_gt.csv", filter_modulation_path=modulation_path)
            self.grid_files = self.grid_files[0:subsample]
            lst = []
            with tqdm(self.grid_files) as pbar:
                for i, f in enumerate(pbar):
                    pbar.set_description("Grid files loaded: {}/{}".format(i, len(self.grid_files)))
                    lst.append(torch.from_numpy(pd.read_csv(f, sep=',',header=None).values))
            self.grid_files = lst
            
            assert len(self.grid_files) == len(self.gt_files)


        print("loading all {} files into memory...".format(len(self.gt_files)))
        loaded_data = []
        preprocessed_pcs = []
        preprocessed_bps = []

        with tqdm(self.gt_files) as pbar:
            for i, f in enumerate(pbar):
                pbar.set_description("Files loaded: {}/{}".format(i, len(self.gt_files)))
                
                # Load CSV file as tensor
                data = torch.from_numpy(pd.read_csv(f, sep=',', header=None).values)
                
                # Get pointcloud from loaded data
                pc = self.get_pointcloud(data, load_from_path=False)
                
                # Apply basis point encoding to the pointcloud
                pc_bps = self.get_base_points(pc)
                
                # Save all for later use
                loaded_data.append(data)
                preprocessed_pcs.append(pc)
                preprocessed_bps.append(pc_bps)

        self.gt_files = loaded_data          # raw loaded CSV tensors
        self.preprocessed_pcs = preprocessed_pcs   # raw pointclouds (N, 3)
        self.preprocessed_bps = preprocessed_bps   # basis point encoded tensors



  
    def __getitem__(self, idx): 

        near_surface_count = int(self.samples_per_mesh*0.7) if self.grid_source else self.samples_per_mesh

        _, sdf_xyz, sdf_gt =  self.labeled_sampling(self.gt_files[idx], near_surface_count, self.pc_size, load_from_path=False)
        
        basis_point = self.preprocessed_bps[idx]
        if self.grid_source is not None:
            grid_count = self.samples_per_mesh - near_surface_count
            _, grid_xyz, grid_gt = self.labeled_sampling(self.grid_files[idx], grid_count, pc_size=grid_count, load_from_path=False)
            # each getitem is one batch so no batch dimension, only N, 3 for xyz or N for gt 
            # for 16000 points per batch, near surface is 11200, grid is 4800
            #print("shapes: ", pc.shape,  sdf_xyz.shape, sdf_gt.shape, grid_xyz.shape, grid_gt.shape)
            sdf_xyz = torch.cat((sdf_xyz, grid_xyz))
            sdf_gt = torch.cat((sdf_gt, grid_gt))
            #print("shapes after adding grid: ", pc.shape, sdf_xyz.shape, sdf_gt.shape, grid_xyz.shape, grid_gt.shape)

        data_dict = {
                    "xyz":sdf_xyz.float().squeeze(),
                    "gt_sdf":sdf_gt.float().squeeze(), 
                    "basis_point":basis_point.float().squeeze(),
                    }

        return data_dict
  

    def _create_bps_grid(self, grid_size=32, radius=1.5):
        """Create a fixed BPS reference grid."""
        bps_grid_np = bps.generate_grid_basis(
            grid_size=grid_size,
            n_dims=3,
            minv=-radius,
            maxv=radius
        )
        return torch.from_numpy(bps_grid_np).float()

    def get_base_points(self, pointcloud: torch.Tensor) -> torch.Tensor:
        """
        Normalize a single pointcloud and encode it into BPS features 
        using a fixed, precomputed BPS grid basis.

        Parameters:
            pointcloud: torch.Tensor, shape (N, 3)
                Single point cloud.

        Returns:
            torch.Tensor with BPS encoding of shape (n_bps_points, 3)
        """
        # Convert to numpy for bps operations
        pointcloud_np = pointcloud.detach().cpu().numpy()  # shape (N, 3)

        # Add batch dimension (needed for bps functions expecting [batch, points, dims])
        pointcloud_np = pointcloud_np[np.newaxis, ...]  # shape (1, N, 3)

        # Normalize (assuming bps.normalize can handle (N,3) arrays)
        pc_normalized = bps.normalize(pointcloud_np)       # shape (1, N, 3)

        # Check if the fixed bps_grid has changed (should not change!)
        current_grid = self.bps_grid.cpu().numpy()
       
        # Encode using fixed custom BPS grid
        x_bps = bps.encode(
            pc_normalized,
            bps_arrangement='custom',
            custom_basis=current_grid,
            bps_cell_type='deltas',
            n_jobs=1
        )  # output shape: (1, n_bps_points, 3)

        return torch.from_numpy(x_bps).float().squeeze(0)


    def __len__(self):
        return len(self.gt_files)



    
