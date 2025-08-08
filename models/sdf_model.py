#!/usr/bin/env python3

import torch.nn as nn
import torch
import torch.nn.functional as F
import pytorch_lightning as pl 

import sys
import os 
from pathlib import Path
import numpy as np 
import math

from einops import rearrange, reduce

from models.archs.sdf_decoder import * 
from models.archs.encoders.conv_pointnet import ConvPointnet
from utils import mesh, evaluate


class SdfModel(pl.LightningModule):

    def __init__(self, specs):
        super().__init__()
        
        self.specs = specs
        model_specs = self.specs["SdfModelSpecs"]
        self.hidden_dim = model_specs["hidden_dim"]
        self.latent_dim = model_specs["latent_dim"]
        self.skip_connection = model_specs.get("skip_connection", True)
        self.tanh_act = model_specs.get("tanh_act", False)

        self.model = SdfDecoder( hidden_dim=self.hidden_dim, skip_connection=self.skip_connection, tanh_act=self.tanh_act, latent_size= self.latent_dim)
        
        self.model.train()
        #print(self.model)


    def configure_optimizers(self):

        optimizer = torch.optim.Adam(self.parameters(), self.specs["sdf_lr"])
        return optimizer

 
    def training_step(self, x, idx):

        xyz = x['xyz'] # (B, 16000, 3)
        gt = x['gt_sdf'] # (B, 16000)
        pc = x['point_cloud'] # (B, 1024, 3)

        shape_features = self.pointnet(pc, xyz)

        pred_sdf = self.model(xyz, shape_features)

        sdf_loss = F.l1_loss(pred_sdf.squeeze(), gt.squeeze(), reduction = 'none')
        sdf_loss = reduce(sdf_loss, 'b ... -> b (...)', 'mean').mean()
    
        return sdf_loss 
            
    

    def forward(self, pc, xyz):
        shape_features = self.pointnet(pc, xyz)

        return self.model(xyz, shape_features).squeeze()

    def forward_with_plane_features(self, plane_features, xyz):
        '''
        plane_features: B, D*3, res, res (e.g. B, 768, 64, 64)
        xyz: B, N, 3
        '''
        point_features = self.pointnet.forward_with_plane_features(plane_features, xyz) # point_features: B, N, D
        pred_sdf = self.model( torch.cat((xyz, point_features),dim=-1) )  
        return pred_sdf # [B, num_points] 


    def forward_with_base_features(self, base_features, xyz):
        # Your original computation
        # Expand base features to match point count
        base_features = base_features.unsqueeze(1)  # [B, 1, D]
        base_features = base_features.expand(-1, xyz.shape[1], -1)  # [B, N, D]
        """ print("Shape of base features_", base_features.shape)
        print("Base point feature checksum:", base_features.sum())
        print("XYZ checksum:", xyz.sum())"""
        combined_input = torch.cat((xyz, base_features), dim=-1)
        pred_sdf = self.model(combined_input)  # [B, num_points]
        
        # Single debug call at the end (can be easily removed)
        self.debug_shapes(
            xyz=xyz,
            base_features=base_features,
            combined_input=combined_input,
            pred_sdf=pred_sdf
        )
        
        return pred_sdf

    def debug_shapes(self,**kwargs):
        """Prints shapes/types of all provided variables. Call this at the end of your function."""
        if False:
            print("\n=== Debug Shapes ===")
            for name, value in kwargs.items():
                shape = str(list(value.shape)) if hasattr(value, 'shape') else str(len(value)) if hasattr(value, '__len__') else 'scalar'
                dtype = str(value.dtype) if hasattr(value, 'dtype') else type(value).__name__
                print(f"{name.ljust(20)}: shape={shape.ljust(25)} type={dtype}")
            print("==================\n")