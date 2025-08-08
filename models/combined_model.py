import torch

import torch.utils.data 
from torch.nn import functional as F
from bps import bps
import pytorch_lightning as pl
from bps import bps
# add paths in model/__init__.py for new models
from models import * 

class CombinedModel(pl.LightningModule):
    def __init__(self, specs):
        super().__init__()
        self.specs = specs


        self.task = specs['training_task'] # 'combined' or 'modulation' or 'diffusion'

        if self.task in ('combined', 'modulation'):
            self.sdf_model = SdfModel(specs=specs) 

            feature_dim = specs["SdfModelSpecs"]["latent_dim"] # latent dim of pointnet 
            modulation_dim = feature_dim # latent dim of modulation
            latent_std = specs.get("latent_std", 0.25) # std of target gaussian distribution of latent space
            hidden_dims = [modulation_dim, modulation_dim, modulation_dim, modulation_dim, modulation_dim]
            self.vae_model = BetaVAE(in_channels=3, latent_dim=feature_dim, hidden_dims=None, kl_std=latent_std)
        if self.task in ('combined', 'diffusion'):
            self.diffusion_model = DiffusionModel(model=DiffusionNet(**specs["diffusion_model_specs"]), **specs["diffusion_specs"]) 
 
        self.bps_grid = self._create_bps_grid()

    def training_step(self, x, idx):

        if self.task == 'combined':
            return self.train_combined(x)
        elif self.task == 'modulation':
            return self.train_modulation_base_points(x)
        elif self.task == 'diffusion':
            return self.train_diffusion(x)
        

    def configure_optimizers(self):

        if self.task == 'combined':
            params_list = [
                    { 'params': list(self.sdf_model.parameters()) + list(self.vae_model.parameters()), 'lr':self.specs['sdf_lr'] },
                    { 'params': self.diffusion_model.parameters(), 'lr':self.specs['diff_lr'] }
                ]
        elif self.task == 'modulation':
            params_list = [
                    { 'params': self.parameters(), 'lr':self.specs['sdf_lr'] }
                ]
        elif self.task == 'diffusion':
            params_list = [
                    { 'params': self.parameters(), 'lr':self.specs['diff_lr'] }
                ]

        optimizer = torch.optim.Adam(params_list)
        return {
                "optimizer": optimizer,
                # "lr_scheduler": {
                # "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=50000, threshold=0.0002, min_lr=1e-6, verbose=False),
                # "monitor": "total"
                # }
        }


    #-----------different training steps for sdf modulation, diffusion, combined----------

    def debug_shapes(self,**kwargs):
        """Prints shapes/types of all provided variables. Call this at the end of your function."""
        if False:
            print("\n=== Debug Shapes ===")
            for name, value in kwargs.items():
                shape = str(list(value.shape)) if hasattr(value, 'shape') else str(len(value)) if hasattr(value, '__len__') else 'scalar'
                dtype = str(value.dtype) if hasattr(value, 'dtype') else type(value).__name__
                print(f"{name.ljust(20)}: shape={shape.ljust(25)} type={dtype}")
            print("==================\n")

    def train_modulation_with_pointnet(self, x):
        xyz = x['xyz']  # (B, N, 3)
        gt = x['gt_sdf']  # (B, N)
        pc = x['point_cloud']  # (B, 1024, 3)
        
         # STEP 1: obtain reconstructed plane feature and latent code 
        points_features = self.sdf_model.pointnet.get_points_features(pc)
        original_features = torch.cat(points_features, dim=1)
        out = self.vae_model(original_features) # out = [self.decode(z), input, mu, log_var, z]
        reconstructed_points_feature, latent = out[0], out[-1]

        # STEP 2: pass recon back to GenSDF pipeline 
        pred_sdf = self.sdf_model.forward_with_points_features(reconstructed_points_feature, xyz)
        

        
        # Single debug call at the end
        self.debug_shapes(
            xyz=xyz,
            gt=gt,
            pc=pc,
            base_points=points_features,
            vae_output=out,
            reconstructed_base_point=reconstructed_points_feature,
            latent=latent,
            pred_sdf=pred_sdf
        )
        

        
        # STEP 3: losses for VAE and SDF
        # we only use the KL loss for the VAE; no reconstruction loss
        try:
            vae_loss = self.vae_model.loss_function(*out, M_N=self.specs["kld_weight"] )
        except:
            print("vae loss is nan at epoch {}...".format(self.current_epoch))
            return None # skips this batch

        sdf_loss = F.l1_loss(pred_sdf.squeeze(), gt.squeeze(), reduction='none')
        sdf_loss = reduce(sdf_loss, 'b ... -> b (...)', 'mean').mean()

        loss = sdf_loss 

        loss_dict =  {"sdf": sdf_loss, "vae": vae_loss}
        self.log_dict(loss_dict, prog_bar=True, enable_graph=False)

        return loss
    

    def train_modulation_base_points(self, x):
        xyz = x['xyz']  # (B, N, 3)
        gt = x['gt_sdf']  # (B, N)
        base_points = x['basis_point']  # (B, 1024, 3)
        
    
   

        
        out = self.vae_model(base_points)  # out = [self.decode(z), input, mu, log_var, z]
        reconstructed_base_point, latent = out[0], out[-1]
        


        pred_sdf = self.sdf_model.forward_with_base_features(reconstructed_base_point, xyz)
        """print("✅ pred_sdf info:")
        print("  Type:", type(pred_sdf))
        print("  Shape:", pred_sdf.shape)
        print("  Dtype:", pred_sdf.dtype)
        print("  Min:", pred_sdf.min().item())
        print("  Max:", pred_sdf.max().item())
        print("  Mean:", pred_sdf.mean().item())
        print("  Std:", pred_sdf.std().item())
        print("  Unique values:", torch.unique(pred_sdf).numel())
        print("  All values equal?", torch.all(pred_sdf == pred_sdf.view(-1)[0]).item())

        # Info gt_sdf
        print("✅ gt_sdf info:")
        print(f"  Type: {type(gt)}")
        print(f"  Shape: {gt.shape}")
        print(f"  Dtype: {gt.dtype}")
        print(f"  Min: {gt.min().item():.6f}")
        print(f"  Max: {gt.max().item():.6f}")
        print(f"  Mean: {gt.mean().item():.6f}")
        print(f"  Std: {gt.std().item():.6f}")
        print(f"  Unique values: {torch.unique(gt).numel()}")
        print(f"  All values equal? {torch.all(gt == gt.view(-1)[0]).item()}")

        # Single debug call at the end
        self.debug_shapes(
            xyz=xyz,
            gt=gt,
            base_points=base_points,
            vae_output=out,
            reconstructed_base_point=reconstructed_base_point,
            latent=latent,
            pred_sdf=pred_sdf
        )
        
       """
        # Mean for each sample in batch
        latent_mean_per_sample = latent.mean(dim=1)  # shape: (B,)
        pred_mean_per_sample = pred_sdf.mean(dim=1)  # shape: (B,)

        # Mean for each sample in batch
        latent_mean_per_sample = latent.mean(dim=1)  # shape: (B,)
        pred_mean_per_sample = pred_sdf.mean(dim=1)  # shape: (B,)

        
        print(f": Latent means = {latent_mean_per_sample.tolist()}")
        print(f": Pred means   = {pred_mean_per_sample.tolist()}")
                    
        # STEP 3: losses for VAE and SDF
        # we only use the KL loss for the VAE; no reconstruction loss
        try:
            vae_loss = self.vae_model.loss_function(*out, M_N=self.specs["kld_weight"] )
        except:
            print("vae loss is nan at epoch {}...".format(self.current_epoch))
            return None # skips this batch

        sdf_loss = F.l1_loss(pred_sdf.squeeze(), gt.squeeze(), reduction='none')
        sdf_loss = reduce(sdf_loss, 'b ... -> b (...)', 'mean').mean()

        loss = sdf_loss + vae_loss

        loss_dict =  {"sdf": sdf_loss, "vae": vae_loss}
        self.log_dict(loss_dict, prog_bar=True, enable_graph=False)

        return loss


    def train_diffusion(self, x):

        self.train()

        pc = x['point_cloud'] # (B, 1024, 3) or False if unconditional 
        latent = x['latent'] # (B, D)

        # unconditional training if cond is None 
        cond = pc if self.specs['diffusion_model_specs']['cond'] else None 

        # diff_100 and 1000 loss refers to the losses when t<100 and 100<t<1000, respectively 
        # typically diff_100 approaches 0 while diff_1000 can still be relatively high
        # visualizing loss curves can help with debugging if training is unstable
        diff_loss, diff_100_loss, diff_1000_loss, pred_latent, perturbed_pc = self.diffusion_model.diffusion_model_from_latent(latent, cond=cond)

        loss_dict =  {
                        "total": diff_loss,
                        "diff100": diff_100_loss, # note that this can appear as nan when the training batch does not have sampled timesteps < 100
                        "diff1000": diff_1000_loss
                    }
        self.log_dict(loss_dict, prog_bar=True, enable_graph=False)

        return diff_loss

    def _create_bps_grid(self, grid_size=32, radius=1.5):
        """Create a fixed BPS reference grid."""
        bps_grid_np = bps.generate_grid_basis(
            grid_size=grid_size,
            n_dims=3,
            minv=-radius,
            maxv=radius
        )
        return torch.from_numpy(bps_grid_np).float().to(self.device)

    def get_base_points(self, pointcloud: torch.Tensor) -> torch.Tensor:
        # pointcloud: (B, N, 3)
        batch_size = pointcloud.shape[0]

        pointcloud_np = pointcloud.detach().cpu().numpy()  # (B, N, 3)
        pc_normalized = bps.normalize(pointcloud_np)       # (B, N, 3)

        # Check bps_grid consistency
        current_grid = self.bps_grid.cpu().numpy()
        if not hasattr(self, '_first_bps_grid'):
            self._first_bps_grid = current_grid
            print("✅ Saved initial bps_grid for comparison.")
        else:
            grid_diff = np.linalg.norm(self._first_bps_grid - current_grid)
            if grid_diff > 1e-12:  # very tight threshold since this should be fixed
                print(f"❗ bps_grid changed! Norm diff: {grid_diff:.12f}")
            else:
                print("✅ bps_grid unchanged.")

        # encode with custom fixed grid basis
        x_bps = bps.encode(
            pc_normalized,
            bps_arrangement='custom',
            custom_basis=self.bps_grid.cpu().numpy(),
            bps_cell_type='deltas',
            n_jobs=1
        )  # (B, n_bps_points, 3)

        

        x_bps_tensor = torch.from_numpy(x_bps).to(pointcloud.device, dtype=pointcloud.dtype)

        # reshape if needed, e.g. (B, 32, 32, 32, 3)
        grid_size = int(round(self.bps_grid.shape[0] ** (1/3)))
        x_bps_tensor = x_bps_tensor.view(batch_size, grid_size, grid_size, grid_size, 3)

        return x_bps_tensor


    
    # the first half is the same as "train_sdf_modulation"
    # the reconstructed latent is used as input to the diffusion model, rather than loading latents from the dataloader as in "train_diffusion"
    def train_combined(self, x):
        xyz = x['xyz'] # (B, N, 3)
        gt = x['gt_sdf'] # (B, N)
        pc = x['point_cloud'] # (B, 1024, 3)

        # STEP 1: obtain reconstructed plane feature for SDF and latent code for diffusion
        plane_features = self.sdf_model.pointnet.get_plane_features(pc)
        original_features = torch.cat(plane_features, dim=1)
        #print("plane feat shape: ", feat.shape)
        out = self.vae_model(original_features) # out = [self.decode(z), input, mu, log_var, z]
        reconstructed_plane_feature, latent = out[0], out[-1] # [B, D*3, resolution, resolution], [B, D*3]

        # STEP 2: pass recon back to GenSDF pipeline 
        pred_sdf = self.sdf_model.forward_with_plane_features(reconstructed_plane_feature, xyz)
        
        # STEP 3: losses for VAE and SDF 
        try:
            vae_loss = self.vae_model.loss_function(*out, M_N=self.specs["kld_weight"] )
        except:
            print("vae loss is nan at epoch {}...".format(self.current_epoch))
            return None # skips this batch
        sdf_loss = F.l1_loss(pred_sdf.squeeze(), gt.squeeze(), reduction='none')
        sdf_loss = reduce(sdf_loss, 'b ... -> b (...)', 'mean').mean()

        # STEP 4: use latent as input to diffusion model
        cond = pc if self.specs['diffusion_model_specs']['cond'] else None
        diff_loss, diff_100_loss, diff_1000_loss, pred_latent, perturbed_pc = self.diffusion_model.diffusion_model_from_latent(latent, cond=cond)
        
        # STEP 5: use predicted / reconstructed latent to run SDF loss 
        generated_plane_feature = self.vae_model.decode(pred_latent)
        generated_sdf_pred = self.sdf_model.forward_with_plane_features(generated_plane_feature, xyz)
        generated_sdf_loss = F.l1_loss(generated_sdf_pred.squeeze(), gt.squeeze())

        # surface weight could prioritize points closer to surface but we did not notice better results when using it 
        #surface_weight = torch.exp(-50 * torch.abs(gt))
        #generated_sdf_loss = torch.mean( F.l1_loss(generated_sdf_pred, gt, reduction='none') * surface_weight )

        # we did not experiment with using constants/weights for each loss (VAE loss is weighted using value in specs file)
        # results could potentially improve with a grid search 
        loss = sdf_loss + vae_loss + diff_loss + generated_sdf_loss

        loss_dict =  {
                        "total": loss,
                        "sdf": sdf_loss,
                        "vae": vae_loss,
                        "diff": diff_loss,
                        # diff_100 and 1000 loss refers to the losses when t<100 and 100<t<1000, respectively 
                        # typically diff_100 approaches 0 while diff_1000 can still be relatively high
                        # visualizing loss curves can help with debugging if training is unstable
                        #"diff100": diff_100_loss, # note that this can sometimes appear as nan when the training batch does not have sampled timesteps < 100
                        #"diff1000": diff_1000_loss,
                        "gensdf": generated_sdf_loss,
                    }
        self.log_dict(loss_dict, prog_bar=True, enable_graph=False)

        return loss