import torch
import os
import numpy as np
import torch.utils.data 
from torch.nn import functional as F
from bps import bps
import pytorch_lightning as pl
from bps import bps
import trimesh
from skimage import measure
# add paths in model/__init__.py for new models
from models import * 
import os, json, numpy as np, torch
from torch.utils.data import DataLoader
from dataloader.sdf_loader import SdfLoader

class CombinedModel(pl.LightningModule):
    def __init__(self, specs):
        super().__init__()
        self.specs = specs
        self.counter = 0

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
            print("Structure of diffusion model ", self.diffusion_model)
 


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
        # ==== SAVE DEBUG CSVs ====
        if getattr(self, "counter", 0) % 50 == 0 :
            print("mean and std and min and max of out[2] and out[3]:")
            print("  Mean:", out[2].mean().item())
            print("  Std:", out[2].std().item())
            print("  Min:", out[2].min().item())
            print("  Max:", out[2].max().item())
            print("  Mean:", out[3].mean().item())
            print("  Std:", out[3].std().item())
            print("  Min:", out[3].min().item())
            print("  Max:", out[3].max().item())
            print("  Mean:", gt.mean().item())
            print("  Std:", gt.std().item())
            print("  Min:", gt.min().item())
            print("  Max:", gt.max().item())

        pred_sdf = self.sdf_model.forward_with_base_features(reconstructed_base_point, xyz)
 
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

        # ==== SAVE DEBUG CSVs ====
        if getattr(self, "counter", 0) % 1000 == 0:
            save_dir = f"visual{self.counter}"
            os.makedirs(save_dir, exist_ok=True)

            # Move to CPU and convert to numpy
            xyz_np = xyz.detach().cpu().numpy()
            gt_np = gt.detach().cpu().numpy()
            pred_np = pred_sdf.detach().cpu().numpy()

            # Take only the first batch for visualization
            xyz_np = xyz_np[0]
            gt_np = gt_np[0]
            pred_np = pred_np[0]

            # Take only the first batch for visualization
            xyz_vis = torch.from_numpy(xyz_np)
            gt_vis = torch.from_numpy(gt_np).unsqueeze(-1)
            pred_vis = torch.from_numpy(pred_np)

            # Save GT file: x,y,z,gt
            visual_data = torch.cat((xyz_vis, gt_vis), dim=1).cpu().numpy()
            visual_path = os.path.join(save_dir, "visual.csv")
            np.savetxt(visual_path, visual_data, delimiter=",", header="x,y,z,gt", comments="")
            print(f"Saved GT visualization to {visual_path}")

            # Save Prediction file: x,y,z,pred
            output_data = torch.cat((xyz_vis, pred_vis), dim=1).cpu().numpy()
            output_path = os.path.join(save_dir, "output.csv")
            np.savetxt(output_path, output_data, delimiter=",", header="x,y,z,pred", comments="")
            print(f"Saved prediction visualization to {output_path}")
            # Increment counter

            ## ==================== NEW LATENT-ONLY GENERATION ====================
            print("🔍 Sampling directly from latent space for VAE-only generation...")

            # 1. Sample a random latent from N(0,1)
            latent_dim = self.vae_model.latent_dim
            z_random = torch.randn(1, latent_dim, device=xyz.device) * self.specs.get("latent_std", 0.25)  # (1, latent_dim)

            grid_points = x["grid_point"][0].unsqueeze(0)  # (1, N, 3)
            with torch.no_grad():
                # 3. Predict SDF
                pred_sdf_rand = self.sdf_model.forward_with_base_features(z_random, grid_points)  # (1, N)
            print("Number of negative SDF values:", len(pred_sdf_rand[pred_sdf_rand<=0]))
            # --- SAVE CSV like before ---
            grid_points_cpu = grid_points.squeeze(0).detach().cpu()   # (N, 3)
            pred_sdf_cpu = pred_sdf_rand.squeeze(0).detach().cpu().unsqueeze(-1)  # (N, 1)

            # Stack together x,y,z,pred
            pred_sdf_cpu = pred_sdf_cpu.squeeze(-1)   # force (N, 1)
            latent_vis = torch.cat((grid_points_cpu, pred_sdf_cpu), dim=1).numpy()

            latent_csv_path = os.path.join(save_dir, "latent_output.csv")
            np.savetxt(latent_csv_path, latent_vis, delimiter=",", header="x,y,z,pred", comments="")
            print(f"Saved latent generation visualization to {latent_csv_path}")

            # --- INTERPOLATION ---

            # Extract mu and logvar from out[2]
            mu1 = out[2][0]        # shape [latent_dim]
            mu2 = out[2][1]    # shape [latent_dim]
            
            # Number of interpolation steps
            n_steps = 10  

            # Create a grid of latent vectors by interpolating between mu and logvar
            # Here we interpolate elementwise between out[2][0] and out[2][1]
            linspace = torch.linspace(0, 1, n_steps, device=mu1.device).unsqueeze(1)  # (n_steps, 1)
            interpolated_latents = mu1 * (1 - linspace) + mu2 * linspace  # (n_steps, latent_dim)
            # Reparametrize with std=1
            std = self.specs.get("latent_std", 0.25)
            logvar = torch.full_like(interpolated_latents, 2 * torch.log(torch.tensor(std)))
            latents = self.vae_model.reparameterize(interpolated_latents, logvar=logvar)
            grid_points_repeat = grid_points.repeat(latents.shape[0], 1, 1)  # (n_steps, N, 3)

            with torch.no_grad():  # evita di tenere grafo in memoria
                for i in range(latents.shape[0]):
                    latent_i = latents[i].unsqueeze(0)  # (1, latent_dim)
                    grid_points_i = grid_points  # (1, N, 3)

                    pred_grid = self.sdf_model.forward_with_base_features(latent_i, grid_points_i)  # (1, N)
                    print("Number of pred_grid negative SDF values:", (pred_grid <= 0).sum().item())
                    print(pred_grid.shape)

                    # Porta su CPU subito e libera la GPU
                    pred_np = pred_grid.detach().cpu().numpy().squeeze()
                    xyz_np = grid_points_repeat[i].detach().cpu().numpy()  # (N, 3)

                    # Concateno e salvo direttamente
                    output_data = np.concatenate([xyz_np, pred_np[:, None]], axis=1)  # (N, 4)
                    output_path = os.path.join(save_dir, f"interpolation{i}.csv")
                    np.savetxt(output_path, output_data, delimiter=",", header="x,y,z,pred", comments="")

                    print(f"Saved prediction visualization to {output_path}")

                    # pulizia memoria GPU
                    del pred_grid
                    torch.cuda.empty_cache()

        self.counter = getattr(self, "counter", 0) + 1

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

    
    # the first half is the same as "train_sdf_modulation"
    # the reconstructed latent is used as input to the diffusion model, rather than loading latents from the dataloader as in "train_diffusion"
    def train_combined(self, x):
        xyz = x['xyz']  # (B, N, 3)
        gt = x['gt_sdf']  # (B, N)
        base_points = x['basis_point']  # (B, 1024, 3)
        pc = x['point_cloud']  # (B, 1024, 3) or False if unconditional
        
        out = self.vae_model(base_points)  # out = [self.decode(z), input, mu, log_var, z]
        reconstructed_base_point, latent = out[0], out[-1]
        # ==== SAVE DEBUG CSVs ====
        if getattr(self, "counter", 0) == 1000:
            print("mean and std and min and max of out[2] and out[3]:")
            print("  Mean:", out[2].mean().item())
            print("  Std:", out[2].std().item())
            print("  Min:", out[2].min().item())
            print("  Max:", out[2].max().item())
            print("  Mean:", out[3].mean().item())
            print("  Std:", out[3].std().item())
            print("  Min:", out[3].min().item())
            print("  Max:", out[3].max().item())

        pred_sdf = self.sdf_model.forward_with_base_features(reconstructed_base_point, xyz)
        
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
        generated_sdf_pred = self.sdf_model.forward_with_base_features(generated_plane_feature, xyz)
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

        # ==== SAVE DEBUG CSVs ====
        if getattr(self, "counter", 0) % 200== 0:
            base_dir = f"visual{self.counter}"
            os.makedirs(base_dir, exist_ok=True)
            try:
                save_root = base_dir
                os.makedirs(save_root, exist_ok=True)

                # nomi comodi per versionare i file
                def make_stem(bidx: int) -> str:
                    # usa global_step se disponibile, altrimenti fallback a self.counter
                    gstep = int(getattr(self, "global_step", self.counter))
                    return f"ep{int(self.current_epoch):03d}_gs{gstep:06d}_b{bidx}"

                # assicurati che le robe siano sul CPU per salvare
                xyz_cpu   = xyz.detach().cpu()                         # [B, M, 3]
                sdf_cpu   = generated_sdf_pred.detach().cpu().squeeze(-1)  # [B, M]
                if 'perturbed_pc' in locals() and perturbed_pc is not None:
                    ppc_cpu = perturbed_pc.detach().cpu()              # [B, N, 3]
                else:
                    ppc_cpu = None

                B = xyz_cpu.shape[0]
                for b in range(B):
                    stem = make_stem(b)

                    # 1) salva la point cloud perturbata (se presente)
                    if ppc_cpu is not None and ppc_cpu.ndim == 3:
                        out_ppc = os.path.join(save_root, f"{stem}_perturbed_pc.csv")
                        np.savetxt(out_ppc, ppc_cpu[b].numpy(), delimiter=",")

                    # 2) salva la ricostruzione come point cloud
                    #    prendi i punti con |SDF| < tau; se zero match, prendi i più vicini allo zero
                    tau = 1e-2
                    sdf_b = sdf_cpu[b]          # [M]
                    xyz_b = xyz_cpu[b]          # [M, 3]
                    mask  = (sdf_b.abs() < tau)

                    if mask.any():
                        recon_pts = xyz_b[mask]
                    else:
                        M = sdf_b.numel()
                        k = max(1, min(10000, M // 10))  # top 10% fino a 10k punti
                        idx = torch.topk(-sdf_b.abs(), k).indices
                        recon_pts = xyz_b[idx]

                    out_recon = os.path.join(save_root, f"{stem}_recon.csv")
                    np.savetxt(out_recon, recon_pts.numpy(), delimiter=",")
                    print(f"Saved prediction visualization to {out_recon}")

            except Exception as e:
                print(f"[warn] failed to dump CSV point clouds: {e}")

            # === GENERAZIONE DA NORMALE + DIFFUSIONE CONDIZIONATA ALLA PC DI TRAIN ===
            try:
                # quante generazioni per ogni PC del batch
                K = int(self.specs.get("train_gen_per_pc", 3))
                use_ddim = bool(self.specs.get("train_ddim", False))

                # alias comodi
                device = latent.device
                B = pc.shape[0] if isinstance(pc, torch.Tensor) else 0

                for b in range(B):
                    if not isinstance(pc, torch.Tensor):
                        break  # serve una point cloud di condizionamento

                    stem = make_stem(b)
                    pc_single = pc[b:b+1]  # (1, N, 3)

                    # genera K latenti partendo da N(0,I) con diffusion condizionata a questa pc
                    samp_b, pert_pc_b = self.diffusion_model.generate_from_pc(
                        pc_single, batch=K, save_pc=None, return_pc=True, ddim=use_ddim, perturb_pc=True
                    )  # samp_b: (K, dim_latent)  |  pert_pc_b: (1, N, 3)

                    # salva la pc perturbata effettivamente usata (una volta per b)
                    if pert_pc_b is not None and pert_pc_b.ndim == 3:
                        out_ppc = os.path.join(save_root, f"{stem}_gennorm_perturbed_pc.csv")
                        np.savetxt(out_ppc, pert_pc_b.squeeze(0).detach().cpu().numpy(), delimiter=",")

                    # decodifica e valuta SDF sugli stessi xyz del batch di training
                    plane_feats_b = self.vae_model.decode(samp_b)                 # (K, feat_dim)
                    xyz_b   = xyz[b:b+1]                                          # (1, M, 3)
                    xyz_rep = xyz_b.repeat(plane_feats_b.shape[0], 1, 1)          # (K, M, 3)
                    gen_sdf_b = self.sdf_model.forward_with_base_features(plane_feats_b, xyz_rep)  # (K, M) o (K, M, 1)

                    # salva K ricostruzioni come point cloud (|SDF| < tau, fallback ai più vicini a 0)
                    tau = 1e-2
                    gen_sdf_cpu = gen_sdf_b.detach().cpu()
                    if gen_sdf_cpu.ndim == 3:
                        gen_sdf_cpu = gen_sdf_cpu.squeeze(-1)                     # (K, M)
                    xyz_cpu_b = xyz_rep.detach().cpu()                             # (K, M, 3)

                    for j in range(plane_feats_b.shape[0]):
                        sdf_j = gen_sdf_cpu[j]         # (M,)
                        xyz_j = xyz_cpu_b[j]           # (M, 3)
                        mask  = (sdf_j.abs() < tau)
                        if mask.any():
                            recon_pts = xyz_j[mask]
                        else:
                            Mpts = sdf_j.numel()
                            k = max(1, min(10000, Mpts // 10))
                            idx = torch.topk(-sdf_j.abs(), k).indices
                            recon_pts = xyz_j[idx]

                        out_recon = os.path.join(save_root, f"{stem}_gennorm{j}_recon.csv")
                        np.savetxt(out_recon, recon_pts.numpy(), delimiter=",")
                        print(f"[train] Saved cond-gen from Normal -> {out_recon}")

            except Exception as e:
                print(f"[warn][train] cond-gen from Normal failed: {e}")
            # === FINE GENERAZIONE CONDIZIONATA (TRAIN) ===


            # === SAME INPUT, NEW STOCHASTICITY (nuovi t e noise ad ogni run) ===
            try:

                # Usa ESATTAMENTE gli stessi input del primo passaggio:
                #   - latent  (uguale)
                #   - perturbed_pc  (uguale; se None, resta None)
                cond_same = perturbed_pc if ('perturbed_pc' in locals()) else None
                B = latent.shape[0]
                device = latent.device

                # Quante repliche vuoi salvare con stesso input
                num_repeats = 2  # produrrà _recon_sameB, _recon_sameC

                for rep in range(num_repeats):
                    # nuovo timestep per ogni sample del batch
                    t_rand = torch.randint(
                        low=0,
                        high=self.diffusion_model.num_timesteps,
                        size=(B,),
                        device=device
                    ).long()

                    # forward della diffusion con stesso latent e stessa cond,
                    # MA senza noise esplicito (verrà ricampionato ogni volta)
                    _, _, _, pred_latent_rep, _ = self.diffusion_model(
                        latent, t_rand, ret_pred_x=True, cond=cond_same
                    )
                    gen_feat_rep = self.vae_model.decode(pred_latent_rep)
                    gen_sdf_rep  = self.sdf_model.forward_with_base_features(gen_feat_rep, xyz)

                    # salva come point cloud vicino allo zero di SDF
                    xyz_cpu   = xyz.detach().cpu()                      # (B, M, 3)
                    sdf_cpu   = gen_sdf_rep.detach().cpu().squeeze(-1)  # (B, M)

                    for b in range(B):
                        stem = make_stem(b)
                        tag  = "sameB" if rep == 0 else "sameC"  # cambia suffissi se aumenti num_repeats
                        tau  = 1e-2
                        sdf_b = sdf_cpu[b]
                        xyz_b = xyz_cpu[b]
                        mask  = (sdf_b.abs() < tau)
                        if mask.any():
                            recon_pts = xyz_b[mask]
                        else:
                            M = sdf_b.numel()
                            k = max(1, min(10000, M // 10))
                            idx = torch.topk(-sdf_b.abs(), k).indices
                            recon_pts = xyz_b[idx]

                        out_recon = os.path.join(save_root, f"{stem}_recon_{tag}.csv")
                        np.savetxt(out_recon, recon_pts.numpy(), delimiter=",")
                        print(f"Saved prediction visualization to {out_recon}")

            except Exception as e:
                print(f"[warn] same-input (new t+noise) forward failed: {e}")

        self.counter = getattr(self, "counter", 0) + 1

        # ===== VALIDATION / TEST SNAPSHOT OGNI 10 STEP =====
        # esegue quando counter è multiplo di 10 (evita step 0)
        if getattr(self, "counter", 0) > 0 and (self.counter % 200) == 0:
            
            # se serve: from <tuo_modulo_dataset> import ModulationLoader

            was_training = self.training
            self.eval()
            device = next(self.parameters()).device

            # --- costruisci il dataloader una sola volta e riusalo ---
            if not hasattr(self, "_val_loader") or (self._val_loader is None):
                split = json.load(open(self.specs["validation_path"], "r"))
                self._val_dataset = SdfLoader(
                        self.specs["DataSource"],
                        split_file=split,
                        pc_size=self.specs.get("PCsize",1024), grid_source=self.specs.get("GridSource", None), modulation_path=self.specs.get("modulation_path", None)
                    )

                self._val_loader = DataLoader(
                    self._val_dataset,
                    batch_size=8,
                    num_workers=8,
                    shuffle=False, drop_last=False, pin_memory=True, persistent_workers=False
                )
            print("Len VALDATALOADER", len(self._val_loader))
            # dir di salvataggio
            base_dir = f"visual{self.counter}"
            os.makedirs(base_dir, exist_ok=True)
            save_root = base_dir

            with torch.no_grad():
                for vbi, vx in enumerate(self._val_loader):
                    # vx è un dict come nel training: 'xyz', 'gt_sdf', 'basis_point', 'point_cloud'
                    # -> manda tutto su device e in float
                    def to_dev(t):
                        return t.to(device).float() if torch.is_tensor(t) else t

                    xyz_v  = to_dev(vx['xyz'])            # (B, M, 3)
                    gt_v   = to_dev(vx['gt_sdf'])         # (B, M) o (B, M, 1)
                    base_v = to_dev(vx['basis_point'])    # (B, 1024, 3)
                    pc_v   = to_dev(vx['point_cloud']) if ('point_cloud' in vx and isinstance(vx['point_cloud'], torch.Tensor)) else None

                    # === pipeline IDENTICA al training ===
                    out_v = self.vae_model(base_v)                 # [decode(z), input, mu, log_var, z]
                    recon_base_v, latent_v = out_v[0], out_v[-1]

                    pred_sdf_v = self.sdf_model.forward_with_base_features(recon_base_v, xyz_v)

                    cond_v = pc_v if self.specs['diffusion_model_specs']['cond'] else None
                    diff_loss_v, _, _, pred_latent_v, perturbed_pc_v = self.diffusion_model.diffusion_model_from_latent(latent_v, cond=cond_v)

                    gen_plane_feat_v = self.vae_model.decode(pred_latent_v)
                    gen_sdf_pred_v   = self.sdf_model.forward_with_base_features(gen_plane_feat_v, xyz_v)

                    # === salvataggio CSV come nel training ===
                    # helper nome
                    def make_stem(bidx: int) -> str:
                        gstep = int(getattr(self, "global_step", self.counter))
                        return f"ep{int(self.current_epoch):03d}_gs{gstep:06d}_valb{vbi}_b{bidx}"

                    xyz_cpu   = xyz_v.detach().cpu()
                    sdf_cpu   = gen_sdf_pred_v.detach().cpu().squeeze(-1)  # [B, M]
                    ppc_cpu   = perturbed_pc_v.detach().cpu() if (perturbed_pc_v is not None) else None

                    B = xyz_cpu.shape[0]
                    for b in range(B):
                        stem = make_stem(b)

                        # salva perturbed pc, se presente
                        if (ppc_cpu is not None) and (ppc_cpu.ndim == 3):
                            out_ppc = os.path.join(save_root, f"{stem}_perturbed_pc.csv")
                            np.savetxt(out_ppc, ppc_cpu[b].numpy(), delimiter=",")

                        # salva "ricostruzione" come PC: punti con |SDF| < tau; fallback: più vicini allo zero
                        tau   = 1e-2
                        sdf_b = sdf_cpu[b]       # (M,)
                        xyz_b = xyz_cpu[b]       # (M,3)
                        mask  = (sdf_b.abs() < tau)
                        if mask.any():
                            recon_pts = xyz_b[mask]
                        else:
                            M = sdf_b.numel()
                            k = max(1, min(10000, M // 10))
                            idx = torch.topk(-sdf_b.abs(), k).indices
                            recon_pts = xyz_b[idx]

                        out_recon = os.path.join(save_root, f"{stem}_recon.csv")
                        np.savetxt(out_recon, recon_pts.numpy(), delimiter=",")
                        print(f"Saved prediction visualization to {out_recon}")

                    # === GENERAZIONE DA NORMALE + DIFFUSIONE CONDIZIONATA ALLA POINTCLOUD ===
                    # per ogni elemento del batch, genero K campioni partendo da N(0, I)
                    # e uso la diffusion condizionata a pc_v[b] per “denoisare” verso un latente plausibile
                    K = int(self.specs.get("val_gen_per_pc", 3))  # quanti campioni per point cloud
                    use_ddim = bool(self.specs.get("val_ddim", False))

                    for b in range(B):
                        if pc_v is None:
                            continue  # richiede condizionamento a una pointcloud

                        stem = make_stem(b)
                        pc_single = pc_v[b:b+1]  # (1, N, 3)

                        # usa l'helper già robusto del modello (fa anche la perturbazione coerente col training)
                        # NB: generate_from_pc lavora meglio con batch=1 per volta, così eviti mix tra batch
                        samp_b, pert_pc_b = self.diffusion_model.generate_from_pc(
                            pc_single, batch=K, save_pc=None, return_pc=True, ddim=use_ddim, perturb_pc=True
                        )  # samp_b: (K, dim_latent)  |  pert_pc_b: (1, N, 3)

                        # salva la point cloud perturbata usata per la condizione (una volta per b)
                        if pert_pc_b is not None and pert_pc_b.ndim == 3:
                            out_ppc = os.path.join(save_root, f"{stem}_gennorm_perturbed_pc.csv")
                            np.savetxt(out_ppc, pert_pc_b.squeeze(0).cpu().numpy(), delimiter=",")

                        # decodifica ogni latente generato e calcola SDF sugli stessi xyz del valid
                        plane_feats_b = self.vae_model.decode(samp_b)                # (K, feat_dim)
                        xyz_b = xyz_v[b:b+1]                                         # (1, M, 3)
                        xyz_rep = xyz_b.repeat(plane_feats_b.shape[0], 1, 1)         # (K, M, 3)
                        gen_sdf_b = self.sdf_model.forward_with_base_features(plane_feats_b, xyz_rep)  # (K, M, 1) o (K, M)

                        # salva K ricostruzioni come point cloud: |SDF| < tau (fallback ai più vicini allo zero)
                        tau = 1e-2
                        gen_sdf_cpu = gen_sdf_b.detach().cpu().squeeze(-1)  # (K, M)
                        xyz_cpu_b   = xyz_rep.detach().cpu()                # (K, M, 3)

                        for j in range(plane_feats_b.shape[0]):
                            sdf_j = gen_sdf_cpu[j]     # (M,)
                            xyz_j = xyz_cpu_b[j]       # (M, 3)
                            mask  = (sdf_j.abs() < tau)
                            if mask.any():
                                recon_pts = xyz_j[mask]
                            else:
                                Mpts = sdf_j.numel()
                                k = max(1, min(10000, Mpts // 10))
                                idx = torch.topk(-sdf_j.abs(), k).indices
                                recon_pts = xyz_j[idx]

                            out_recon = os.path.join(save_root, f"{stem}_gennorm{j}_recon.csv")
                            np.savetxt(out_recon, recon_pts.numpy(), delimiter=",")
                            print(f"Saved cond-gen from Normal to {out_recon}")
                    # === FINE GENERAZIONE CONDIZIONATA ===

            if was_training:
                self.train()
        # ===== FINE VALIDATION SNAPSHOT =====

        return loss