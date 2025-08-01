#!/usr/bin/env python3

import torch
import torch.utils.data 
from torch.nn import functional as F
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, Callback, LearningRateMonitor
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.loggers import TensorBoardLogger

import os
import json, csv
import time
from tqdm.auto import tqdm
from einops import rearrange, reduce
import numpy as np
import trimesh
import warnings

# add paths in model/__init__.py for new models
from models import * 
from utils import mesh, evaluate
from utils.reconstruct import *
from diff_utils.helpers import * 
#from metrics.evaluation_metrics import *#compute_all_metrics
#from metrics import evaluation_metrics

from dataloader.pc_loader import PCloader
from dataloader.sdf_loader import SdfLoader
from dataloader.modulation_loader import ModulationLoader


def train():
    
    # initialize dataset and loader
    split = json.load(open(specs["TrainSplit"], "r"))
    if specs['training_task'] == 'diffusion':
        train_dataset = ModulationLoader(specs["data_path"], pc_path=specs.get("pc_path",None), split_file=split, pc_size=specs.get("total_pc_size", None))
    else:
        train_dataset = SdfLoader(specs["DataSource"], split, pc_size=specs.get("PCsize",1024), grid_source=specs.get("GridSource", None), modulation_path=specs.get("modulation_path", None))
    train_dataloader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=args.batch_size, num_workers=args.workers,
            drop_last=False, shuffle=True, pin_memory=True, persistent_workers=True
        )
    print(f"Length Data: {len(train_dataset)}")  # Should be > 0
    # creates a copy of current code / files in the config folder
    save_code_to_conf(args.exp_dir) 
    
    # pytorch lightning callbacks 
    callback = ModelCheckpoint(dirpath=args.exp_dir, filename='{epoch}', save_top_k=-1, save_last=True, every_n_epochs=specs["log_freq"])
    lr_monitor = LearningRateMonitor(logging_interval='step')
    callbacks = [callback, lr_monitor]

    model = CombinedModel(specs)

    print(f"Model structure: {model}")  # Simple but effective overview

    # note on loading from checkpoint:
    # if resuming from training modulation, diffusion, or end-to-end, just load saved checkpoint 
    # however, if fine-tuning end-to-end after training modulation and diffusion separately, will need to load sdf and diffusion checkpoints separately
    if args.resume == 'finetune':
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = model.load_from_checkpoint(specs["modulation_ckpt_path"], specs=specs, strict=False)
            # loads the diffusion model; directly calling diffusion_model.load_state_dict to prevent overwriting sdf and vae params
            ckpt = torch.load(specs["diffusion_ckpt_path"])
            new_state_dict = {}
            for k,v in ckpt['state_dict'].items():
                new_key = k.replace("diffusion_model.", "") # remove "diffusion_model." from keys since directly loading into diffusion model
                new_state_dict[new_key] = v
            model.diffusion_model.load_state_dict(new_state_dict)
        resume = None
    elif args.resume is not None:
        ckpt = "{}.ckpt".format(args.resume) if args.resume=='last' else "epoch={}.ckpt".format(args.resume)
        resume = os.path.join(args.exp_dir, ckpt)
    else:
        resume = None  

    tb_logger = TensorBoardLogger(
        save_dir=os.path.join("tensorboard_logs", args.exp_dir),
        name="logs"
    )

    trainer = pl.Trainer(
        accelerator='gpu',
        devices=-1,
        precision=32,
        max_epochs=specs["num_epochs"],
        callbacks=callbacks,
        log_every_n_steps=1,
        logger=tb_logger,
        default_root_dir=os.path.join("tensorboard_logs", args.exp_dir)
    )
    trainer.fit(model=model, train_dataloaders=train_dataloader, ckpt_path=resume)

    

    
if __name__ == "__main__":

    import argparse

    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument(
        "--exp_dir", "-e", required=True,
        help="This directory should include experiment specifications in 'specs.json,' and logging will be done in this directory as well.",
    )
    arg_parser.add_argument(
        "--resume", "-r", default=None,
        help="continue from previous saved logs, integer value, 'last', or 'finetune'",
    )

    arg_parser.add_argument("--batch_size", "-b", default=32, type=int)
    arg_parser.add_argument( "--workers", "-w", default=8, type=int)

    args = arg_parser.parse_args()
    specs = json.load(open(os.path.join(args.exp_dir, "specs.json")))
    print(specs["Description"])


    train()