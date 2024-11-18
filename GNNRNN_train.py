import numpy as np
import os
import torch
import torch.nn as nn
import torch_geometric
import torch_geometric.nn
import pytorch_lightning as pl
import wandb
from torch_geometric.nn import GCNConv, global_add_pool, global_mean_pool
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import RandomSampler
from datetime import datetime
import gc
import argparse
import sys

from utils.data_utils import MasterDataModule
from config import *
from models.GNNRNN import GNNRNN

def parse_args():
    parser = argparse.ArgumentParser(description="Script to train GNNRNN model")
    parser.add_argument('--extend', action='store_true', help="Extending a model's training")
    parser.add_argument('-f', type=str, help="Path to checkpoint file", required=False)
    parser.add_argument('-e', type=int, help="New number of epochs to train model for", required=False)
    args = parser.parse_args()

    if args.extend:
        if args.f is None:
            raise ValueError("Error: checkpoint required for extending training")
    if args.e and not args.extend:
        raise ValueError("Error: epochs provided for non-extension training run")
    
    return args

def setup_training_dirs(experiment_name: str = None, root_dir: str = "./", timestamp: str = None):
    # Create directories if they don't exist
    checkpoints_dir = root_dir + "checkpoints"
    logs_dir = root_dir + "logs"

    for dir_path in [checkpoints_dir, logs_dir]:
        if not os.path.exists(dir_path):
            os.makedirs(dir_path, exist_ok=True)

    if experiment_name:
        run_name = f"{experiment_name}_{timestamp}"
    else:
        run_name = timestamp

    run_dir = checkpoints_dir + "/" + run_name
    os.makedirs(run_dir, exist_ok=True)

    return run_dir, logs_dir

def main():
    gc.collect()
    torch.cuda.empty_cache()

    args = parse_args()
    extend = args.extend
    epochs = EPOCHS if not extend else args.e
    checkpoint_path = None if not extend else args.f
    checkpoint_dir = None if not checkpoint_path else os.path.dirname(checkpoint_path) 
    experiment_name = EXPERIMENT_NAME if not extend else f'Extending_to_{epochs}_epochs'

    if not extend:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # Setting up training directories
        checkpoint_dir, logs_dir = setup_training_dirs(experiment_name, timestamp=timestamp)

    checkpoint_callback = ModelCheckpoint(
        dirpath=str(checkpoint_dir),
        filename='{epoch}-{train_loss:.2f}',
        save_top_k=-1,  # saving top model of every epoch
        monitor='train_loss',
        mode='min',
        save_last=True, # save last model
    )
    
    logger = WandbLogger(
            # save_dir=str(logs_dir),
            save_dir='./logs', # bad
            name=experiment_name
        )

    model = GNNRNN(graph_emb_dim=GCN_EMB_DIM, 
            enc_hidden_dim=ENC_HIDDEN_DIM, 
            output_length=NUM_OUTPUT_MONTHS,
            gcn_layers=GCN_NUM_LAYERS,
            gcn_dropout=GCN_DROPOUT,
            lr=LEARNING_RATE)

    data_module = MasterDataModule(batch_size=NUM_INPUT_MONTHS *  BATCH_SIZE)

    trainer = pl.Trainer(
        max_epochs=epochs,
        callbacks=[checkpoint_callback],
        logger=logger,
        default_root_dir=str(checkpoint_dir),
        accelerator='auto', 
        devices=1,
        log_every_n_steps=1
    )

    trainer.fit(model, datamodule=data_module, ckpt_path=checkpoint_path)


if __name__ == "__main__":
    main()