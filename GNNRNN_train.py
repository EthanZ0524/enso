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

from utils.data_utils import MasterDataModule
from config import *
from models.GNNRNN import GNNRNN

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

    experiment_name = EXPERIMENT_NAME
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Setting up training directories
    checkpoint_dir, logs_dir = setup_training_dirs(experiment_name, timestamp=timestamp)

    logger = WandbLogger(
        save_dir=str(logs_dir),
        name=experiment_name
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath=str(checkpoint_dir),
        filename='{epoch}-{train_loss:.2f}',
        save_top_k=-1,  # saving top model of every epoch
        monitor='train_loss',
        mode='min',
        save_last=True, # save last model
    )

    model = GNNRNN(graph_emb_dim=GCN_EMB_DIM, 
        enc_hidden_dim=ENC_HIDDEN_DIM, 
        output_length=NUM_OUTPUT_MONTHS,
        gcn_layers=GCN_NUM_LAYERS,
        gcn_dropout=GCN_DROPOUT,
        lr=LEARNING_RATE)

    data_module = MasterDataModule(batch_size=NUM_INPUT_MONTHS *  BATCH_SIZE)

    trainer = pl.Trainer(
        max_epochs=EPOCHS,
        callbacks=[checkpoint_callback],
        logger=logger,
        default_root_dir=str(checkpoint_dir),
        accelerator='auto', 
        devices=1,
        log_every_n_steps=1
    )

    trainer.fit(model, datamodule=data_module)

if __name__ == "__main__":
    main()