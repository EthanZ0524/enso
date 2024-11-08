import numpy as np
import os
import torch
import torch.nn as nn
import torch_geometric
import torch_geometric.nn
import lightning as L
import wandb
from torch_geometric.nn import GCNConv, global_add_pool, global_mean_pool
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from lightning.pytorch.loggers import TensorBoardLogger, WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader, RandomSampler
from datetime import datetime

from utils.data_utils import cmip
from utils.data_utils import ShuffledBatchSampler
import config
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
    if torch.cuda.is_available():
        device = torch.device("cuda:0")  # Using the first CUDA device
        print('Using GPU')
    else:
        device = torch.device("cpu")
        print('Using CPU')

    experiment_name = "GNNRNN_5epochs_1e-5"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Setup the training directories
    checkpoint_dir, logs_dir = setup_training_dirs(experiment_name, timestamp=timestamp)

    logger = WandbLogger(
        save_dir=str(logs_dir),
        name=experiment_name
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath=str(checkpoint_dir),
        filename='{epoch}-{train_loss:.2f}',
        save_top_k=3,  # Save the top 3 models
        monitor='train_loss',
        mode='min',
        save_last=True,  # Additionally save the last model
    )

    model = GNNRNN(graph_emb_dim=32, hidden_dim=32, output_length=32, device=device).to(device)
    data = cmip(root=None)

    batch_size = config.NUM_PREDICTION_MONTHS * 16
    sampler = RandomSampler(data)
    batch_sampler = ShuffledBatchSampler(sampler, batch_size=batch_size, drop_last=False)
    loader = DataLoader(data, batch_sampler=batch_sampler)

    # Create the trainer
    trainer = L.Trainer(
        max_epochs=5,
        callbacks=[checkpoint_callback],
        logger=logger,
        default_root_dir=str(checkpoint_dir),
        log_every_n_steps=1
    )

    trainer.fit(model, train_dataloaders=loader)

if __name__ == "__main__":
    main()