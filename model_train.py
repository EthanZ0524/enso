import os
import torch
import pytorch_lightning as pl
import wandb
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from datetime import datetime
import gc
import argparse
import sys
import warnings
import importlib.util

warnings.filterwarnings("ignore", category=FutureWarning, message=".*torch.load.*weights_only=False.*")

from utils.data_utils import MasterDataModule
from global_vars import *
from models.master_model import MasterModel

def parse_args():
    parser = argparse.ArgumentParser(description="Script to train models")
    parser.add_argument('--extend', action='store_true', help="Extending/continuing a model's training")
    parser.add_argument('--finetune', action='store_true', help="Finetune a pretrained model")
    parser.add_argument('-f', type=str, help="Path to checkpoint file", required=False)
    parser.add_argument('-e', type=int, help="New number of epochs to train model for", required=False)
    parser.add_argument('--config', type=str, required=True, help="Path to the config.py file to use for training")

    args = parser.parse_args()

    if args.extend or args.finetune:
        if args.f is None:
            raise ValueError("Error: checkpoint required for extending training/finetuning")
    if args.e and not args.extend:
        raise ValueError("Error: epochs provided for non-extension training run")
    
    return args

def load_config_into_globals(config_path, global_vars):
    """Load the config module and inject its variables into the global namespace."""
    spec = importlib.util.spec_from_file_location("config", config_path)
    config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config)

    # Inject variables from config into the provided global_vars dictionary
    for attr in dir(config):
        if not attr.startswith("__"):  # Ignore special/private attributes
            global_vars[attr] = getattr(config, attr)


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
    finetune = args.finetune            
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Load the specified config into globals
    globals_dict = globals()
    load_config_into_globals(args.config, globals_dict)

    if extend:
        epochs = args.e
        checkpoint_path = args.f
        checkpoint_dir = os.path.dirname(checkpoint_path) # NEED TO FIX: this should also be the case when BOTH ft and extend
        learning_rate = LEARNING_RATE
        batch_size = BATCH_SIZE
        experiment_name = None
    elif finetune:
        epochs = FT_EPOCHS
        checkpoint_path = args.f
        experiment_name = f'{os.path.basename(os.path.dirname(checkpoint_path))}_finetune_{FT_EXPERIMENT_NAME}'
        learning_rate = FT_LEARNING_RATE
        batch_size = FT_BATCH_SIZE
    else: 
        epochs = EPOCHS
        experiment_name = EXPERIMENT_NAME
        learning_rate = LEARNING_RATE
        batch_size = BATCH_SIZE

    if not extend:
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
        save_dir='./logs',
        name=experiment_name
    )

    if finetune:
        model = MasterModel.load_from_checkpoint(checkpoint_path) # this loads weights but the training starts fresh.

    else:
        model = MasterModel(graph_emb_dim=GENET_EMB_DIM, 
                ge_net_layers=GENET_NUM_LAYERS,
                ge_net_dropout=GENET_DROPOUT,
                node_embedder=NODE_EMBEDDER,
                enc_hidden_dim=ENC_HIDDEN_DIM, 
                enc_dec=ENC_DEC,
                output_length=NUM_OUTPUT_MONTHS,
                lr=learning_rate
            )

    data_module = MasterDataModule(batch_size=NUM_INPUT_MONTHS *  batch_size, 
        finetune=finetune, 
        adjacency=ADJACENCY,
        num_cmip_models=NUM_MODELS
    )

    trainer = pl.Trainer(
        max_epochs=epochs,
        callbacks=[checkpoint_callback],
        logger=logger,
        default_root_dir=str(checkpoint_dir),
        accelerator='auto', 
        log_every_n_steps=1
    )

    if extend:
        trainer.fit(model, datamodule=data_module, ckpt_path=checkpoint_path) # for extending training (keeping old epoch #, etc.)

    else:
        trainer.fit(model, datamodule=data_module)


if __name__ == "__main__":
    main()