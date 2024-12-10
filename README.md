# GNN ENSO Modeling
Project Authors: Ethan Zhang, Yuchen Li, and Saahil Sundaresan. Forecasting ONI using GNNs.
The associated report can be found in the docs folder (https://github.com/EthanZ0524/enso/blob/train_test/docs/CS224w_Report.pdf).

--------------------

## Setup Instructions
### Data
Create a ./data folder in the home directory. The PyG DataLoader classes will handle all data downloading and processing in the data folder. Downloading and processing data can take around 30 minutes. Data will require around 25 GB of storage.

### Dependencies
Run the following to install the necessary packages.
```sh
pip install -r requirements.txt
```

### Usage
Run the following to train the GNNRNN.
```sh
python model_train.py [options]
```

| Flag/Option      | Description                                               | Default Value   |
|------------------|-----------------------------------------------------------------|-----------------|
| `--extend`       | If a training run is an extension of a previous run             | False           |
| `--finetune`     | If a training run is finetuning a previous model                | False           |
| `-f`             | Filepath to model checkpoint (required for extend and finetune) | None            |
| `--help`         | Show help message and exit                                      | -               |


Edit ./config.py to set model parameters accordingly. Model weights of the top model (lowest training losses) per epoch, in addition to the last model, will be saved into the automatically-created ./checkpoints directory.

### To-do's:
- Python-ize and train GNNRNN (DONE)
- Implement SODA and GODAS PyG Dataset classes (DONE)
- Clean up/sanity check CMIP PyG Dataset class (DONE)
- Write regridding code for general CMIP data (DONE)
- Implement multi-GPU training (Saahil)
- Feature engineering (position, etc.)
- Write evaluation/model comparison code (DONE)
- Create better adjacency matrices (Yuchen)

