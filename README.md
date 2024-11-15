# GNN ENSO Modeling
Project Authors: Ethan Zhang, Yuchen Li, and Saahil Sundaresan. Forecasting ONI using GNNs.

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
python GNNRNN_train.py
```
Edit ./config.py to set model parameters accordingly. Model weights of the top model (lowest training losses) per epoch, in addition to the last model, will be saved into the automatically-created ./checkpoints directory.

### To-do's:
~~- Python-ize and train GNNRNN~~
- Implement ~~SODA and~~ GODAS PyG Dataset classes
~~- Clean up/sanity check CMIP PyG Dataset class~~
- Write regridding code for general CMIP data (Yuchen)
- Implement multi-GPU training
- Feature engineering (position, etc.)
- Write evaluation/model comparison code
- Implement validation dataloader/loss stuff (probably need a LightningDataModule implementation)
- Create better adjacency matrices (Yuchen)

