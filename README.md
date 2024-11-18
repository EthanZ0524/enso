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
python GNNRNN_train.py
```
To extend training starting from a given checkpoint, run the following:
```sh
python GNNRNN_train.py --extend -e {new num epochs} -f {path to checkpoint}
```

Edit ./config.py to set model parameters accordingly. Model weights of the top model (lowest training losses) per epoch, in addition to the last model, will be saved into the automatically-created ./checkpoints directory.

### To-do's:
- Python-ize and train GNNRNN (DONE)
- Implement SODA and GODAS PyG Dataset classes (SODA DONE)
- Clean up/sanity check CMIP PyG Dataset class (DONE)
- Write regridding code for general CMIP data (Yuchen)
- Implement multi-GPU training (Saahil)
- Feature engineering (position, etc.)
- Write evaluation/model comparison code (kind of done)
- Implement validation dataloader/loss stuff (probably need a LightningDataModule implementation)
- Create better adjacency matrices (Yuchen)

