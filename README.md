# GNN ENSO Modeling
Final CS224w project authored by Ethan Zhang, Yuchen Li, and Saahil Sundaresan. Forecasting ONI using GNNs.

--------------------

## Setup Instructions
### Data
Create a ./data folder in the home directory. Then, unzip the enso_round1_train file found at [this](https://tianchi.aliyun.com/dataset/98942) link into the data folder.

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

Model weights of the top 3 models (lowest training losses) will be saved into the automatically-created ./checkpoints directory.

### To-do's:
~~- Python-ize and train GNNRNN (Ethan)~~
- Implement SODA and GODAS PyG Dataset classes
- Clean up/sanity check CMIP PyG Dataset class
- Write regridding code for general CMIP data (Yuchen)
- Implement multi-GPU training
- Feature engineering (position, etc.)
- Write evaluation/model comparison code
- Implement validation dataloader/loss stuff (probably need a LightningDataModule implementation)
- Probably some other important things I am forgetting yay

