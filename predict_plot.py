experiment_path = 'checkpoints/GATLSTM_newdata_multimesh_20241209_232544'
checkpoint_name = 'epoch=3-train_loss=0.54.ckpt'

import numpy as np
import xarray as xr
import pandas as pd 
from scipy.stats import pearsonr
import os
from os import path as osp
from tqdm import tqdm

import torch
import torch_geometric
from torch_geometric.data import Batch

import matplotlib.pyplot as plt 
import matplotlib.cm as cm

from models.master_model import MasterModel
from utils.data_utils import GODAS
from torch_geometric.data import Batch
from data_retrieval.data_config import DATA_DIR


dataset = GODAS(adjacency_method='multimesh1')
num_graphs = len(dataset)
# loading label timeseries

godas_ds = xr.open_dataset(osp.join(DATA_DIR, 'GODAS/raw/GODAS.nc'))

labels = godas_ds.oni.to_numpy() 
dates = godas_ds.time

#pd.Series(xr.open_dataset(godas_path)['time'].to_numpy()).dt.strftime('%Y-%m').to_numpy()
dates = dates[36:-1]
# currently month 1 as first elem - we want month 37
labels = labels[36:-1]


if not os.path.exists(osp.join(experiment_path, f'{checkpoint_name}_GODAS_predictions.pt')):
    device = torch.device('cuda')
    print(f'Using {device}')

    predictor = MasterModel.load_from_checkpoint(osp.join(experiment_path, checkpoint_name), map_location=device, num_heads=1)

    predictor.to(device)  
    predictor.eval()

    graphs = [0] + [dataset.get(i).to(device) for i in range(35)]
    predictions = []

    for i in tqdm(range(num_graphs - 36 + 1)): 
        graphs = graphs[1:]
        graphs.append(dataset.get(i+35).to(device))
        batch = Batch.from_data_list(graphs).to(device)
        with torch.no_grad():
            predictions.append(predictor(batch))

    predictions = torch.stack(predictions, dim=0)
    torch.save(predictions, osp.join(experiment_path, f'{checkpoint_name}_GODAS_predictions.pt'))

predictions = torch.load(osp.join(experiment_path, f'{checkpoint_name}_GODAS_predictions.pt'))
predictions = predictions.reshape(predictions.size(0), -1).cpu()

leads_to_plot = {
    0: 1,
    1: 3, 
    2: 6,
    3: 12
}

fig, axs = plt.subplots(figsize=(8,6), nrows=4, ncols=1, sharex=True, sharey=True)

for i, ax in enumerate(axs.flatten()):
    # plot ground truth 
    ax.plot(dates, labels, linewidth=1, color='black')

    # plot lead 
    lead = leads_to_plot[i]
    lead_prediction = predictions[:, lead].numpy()
    truth = labels[lead:lead+len(lead_prediction)]
    dates_adjusted = dates[lead:lead+len(lead_prediction)]
    rmse = np.sqrt(np.mean((truth - lead_prediction) ** 2))
    r, _ = pearsonr(truth, lead_prediction)

    ax.plot(dates_adjusted, lead_prediction, linewidth=1, color='tab:red')
    ax.set_title(rf"Lead {lead}: $r$ = {r:.2f}, RMSE = {rmse:.2f}")
    ax.set_ylabel("ONI (˚C)")
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')

fig.tight_layout()

plt.savefig("figures/GATLSTM_5models_multimesh_predictions.jpg", dpi=300)