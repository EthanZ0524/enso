'''
Simple Dataset classes to help with loading in CMIP and SODA (to be implemented) datasets.
'''

import xarray as xr
import numpy as np
import torch
import subprocess
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader


REPO_ROOT = subprocess.check_output(['git', 'rev-parse', '--show-toplevel']).strip().decode('utf-8')
VAR_NAMES = ['sst', 't300', 'ua', 'va'] # hardcoded, sorry...

def construct_adjacency_list(method="grid"):
    """
    This generates the adjacency list for a 24x72 grid of nodes.

    Param:
        method (str): method to use for constructing adjacency list. 
            Options: "grid": trivial connection of each node to its 4 up-down-left-right
                             neighbors, if they exist

                     "mesh1": to be implemented

                     "mesh2": to be implemented

    Returns:
        np.array: adjacency list of shape (2, num_edges) 
    """

    rows, cols = 24, 72
    adjacency_list = []

    if method == "grid":    
        for i in range(rows):
            for j in range(cols):
                current_index = i * cols + j

                if i > 0: # row above
                    above_index = (i - 1) * cols + j
                    adjacency_list.append((current_index, above_index))

                if i < rows - 1: # row below
                    below_index = (i + 1) * cols + j
                    adjacency_list.append((current_index, below_index))

                if j > 0: # col left
                    left_index = i * cols + (j - 1)
                    adjacency_list.append((current_index, left_index))

                if j < cols - 1: # col right
                    right_index = i * cols + (j + 1)
                    adjacency_list.append((current_index, right_index))

    adj_t = np.array(adjacency_list).T
    return adj_t

'''
Jankiest thing ever

Original NetCDF data: 4645 (years) x 36 (months) x 24 (lat) x 72 (long) x 4 (features)

The dataset contains (36 x 4645) entries. Eg. cmip.get(x) gives you a graph from month (x%36), year (x//36).
Each entry is a (72 x 24) x 4 matrix. Eg. cmip.get(x)[y] gives you latitude index(y//72), long index (y%72)
'''
class cmip(Dataset):
    def __init__(self, root, transform=None):
        super(cmip, self).__init__(root, transform)

    @property
    def raw_file_names(self):
        return ['CMIP_train.nc', 'CMIP_label.nc']
    
    @property
    def processed_file_names(self):
        return 'aaa'
    
    def download(self):
        pass

    def process(self):
        pass

    def len(self):
        return len(xr.open_dataset(f'{REPO_ROOT}/data/raw/CMIP_train.nc').year.values) * len(xr.open_dataset(f'{REPO_ROOT}/data/raw/CMIP_train.nc').month.values)
    
    def get(self, idx, adjacency_method="grid"):
        x_unprocessed = xr.open_dataset(f'{REPO_ROOT}/data/raw/CMIP_train.nc')
        year = idx // 36
        month = idx % 36

        labels = torch.from_numpy(xr.open_dataset(f'{REPO_ROOT}/data/raw/CMIP_label.nc')['nino'].to_numpy()[year])
        temp = np.array([x_unprocessed[var].isel(year=year, month=month).to_numpy() for var in VAR_NAMES]) # 4 (var) x 24 (lat) x 72 (lon)
        x = np.transpose(temp, (1, 2, 0)).reshape(-1, 4) # 4 x (24 x 72). Caution: stacks 72's on top of each other, not 24's!

        # Creating graph from x:
        adj_t = construct_adjacency_list(method=adjacency_method)
        g = Data(x=torch.from_numpy(x), edge_index=torch.from_numpy(adj_t))

        return (g, labels)

# from torch.utils.data import Dataset, DataLoader

# class cmip(Dataset):
#     def __init__(self, data_dir, label_dir):
#         self.data = xr.open_dataset(data_dir)
#         self.labels = xr.open_dataset(label_dir)

#     def __len__(self):
#         return len(self.data.year.values)
    
#     def __getitem__(self, idx):
#         # Returns a 36 x 24 x 72 x 4 data array and a list of 36 ground-truth ONIs. idx corresponds to the year entry
#         temp = np.array([self.data[var].isel(year=idx).to_numpy() for var in list(self.data.data_vars)])
#         x = np.transpose(temp, (1, 2, 3, 0))
#         label = self.labels.isel(year=idx).nino.to_numpy()
#         print(np.shape(label))

#         return x, label
