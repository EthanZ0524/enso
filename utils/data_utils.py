'''
Simple Dataset classes to help with loading in CMIP and SODA (to be implemented) datasets.
'''

import xarray as xr
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class cmip(Dataset):
    def __init__(self, data_dir, label_dir):
        self.data = xr.open_dataset(data_dir)
        self.labels = xr.open_dataset(label_dir)

    def __len__(self):
        return len(self.data.year.values)
    
    def __getitem__(self, idx):
        # Returns a 36 x 24 x 72 x 4 data array and a list of 36 ground-truth ONIs. idx corresponds to the year entry
        temp = np.array([self.data[var].isel(year=idx).to_numpy() for var in list(self.data.data_vars)])
        x = np.transpose(temp, (1, 2, 3, 0))
        label = self.labels.isel(year=idx).nino.to_numpy()
        print(np.shape(label))

        return x, label