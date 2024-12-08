'''
Simple Dataset classes to help with loading in CMIP and SODA datasets.

For more extensive documentation, please see enso/docs/data_utils.md
'''

import xarray as xr
import numpy as np
import torch
import subprocess
from torch_geometric.data import Data, Dataset, download_url
from torch_geometric.loader import DataLoader
import os
from os import path as osp
from torch.utils.data import Sampler
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
import pytorch_lightning as pl

DATA_ROOT = subprocess.check_output(['git', 'rev-parse', '--show-toplevel']).strip().decode('utf-8') + '/data'
VAR_NAMES = ['sst', 't300', 'ua', 'va'] # hardcoded, sorry...

def get_allocated_cpus():
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:
        return 0

NUM_WORKERS = min(get_allocated_cpus() - 1, 0)

def construct_adjacency_list(method="grid"):
    """
    Helper function: generates the adjacency list for a 24x72 grid of nodes.

    Params:
        method (str): method to use for constructing adjacency list. The following options are available:
            "grid": trivial connection of each node to its 4 up-down-left-right neighbors, if they exist
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
                # Row above
                if i > 0: 
                    above_index = (i - 1) * cols + j
                    adjacency_list.append((current_index, above_index))

                # Row below
                if i < rows - 1: 
                    below_index = (i + 1) * cols + j
                    adjacency_list.append((current_index, below_index))

                # Column left (wrap around for periodic boundary)
                left_index = i * cols + (j - 1) % cols
                adjacency_list.append((current_index, left_index))

                # Column right (wrap around for periodic boundary)
                right_index = i * cols + (j + 1) % cols
                adjacency_list.append((current_index, right_index))

    elif method == "dense_grid":
        for i in range(rows):
            for j in range(cols):
                current_index = i * cols + j

                # Loop over all neighbors (including diagonals)
                for di in [-1, 0, 1]:
                    for dj in [-1, 0, 1]:
                        if di == 0 and dj == 0:
                            continue  # Skip the current node

                        neighbor_row = i + di
                        neighbor_col = (j + dj) % cols  # Periodic in east-west direction

                        # Check if neighbor is within bounds in the north-south direction
                        if 0 <= neighbor_row < rows:
                            neighbor_index = neighbor_row * cols + neighbor_col
                            adjacency_list.append((current_index, neighbor_index))


    adj_t = np.array(adjacency_list).T
    return adj_t


class CMIP(Dataset):
    def __init__(self, name='CMIP', root=DATA_ROOT, transform=None):
        self.name = name
        self.root = root
        self.urls = ['https://www.dropbox.com/scl/fi/ydthqdfbhibc9hbbelpzj/CMIP_train.nc?rlkey=92kphas2yj5zmnfnb4kffzwtf&st=hg9n0vwd&dl=1', 
                    'https://www.dropbox.com/scl/fi/gf5riugm3atj47rf3sbvi/CMIP_label.nc?rlkey=u4mtaeel1lpr8vho481l5de7p&st=hluf1xgo&dl=1']   
        self.length = None
        self.labels = None
        self.years = []
        for i in range(2265):
            if i % 151 != 149 and i % 151 != 150:
                self.years.append(i)
        for i in range(2265, 4645):
            if  (i-2265) % 140 != 138 and (i-2265) % 140 != 139:
                self.years.append(i)

        super().__init__(root, transform)
        
    @property
    def raw_dir(self) -> str:
        return osp.join(self.root, self.name, 'raw')

    def len(self):
        if self.length is None:
            months = len(xr.open_dataset(f'{self.raw_dir}/CMIP_train.nc', engine="netcdf4").month.values)
            self.length = len(self.years) * months
            # 32 models: 15 CMIP6, 17 CMIP5. 2 simulations (years) need to be discarded per model b/c of 2-year lead time
        return self.length
    
    def get_labels(self):
        if self.labels is None:
            labels_unprocessed = xr.open_dataset(f'{self.raw_dir}/CMIP_label.nc', engine="netcdf4")['nino'].to_numpy() # 4464 x 36

            cmip6 = labels_unprocessed[:2265]
            cmip5 = labels_unprocessed[2265:]

            # see https://tianchi.aliyun.com/dataset/98942
            cmip6 = np.reshape(cmip6, (15, 151, 36))
            cmip5 = np.reshape(cmip5, (17, 140, 36))

            final = []

            for model in cmip6:
                shifted = np.roll(model, shift=-2, axis=0)[:, 11:11+24] # up to 2 years' lead time is typically tested. Thus, we need to shift by 2 years
                final.append(shifted[:-2]) # discard last two 

            for model in cmip5:
                shifted = np.roll(model, shift=-2, axis=0)[:, 11:11+24] # up to 2 years' lead time is typically tested. Thus, we need to shift by 2 years
                final.append(shifted[:-2]) # discard last two 

            self.labels = np.vstack(final) 
            
        return self.labels

    @property
    def processed_dir(self) -> str:
        return osp.join(self.root, self.name, 'processed')

    @property
    def raw_file_names(self):
        return ['CMIP_train.nc', 'CMIP_label.nc']
    
    @property
    def processed_file_names(self):
        # cmip6 start indices: 0, 151, 302... cmip5: 2265, 2405, 2545...
        # need to remove last two years (eg. 149, 150, 300, 301, 2403, 2404, etc.)
        files = []
        for year in self.years:
            for month in range(36):
                files.append(f'{year}_{month}.pt')
        
        return files

    def download(self) -> None:
        for filename, url in zip(self.raw_file_names, self.urls):
            download_url(url, f'{self.raw_dir}')

    def process(self, adjacency_method='grid'):
        idx = 0
        x_unprocessed = xr.open_dataset(f'{self.raw_dir}/CMIP_train.nc', engine="netcdf4")
        adj_t = construct_adjacency_list(method=adjacency_method)
        
        for year in self.years:
            for month in range(36):
                temp = np.stack([x_unprocessed[var].isel(year=year, month=month).to_numpy() for var in VAR_NAMES], axis=0) # 4 (var) x 24 (lat) x 72 (lon)
                x = temp.transpose(1, 2, 0).reshape(-1, 4) # (24 x 72) x 4. Caution: stacks 72's on top of each other, not 24's!

                # Removing NaNs (terrestial nodes)
                valid_nodes = ~np.isnan(x).any(axis=1) # (24x72) bool array
                filtered_x = x[valid_nodes]

                valid_indices = np.where(valid_nodes)[0] # array w/ indices of valid nodes 
                index_mapping = -np.ones(x.shape[0], dtype=int)  
                index_mapping[valid_indices] = np.arange(len(valid_indices))  # (24x72) array, -1 if node contains NaN, new numbering for valid nodes
                # original index goes in, new index or -1 comes out

                filtered_edges = (index_mapping[adj_t[0]] != -1) & (index_mapping[adj_t[1]] != -1)
                filtered_adj_t = index_mapping[adj_t[:, filtered_edges]]

                # Creating graph from x:
                g = Data(x = torch.tensor(filtered_x, dtype=torch.float32), edge_index=torch.tensor(filtered_adj_t, dtype=torch.int64))
                torch.save(g, f'{self.processed_dir}/{year}_{month}.pt')
    
    def get(self, idx):
        year = self.years[idx // 36]
        month = idx % 36
        g = torch.load(osp.join(self.processed_dir, f'{year}_{month}.pt'))
        return g

class SODA(Dataset):
    def __init__(self, name='SODA', root=DATA_ROOT, transform=None):
        self.name = name
        self.root = root
        self.urls = ['https://www.dropbox.com/scl/fi/cj9ohivhppqe4u2njn0v7/SODA_train.nc?rlkey=ur7i69js5oejrxgdv8m73r2y4&st=vt4lt2na&dl=1', 
                    'https://www.dropbox.com/scl/fi/9exxnozz8195i4rp2ifac/SODA_label.nc?rlkey=su7x1rnq3ndzwc5yhi53jg74q&st=oeacxlo5&dl=1']
        self.length = None
        self.labels = None
        super().__init__(root, transform)
        
    @property
    def raw_dir(self) -> str:
        return osp.join(self.root, self.name, 'raw')

    def len(self):
        if self.length is None:
            years = len(xr.open_dataset(f'{self.raw_dir}/SODA_train.nc', engine="netcdf4").year.values)
            months = len(xr.open_dataset(f'{self.raw_dir}/SODA_train.nc', engine="netcdf4").month.values)
            self.length = years * months - 72 # last 2 simulations need to be discarded b/c of 2-year lead time
        return self.length
    
    def get_labels(self):
        if self.labels is None:
            labels_unprocessed = xr.open_dataset(f'{self.raw_dir}/SODA_label.nc', engine="netcdf4")['nino'].to_numpy() # 100 x 36
            # Adjusting labels to account for lead time
            shifted = np.roll(labels_unprocessed, shift=-2, axis=0)[:, 11:11+24] # up to 2 years' lead time is typically tested. Thus, we need to shift by 2 years
            self.labels = shifted[:-2]
        return self.labels

    @property
    def processed_dir(self) -> str:
        return osp.join(self.root, self.name, 'processed')

    @property
    def raw_file_names(self):
        return ['SODA_train.nc', 'SODA_label.nc']
    
    @property
    def processed_file_names(self):
        return [f'{i//36}_{i%36}.pt' for i in range(self.len())]

    def download(self) -> None:
        for filename, url in zip(self.raw_file_names, self.urls):
            download_url(url, f'{self.raw_dir}')

    def process(self, adjacency_method='grid'):
        idx = 0
        x_unprocessed = xr.open_dataset(f'{self.raw_dir}/SODA_train.nc', engine="netcdf4")
        adj_t = construct_adjacency_list(method=adjacency_method)

        
        for i in range(self.len()):
            year = i // 36
            month = i % 36
            temp = np.array([x_unprocessed[var].isel(year=year, month=month).to_numpy() for var in VAR_NAMES]) # 4 (var) x 24 (lat) x 72 (lon)
            x = np.transpose(temp, (1, 2, 0)).reshape(-1, 4) # (24 x 72) x 4. Caution: stacks 72's on top of each other, not 24's!

            # Removing NaNs (terrestial nodes)
            valid_nodes = ~np.isnan(x).any(axis=1) # (24x72) bool array
            filtered_x = x[valid_nodes]

            valid_indices = np.where(valid_nodes)[0] # array w/ indices of valid nodes 
            index_mapping = -np.ones(x.shape[0], dtype=int)  
            index_mapping[valid_indices] = np.arange(len(valid_indices))  # (24x72) array, -1 if node contains NaN, new numbering for valid nodes

            filtered_edges = (index_mapping[adj_t[0]] != -1) & (index_mapping[adj_t[1]] != -1)
            filtered_adj_t = index_mapping[adj_t[:, filtered_edges]]

            # Creating graph from x:
            g = Data(x=torch.from_numpy(filtered_x), edge_index=torch.from_numpy(filtered_adj_t))
            torch.save(g, f'{self.processed_dir}/{year}_{month}.pt')
    
    def get(self, idx):
        year = idx // 36
        month = idx % 36
        g = torch.load(osp.join(self.processed_dir, f'{year}_{month}.pt'))
        return g

class SGS(Sampler): # shuffle-by-group sampling
    def __init__(self, data_source, group_size, shuffle=True):
        self.data_source = data_source
        self.group_size = group_size
        self.shuffle = shuffle
        self.num_samples = len(data_source)
        self.num_groups = self.num_samples // self.group_size
        self.shuffle_indices = None

    def __iter__(self): # called once per epoch
        indices = np.arange(self.num_samples)
        indices = indices[:self.num_groups * self.group_size] # dropping last batch
        indices = indices.reshape(self.num_groups, self.group_size)
        
        if self.shuffle:
            shuffler = np.random.permutation(self.num_groups)
            indices = indices[shuffler]
            self.shuffle_indices = shuffler
        return iter(indices.flatten())

    def __len__(self):
        return self.num_samples

class MasterDataModule(pl.LightningDataModule):
    def __init__(self, batch_size):
        super().__init__()
        self.batch_size = batch_size
        self.dataset = None
        self.sampler = None

    def setup(self, stage=None):
        if stage == "fit":
            self.dataset = CMIP() 
            self.sampler = SGS(self.dataset, group_size=self.batch_size, shuffle=True)
        elif stage == "test":
            self.dataset = SODA()
            self.sampler = SGS(self.dataset, group_size=self.batch_size, shuffle=False)
            
    def train_dataloader(self):
        return DataLoader(self.dataset, sampler=self.sampler, batch_size=self.batch_size, num_workers=NUM_WORKERS)

    def test_dataloader(self):
        return DataLoader(self.dataset, sampler=self.sampler, batch_size=self.batch_size, num_workers=NUM_WORKERS)