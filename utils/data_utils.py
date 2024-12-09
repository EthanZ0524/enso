'''
Simple Dataset classes to help with loading in CMIP and SODA datasets.

For more extensive documentation, please see enso/docs/data_utils.md
'''

import xarray as xr
from tqdm import tqdm
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
from global_vars import *

DATA_ROOT = subprocess.check_output(['git', 'rev-parse', '--show-toplevel']).strip().decode('utf-8') + '/data'
VAR_NAMES = ['sst', 't300'] # hardcoded, sorry...

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

class SODA(Dataset):
    '''
    OVERVIEW:
        PyG Dataset class to load in SODA dataset created by authors of https://spj.science.org/doi/10.34133/olar.0012

        This dataset contains graphs for months 0 - m and labels for months 1 - m+1. A 'row' in the dataset corresponds to 
        36 graphs (months n - n+35) and 36 labels (months n+1 - n+36). Each row overlaps the previous by 12 months.

        We save the first 12 months (graphs) from each year to create a graph timeseries. This leaves us with 24 months of 
        labels extending past our latest graph (technically 25, but I just ignore the off-by-one). We do this because 
        labels must exist 24 months further than graphs.

        Graph file names are 0-indexed.

    WINDOW_INDICES AND LABELS
        The most important implementations in this class other than the graph processing is the creation of the self.window_indices
        and self.labels variables. Both will be necessary for proper shuffling and label retrieval further downstream.

        window_indices is a bookkeeping array for eventual shuffling/batching. It is an array of arrays. Each 
        subarray contains 36 graph indices. All valid 36-graph minibatches for the given dataset are represented by 
        these subarrays.

        labels is an array of arrays, where each subarray contains a 24-month label (ONI) sequence corresponding to the 
        window_indices subarray of the same index. 
    '''
    def __init__(self, name='SODA', root=DATA_ROOT, transform=None, adjacency_method='grid'):
        self.name = name
        self.root = root
        self.urls = ['https://www.dropbox.com/scl/fi/cj9ohivhppqe4u2njn0v7/SODA_train.nc?rlkey=ur7i69js5oejrxgdv8m73r2y4&st=vt4lt2na&dl=1', 
                    'https://www.dropbox.com/scl/fi/9exxnozz8195i4rp2ifac/SODA_label.nc?rlkey=su7x1rnq3ndzwc5yhi53jg74q&st=oeacxlo5&dl=1']
        self.length = None
        self.labels = None
        
        model_indices = np.array([np.arange(self.len())])
        window_indices = []
        for sublist in model_indices:
            window_indices.extend([sublist[i:i+NUM_INPUT_MONTHS] for i in range(len(sublist) - NUM_INPUT_MONTHS + 1)])
        self.window_indices = np.array(window_indices)

        self.adjacency_method = adjacency_method
        super().__init__(root, transform)
        
    @property
    def raw_dir(self) -> str:
        return osp.join(self.root, self.name, 'raw')

    @property
    def processed_dir(self) -> str:
        return osp.join(self.root, self.name, f'{self.adjacency_method}_processed')

    @property
    def raw_file_names(self):
        return ['SODA_train.nc', 'SODA_label.nc']

    def download(self) -> None:
        for filename, url in zip(self.raw_file_names, self.urls):
            download_url(url, f'{self.raw_dir}')

    def len(self):
        if self.length is None:
            years = len(xr.open_dataset(f'{self.raw_dir}/SODA_train.nc', engine="netcdf4").year.values)
            self.length = 12 * years
        return self.length

    def get_labels(self):
        if self.labels is None:
            labels_unprocessed = xr.open_dataset(f'{self.raw_dir}/SODA_label.nc', engine="netcdf4")['nino'].to_numpy() # 100 x 36
            labels_unprocessed = labels_unprocessed[:, -12:].flatten() # (100 x 12)-element array. First point corresponds to 26th month, want 37th 
            labels_unprocessed = labels_unprocessed[11:-1] # off-by-one on the dataset. Thanks...
            self.labels = np.array([labels_unprocessed[i:i+NUM_OUTPUT_MONTHS] for i in range(len(labels_unprocessed) - NUM_OUTPUT_MONTHS + 1)])
        return self.labels 
    
    @property
    def processed_file_names(self):
        return [f'{i}.pt' for i in range(self.len())]

    # Helper function: given a year and month, save the graph as f'm{idx}.pt'
    def save_graph(self, x_unprocessed, adj_t, year, month, idx): 
        temp = [x_unprocessed[var].isel(year=year, month=month).to_numpy() for var in VAR_NAMES] # list of 24 (lat) x 72 (lon) arrays
        lats = x_unprocessed['lat'].to_numpy() # 24
        lats_features = np.repeat(np.sin(np.radians(lats))[:, np.newaxis], 72, axis=1)
        temp.append(lats_features)
        lons = x_unprocessed['lon'].to_numpy() # 72
        lons_features = np.repeat(np.sin(np.radians(lons))[:, np.newaxis], 24, axis=1).T
        temp.append(lons_features)

        temp = np.stack(temp, axis=0) # 4 x 24 (lat) x 72 (lon)
        x = temp.transpose(1, 2, 0).reshape(-1, 4) # (24 x 72) x 4.

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
        torch.save(g, f'{self.processed_dir}/{idx}.pt')

    def process(self):
        x_unprocessed = xr.open_dataset(f'{self.raw_dir}/SODA_train.nc', engine="netcdf4")
        adj_t = construct_adjacency_list(method=self.adjacency_method)
        
        for i in range(self.len()):
            year = i // 12
            month = i % 12
            self.save_graph(x_unprocessed=x_unprocessed, adj_t=adj_t, year=year, month=month, idx=i)
    
    def get(self, idx):
        g = torch.load(osp.join(self.processed_dir, f'{idx}.pt'))
        return g


class CMIP(Dataset):
    '''
    OVERVIEW:
        PyG Dataset class to load in CMIP dataset created by authors of https://spj.science.org/doi/10.34133/olar.0012

        This dataset contains data for 15 CMIP6 models and 17 CMIP5 models. For each model, the dataset contains
        graphs for months 0 - m and labels for months 1 - m+1. A 'row' in the dataset corresponds to 
        36 graphs (months n - n+35) and 36 labels (months n+1 - n+36). Each row overlaps the previous by 12 months.
        CMIP6 models run for 151 years, and CMIP5 models run for 140 years.

        We save the first 12 months (graphs) from each year to create a graph timeseries for each model, containing
        year x 12 graphs (where year = 151 or 140). However, for each model, the last two years' graphs won't be saved 
        because no labels exist for them (labels must exist 24 months further than graphs).

        Graph file names are 0-indexed. Graphs will be indexed consecutively (ie. if the last graph of model 0 is index x,
        the first graph of model 1 is index x+1).

    WINDOW_INDICES AND LABELS
        The most important implementations in this class other than the graph processing is the creation of the self.window_indices
        and self.labels variables. Both will be necessary for proper shuffling and label retrieval further downstream.

        window_indices is a bookkeeping array for eventual shuffling/batching. It is an array of arrays. Each 
        subarray contains 36 graph indices. All valid 36-graph minibatches for the given dataset are represented by 
        these subarrays. Because this CMIP dataset contains 32 models, the subarrays are set up such that no subarray
        contains indices of graphs from 2 models.

        labels is another array of arrays, where each subarray contains a 24-month label (ONI) sequence corresponding to the 
        window_indices subarray of the same index. 
    '''

    class CMIP(Dataset):
    def __init__(self, name='CMIP', root=DATA_ROOT, transform=None, adjacency_method='grid'):
        self.name = name
        self.root = root
        self.urls = ['https://www.dropbox.com/scl/fi/i22s15q6b9c8q205dhe20/CMIP_merged.nc?rlkey=k4qgkaluc1267tlp6y8u3uaoy&st=14yvvnu3&dl=1']   
        self.length = None
        self.labels = None
        self.adjacency_method = adjacency_method

        best_models = get_best_cmip_models(NUM_MODELS)['model_name'].to_numpy()
        x_unprocessed = xr.open_dataset(f'{self.raw_dir}/CMIP_merged.nc', engine="netcdf4")
        self.members = [str(sim_id) for sim_id in x_unprocessed['simulation_id'].to_numpy() if sim_id.split(':')[0] in best_models]

        # Constructing window_indices, a bookkeeping array for eventual shuffling/batching. 
        # Creates a np.array of np.arrays of 36 graph indices - ensures inter-model shuffling doesn't occur.

        # self.members models, 1980 - 25 graphs each
        window_indices = []
        lists = [np.array(list(range(i, i + 1980 - 25))) for i in range(0, len(self.members) * (1980 - 25), 1980 - 25)]

        for sublist in lists:
            window_indices.extend(np.array([sublist[i:i+NUM_INPUT_MONTHS] for i in range(len(sublist) - NUM_INPUT_MONTHS + 1)]))
        self.window_indices = np.array(window_indices) 

        super().__init__(root, transform)
        
    @property
    def raw_dir(self) -> str:
        return osp.join(self.root, self.name, 'raw')
    
    @property
    def processed_dir(self) -> str:
        return osp.join(self.root, self.name, f'{self.adjacency_method}_processed')

    @property
    def raw_file_names(self):
        return ['CMIP_merged.nc']

    def download(self) -> None:
        for filename, url in zip(self.raw_file_names, self.urls):
            download_url(url, f'{self.raw_dir}')

    def len(self):
        return len(self.members) * (1980 - 25)
    
    def get_labels(self):
        if self.labels is None:
            labels_unprocessed = xr.open_dataset(f'{self.raw_dir}/CMIP_merged.nc', engine="netcdf4").sel(simulation_id=self.members)['oni'].to_numpy() # self.members x 1980
            labels_unprocessed = labels_unprocessed[:, 24:-1]
            labels_unprocessed = labels_unprocessed.flatten()
            self.labels = labels_unprocessed[self.window_indices[:, 12:]]
        return self.labels

    @property
    def processed_file_names(self):
        return [f'{i}.pt' for i in range(self.len())]
        
    def process(self):
        members = self.members
        adj_t = construct_adjacency_list(method=self.adjacency_method)
        idx = 0
        x_unprocessed = xr.open_dataset(f'{self.raw_dir}/CMIP_merged.nc', engine="netcdf4")

        for member in tqdm(members):
            member_data = x_unprocessed.sel(simulation_id=member)
            for time in range(len(x_unprocessed.time.values) - 24 - 1): # off-by-one because last label is NaN
                temp = [member_data[var].isel(time=time).to_numpy() for var in VAR_NAMES] # list of 24 (lat) x 72 (lon) arrays
                lats = member_data['lat'].to_numpy() # 24
                lats_features = np.repeat(np.sin(np.radians(lats))[:, np.newaxis], 72, axis=1)
                temp.append(lats_features)
                lons = member_data['lon'].to_numpy() # 72
                lons_features = np.repeat(np.sin(np.radians(lons))[:, np.newaxis], 24, axis=1).T
                temp.append(lons_features)

                temp = np.stack(temp, axis=0) # 4 x 24 (lat) x 72 (lon)
                x = temp.transpose(1, 2, 0).reshape(-1, 4) # (24 x 72) x 2. Caution: stacks 72's on top of each other, not 24's!

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
                torch.save(g, f'{self.processed_dir}/{idx}.pt')
                idx += 1
    
    def get(self, idx):
        g = torch.load(osp.join(self.processed_dir, f'{idx}.pt'))
        return g

class GODAS(Dataset):
    '''
    OVERVIEW:
        PyG Dataset class to load in GODAS dataset, regridded by Yuchen.

        This dataset contains graphs for months 0 - m and labels for months 0 - m. 

        Graph file names are 0-indexed.

        Because we never train on GODAS, we don't need window_indices or labels class
        variables. The dataset is small enough that we can directly load labels in predict_plot.ipynb 
    '''
    def __init__(self, name='GODAS', root=DATA_ROOT, transform=None, adjacency_method='grid'):
        self.name = name
        self.root = root
        self.urls = ['https://www.dropbox.com/scl/fi/nh7jlpt377sesz5urd4tr/GODAS_regridded.nc?rlkey=crfxbb5i9qjol39syqrmvcz6l&st=jwhnaich&dl=1']
        self.length = None
        self.adjacency_method = adjacency_method
        super().__init__(root, transform)
        
    @property
    def raw_dir(self) -> str:
        return osp.join(self.root, self.name, 'raw')

    @property
    def processed_dir(self) -> str:
        return osp.join(self.root, self.name, f'{self.adjacency_method}_processed')

    @property
    def raw_file_names(self):
        return ['GODAS_regridded.nc']

    def download(self) -> None:
        for filename, url in zip(self.raw_file_names, self.urls):
            download_url(url, f'{self.raw_dir}')

    def len(self):
        if self.length is None:
            num_graphs = len(xr.open_dataset(f'{self.raw_dir}/GODAS_regridded.nc', engine="netcdf4").time.values)
            self.length = num_graphs - 25 # 25 instead of 24 b/c last label is missing
        return self.length
    
    @property
    def processed_file_names(self):
        return [f'{i}.pt' for i in range(self.len())]

    # Helper function: given a year and month, save the graph as f'm{idx}.pt'
    def save_graph(self, x_unprocessed, adj_t, idx): 
        temp = [x_unprocessed[var].to_numpy()[idx] for var in VAR_NAMES] # 2 (var) x 24 (lat) x 72 (lon)
        x = np.transpose(temp, (1, 2, 0)).reshape(-1, 2) # (24 x 72) x 2. Caution: stacks 72's on top of each other, not 24's!

        lats = x_unprocessed['lat'].to_numpy() # 24
        lats_features = np.repeat(np.sin(np.radians(lats))[:, np.newaxis], 72, axis=1)
        temp.append(lats_features)
        lons = x_unprocessed['lon'].to_numpy() # 72
        lons_features = np.repeat(np.sin(np.radians(lons))[:, np.newaxis], 24, axis=1).T
        temp.append(lons_features)
        temp = np.stack(temp, axis=0) # 4 x 24 (lat) x 72 (lon)
        x = temp.transpose(1, 2, 0).reshape(-1, 4) # (24 x 72) x 2. Caution: stacks 72's on top of each other, not 24's!

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
        torch.save(g, f'{self.processed_dir}/{idx}.pt')

    def process(self):
        x_unprocessed = xr.open_dataset(f'{self.raw_dir}/GODAS_regridded.nc', engine="netcdf4")
        adj_t = construct_adjacency_list(method=self.adjacency_method)
        
        for i in range(self.len()):
            self.save_graph(x_unprocessed=x_unprocessed, adj_t=adj_t, idx=i)
    
    def get(self, idx):
        g = torch.load(osp.join(self.processed_dir, f'{idx}.pt'))
        return g
    
class SGS(Sampler): # shuffle-by-group sampling
    def __init__(self, data_source, group_size, shuffle=True):
        self.data_source = data_source
        self.group_size = group_size
        self.shuffle = shuffle
        self.shuffle_indices = None # np.random.permutation array to shuffle windows and labels
        self.window_indices = self.data_source.window_indices
        self.num_batches = len(self.window_indices.flatten()) // self.group_size # throwing away remainder

    def __iter__(self): # called once per epoch
        indices = self.window_indices
        if self.shuffle:
            shuffler = np.random.permutation(len(indices))
            indices = indices[shuffler]
            self.shuffle_indices = shuffler
        indices = indices.flatten()[:self.group_size * self.num_batches]
        return iter(indices)

    def __len__(self): # number of 'logical samples' (eg. graphs)!
        return len(self.data_source)

'''
Quick wrapper class that properly counts the number of batches in an epoch. DataLoaders typically assume batches
are non-overlapping - this class adjusts for that, because our batches do overlap, which would cause miscounting.
'''
class CustomDataLoader(DataLoader):
    def __init__(self, custom_len: int= None, **kwargs):
        self.custom_len = custom_len
        super().__init__(**kwargs)
    
    def __len__(self):
        if self.custom_len is not None:
            return self.custom_len
        return super().__len__()

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
        return CustomDataLoader(custom_len=self.sampler.num_batches, dataset=self.dataset, sampler=self.sampler, batch_size=self.batch_size, num_workers=NUM_WORKERS)

    def test_dataloader(self):
        return CustomDataLoader(custom_len=self.sampler.num_batches, dataset=self.dataset, sampler=self.sampler, batch_size=self.batch_size, num_workers=NUM_WORKERS)