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

from data_retrieval.cmip_data_utils import get_best_cmip_models

from data_retrieval.data_config import DATA_DIR 

VAR_NAMES = ['sst', 't300'] # hardcoded, sorry...

def get_allocated_cpus():
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:
        return 0

NUM_WORKERS = min(get_allocated_cpus() - 1, 0)



def get_neighbors(base_rows, base_cols, i, j, method, scale=1):
    if type(scale) != int: raise ValueError("scale must be int")
    
    neighbors = []

    if method == "grid" or method == "dense_grid":
        directions = [(-scale, 0), (scale, 0), (0, -scale), (0, scale)]
        if method == "dense_grid":
            directions += [(-scale, -scale), (-scale, scale), (scale, -scale), (scale, scale)]

        for di, dj in directions:
            neighbor_row = i + di
            neighbor_col = (j + dj) % base_cols  # Periodic in east-west
            if 0 <= neighbor_row < base_rows:
                neighbors.append((neighbor_row, neighbor_col))

    return neighbors


def construct_adjacency_list_core(grid_size, method="grid", scales=None, origins=None, verbose=False):
    """
    Helper function: generates the adjacency list for a 24x72 grid of nodes.

    Params:
        method (str): base grid method. Options:
            "grid": connect each node to its 4 up-down-left-right neighbors.
            "dense_grid": connect each node to its 8 neighbors (including diagonals).
        scales (list of int): list of scales for generating downscaled grids. Each scale indicates the reduction factor.
        origins (list of tuple): list of origin tuples (row, col) for each downscaled grid. Should match the length of scales.

    Returns:
        np.array: adjacency list of shape (2, num_edges) 
    """
    rows, cols = grid_size
    adjacency_list = []

    if not scales:
        if origins: 
            raise ValueError("Cannot set `origins` if your scale is 1")
        scales = [1]

    if not origins:
        origins = [(0,0) for i in range(len(scales))]

    if len(scales) != len(origins):
            raise ValueError("`scales` and `origins` must have the same length.")

    for scale, origin in zip(scales, origins):
        for i in range(origin[0], rows, scale):
            for j in range(origin[1], cols, scale):
                curr_index = i * cols + j 

                neighbors = get_neighbors(rows, cols, i, j, method, scale=scale)
                if verbose: print(f"node {i,j} has neighbors {neighbors}")
                for neighbor_i, neighbor_j in neighbors:
                    neighbor_index = neighbor_i * cols + neighbor_j 
                    if verbose: print(f"appending {(curr_index, neighbor_index)}")
                    adjacency_list.append((curr_index, neighbor_index))

    adj_t = np.array(adjacency_list).T
    return adj_t


def construct_adjacency_list(method):
    """ 
    Wrapper function for constructing adjacency list with some presets
    """
    grid_size = (24, 72)

    if method == "simple_grid": 
        return construct_adjacency_list_core(grid_size, method="grid", scales=[1], origins=[(0,0)])
    
    elif method == "simple_grid_dense":
        return construct_adjacency_list_core(grid_size, method="dense_grid", scales=[1], origins=[(0,0)])

    elif method == "multimesh1":
        return construct_adjacency_list_core(grid_size, method="dense_grid", 
                                            scales=[1, 3, 6], origins=[(0,0), (1,1), (4,1)])
    
    else:
        raise NotImplementedError(f"That adjacency method has not been implemented!\
             Current settings: simple_grid, simple_grid_dense, multimesh1 ")

class SODA_train(Dataset):
    '''
    OVERVIEW:
        PyG Dataset class to load in SODA dataset.

        This dataset contains graphs and labels for 1236 months. 

        We save all but the last 25 graphs to create a graph timeseries. This leaves us with 25 months of 
        labels extending past our latest graph. We do this because labels must exist 24 months further than graphs;
        the last label is a NaN so we discard it.

        Graph file names are 0-indexed.

    WINDOW_INDICES AND LABELS
        The most important implementations in this class other than the graph processing is the creation of the self.window_indices
        and self.labels variables. Both will be necessary for proper shuffling and label retrieval further downstream.

        window_indices is a bookkeeping array for eventual shuffling/batching. It is an array of arrays. Each 
        subarray contains 36 graph indices. All valid 36-graph minibatches for the given dataset are represented by 
        these subarrays.

        labels is an array of arrays, where each subarray contains a 24-month label (ONI) sequence corresponding to the 
        window_indices subarray of the same index. 

        Hardcoded length: 800 graphs
    '''
    def __init__(self, name='SODA', root=DATA_DIR, transform=None, adjacency_method='grid'):
        self.name = name
        self.root = root
        self.urls = ['https://www.dropbox.com/scl/fi/iqsidx76nexabhqv7eygz/SODA.nc?rlkey=bho72gwfiug3yompoevadx348&st=n3nvbcwf&dl=1']   
        self.labels = None
        self.adjacency_method = adjacency_method
        
        # Constructing window_indices, a bookkeeping array for eventual shuffling/batching. 
        # Creates a np.array of np.arrays of 36 graph indices

        # using graphs 0 - 799
        window_indices = []
        window_indices.extend(np.array([np.arange(i, i+36) for i in range(765)])) # last window will be 764 - 799
        self.window_indices = np.array(window_indices) 

        super().__init__(root, transform)
        
    @property
    def raw_dir(self) -> str:
        return osp.join(self.root, self.name, 'raw')
    
    @property
    def processed_dir(self) -> str:
        return osp.join(self.root, self.name, 'train', f'{self.adjacency_method}_processed')

    @property
    def raw_file_names(self):
        return ['SODA.nc']

    def download(self) -> None:
        for filename, url in zip(self.raw_file_names, self.urls):
            download_url(url, f'{self.raw_dir}')

    def len(self):
        return 800

    def get_labels(self):
        if self.labels is None:
            labels_unprocessed = xr.open_dataset(f'{self.raw_dir}/SODA.nc', engine="netcdf4")['oni'].to_numpy() # 1236
            labels_unprocessed = labels_unprocessed[24:-1] # first element is first label of interest
            self.labels = labels_unprocessed[self.window_indices[:, 12:]]
        return self.labels

    @property
    def processed_file_names(self):
        return [f'{i}.pt' for i in range(800)]
        
    def process(self):
        adj_t = construct_adjacency_list(method=self.adjacency_method)
        x_unprocessed = xr.open_dataset(f'{self.raw_dir}/SODA.nc', engine="netcdf4")

        for time in range(800): 
            temp = [x_unprocessed[var].isel(time=time).to_numpy() for var in VAR_NAMES] # list of 24 (lat) x 72 (lon) arrays
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
            # original index goes in, new index or -1 comes out

            filtered_edges = (index_mapping[adj_t[0]] != -1) & (index_mapping[adj_t[1]] != -1)
            filtered_adj_t = index_mapping[adj_t[:, filtered_edges]]

            # Creating graph from x:
            g = Data(x = torch.tensor(filtered_x, dtype=torch.float32), edge_index=torch.tensor(filtered_adj_t, dtype=torch.int64))
            torch.save(g, f'{self.processed_dir}/{time}.pt')
    
    def get(self, idx):
        g = torch.load(osp.join(self.processed_dir, f'{idx}.pt'))
        return g

class SODA_val(Dataset):
    '''
    OVERVIEW:
        PyG Dataset class to load in SODA dataset.

        This dataset contains graphs and labels for 1236 months. 

        We save all but the last 25 graphs to create a graph timeseries. This leaves us with 25 months of 
        labels extending past our latest graph. We do this because labels must exist 24 months further than graphs;
        the last label is a NaN so we discard it.

        Graph file names are 0-indexed.

    WINDOW_INDICES AND LABELS
        The most important implementations in this class other than the graph processing is the creation of the self.window_indices
        and self.labels variables. Both will be necessary for proper shuffling and label retrieval further downstream.

        window_indices is a bookkeeping array for eventual shuffling/batching. It is an array of arrays. Each 
        subarray contains 36 graph indices. All valid 36-graph minibatches for the given dataset are represented by 
        these subarrays.

        labels is an array of arrays, where each subarray contains a 24-month label (ONI) sequence corresponding to the 
        window_indices subarray of the same index. 

        Hardcoded length: 411 graphs (1236 - 800 - 24 - 1)
    '''
    def __init__(self, name='SODA', root=DATA_DIR, transform=None, adjacency_method='grid'):
        self.name = name
        self.root = root
        self.urls = ['https://www.dropbox.com/scl/fi/iqsidx76nexabhqv7eygz/SODA.nc?rlkey=bho72gwfiug3yompoevadx348&st=n3nvbcwf&dl=1']   
        self.labels = None
        self.adjacency_method = adjacency_method
        
        # Constructing window_indices, a bookkeeping array for eventual shuffling/batching. 
        # Creates a np.array of np.arrays of 36 graph indices
        # saving graphs 800 - 1210

        window_indices = []
        window_indices.extend(np.array([np.arange(i, i+36) for i in range(376)])) # last window will be 375 - 410
        # first window raw indices: 800 - 835. First label should then be 836
        self.window_indices = np.array(window_indices) 

        super().__init__(root, transform)
        
    @property
    def raw_dir(self) -> str:
        return osp.join(self.root, self.name, 'raw')
    
    @property
    def processed_dir(self) -> str:
        return osp.join(self.root, self.name, 'val', f'{self.adjacency_method}_processed')

    @property
    def raw_file_names(self):
        return ['SODA.nc']

    def download(self) -> None:
        for filename, url in zip(self.raw_file_names, self.urls):
            download_url(url, f'{self.raw_dir}')

    def len(self):
        return 411

    def get_labels(self):
        if self.labels is None:
            labels_unprocessed = xr.open_dataset(f'{self.raw_dir}/SODA.nc', engine="netcdf4")['oni'].to_numpy() # 1236
            labels_unprocessed = labels_unprocessed[824:-1] 
            self.labels = labels_unprocessed[self.window_indices[:, 12:]]
        return self.labels

    @property
    def processed_file_names(self):
        return [f'{i}.pt' for i in range(self.len())]
        
    def process(self):
        adj_t = construct_adjacency_list(method=self.adjacency_method)
        x_unprocessed = xr.open_dataset(f'{self.raw_dir}/SODA.nc', engine="netcdf4")

        for idx, time in enumerate(range(800, 1211)): 
            temp = [x_unprocessed[var].isel(time=time).to_numpy() for var in VAR_NAMES] # list of 24 (lat) x 72 (lon) arrays
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
            # original index goes in, new index or -1 comes out

            filtered_edges = (index_mapping[adj_t[0]] != -1) & (index_mapping[adj_t[1]] != -1)
            filtered_adj_t = index_mapping[adj_t[:, filtered_edges]]

            # Creating graph from x:
            g = Data(x = torch.tensor(filtered_x, dtype=torch.float32), edge_index=torch.tensor(filtered_adj_t, dtype=torch.int64))
            torch.save(g, f'{self.processed_dir}/{idx}.pt')
    
    def get(self, idx):
        g = torch.load(osp.join(self.processed_dir, f'{idx}.pt'))
        return g


class CMIP(Dataset):
    '''
    OVERVIEW:
        PyG Dataset class to load in CMIP dataset.

        This dataset contains graphs and labels for 1980 months. The number of models in the dataaset is determined
        by the global variable NUM_MODELS.  

        For each model ensemble member, save all but the last 25 graphs to create a graph timeseries. This leaves us with 
        25 months of labels extending past the latest graph for each member. We do this because labels must exist 24 months 
        further than graphs; the last label is a NaN so we discard it.

        Graph file names are 0-indexed. Graphs will be indexed consecutively (ie. if the last graph of model 0 is index x,
        the first graph of model 1 is index x+1).

    WINDOW_INDICES AND LABELS
        The most important implementations in this class other than the graph processing is the creation of the self.window_indices
        and self.labels variables. Both will be necessary for proper shuffling and label retrieval further downstream.

        window_indices is a bookkeeping array for eventual shuffling/batching. It is an array of arrays. Each 
        subarray contains 36 graph indices. All valid 36-graph minibatches for the given dataset are represented by 
        these subarrays. Because this CMIP dataset data from multiple ensemble members, the subarrays are set up such 
        that no subarray contains indices of graphs from 2 ensemble members.

        labels is another array of arrays, where each subarray contains a 24-month label (ONI) sequence corresponding to the 
        window_indices subarray of the same index. 
    '''
    def __init__(self, name='CMIP', num_models=NUM_MODELS, root=DATA_DIR, transform=None, adjacency_method='grid'):
        self.name = name
        self.num_models = num_models
        self.root = root
        self.urls = ['https://www.dropbox.com/scl/fi/i22s15q6b9c8q205dhe20/CMIP_merged.nc?rlkey=k4qgkaluc1267tlp6y8u3uaoy&st=14yvvnu3&dl=1']   
        self.labels = None
        self.adjacency_method = adjacency_method

        best_models = get_best_cmip_models(self.num_models)['model_name'].to_numpy()
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
    def __init__(self, name='GODAS', root=DATA_DIR, transform=None, adjacency_method='grid'):
        self.name = name
        self.root = root
        self.urls = ['https://www.dropbox.com/scl/fi/uzlgv1khwiz9rwb1ipbsc/GODAS.nc?rlkey=14iwz99wzhdqd7ml3tdrwe660&st=nd5gpgaf&dl=1']
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
        return ['GODAS.nc']

    def download(self) -> None:
        for filename, url in zip(self.raw_file_names, self.urls):
            download_url(url, f'{self.raw_dir}')

    def len(self):
        num_graphs = len(xr.open_dataset(f'{self.raw_dir}/GODAS.nc', engine="netcdf4").time.values)
        return num_graphs - 24 - 1
    
    @property
    def processed_file_names(self):
        return [f'{i}.pt' for i in range(self.len())]

    def process(self):
        adj_t = construct_adjacency_list(method=self.adjacency_method)
        x_unprocessed = xr.open_dataset(f'{self.raw_dir}/GODAS.nc', engine="netcdf4")

        for time in range(self.len()): 
            temp = [x_unprocessed[var].isel(time=time).to_numpy() for var in VAR_NAMES] # list of 24 (lat) x 72 (lon) arrays
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
            # original index goes in, new index or -1 comes out

            filtered_edges = (index_mapping[adj_t[0]] != -1) & (index_mapping[adj_t[1]] != -1)
            filtered_adj_t = index_mapping[adj_t[:, filtered_edges]]

            # Creating graph from x:
            g = Data(x = torch.tensor(filtered_x, dtype=torch.float32), edge_index=torch.tensor(filtered_adj_t, dtype=torch.int64))
            torch.save(g, f'{self.processed_dir}/{time}.pt')
    
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
    def __init__(self, batch_size, finetune, adjacency, num_cmip_models):
        super().__init__()
        self.batch_size = batch_size
        self.dataset = None
        self.sampler = None
        self.val_dataset = None
        self.val_sampler = None
        self.finetune = finetune
        self.adjacency = adjacency
        self.num_cmip_models = num_cmip_models

    def setup(self, stage=None):
        if stage == "fit":
            if not self.finetune:
                self.dataset = CMIP(adjacency_method=self.adjacency, num_models=self.num_cmip_models) 
                self.sampler = SGS(self.dataset, group_size=self.batch_size, shuffle=True)
            else:
                self.dataset = SODA_train(adjacency_method=self.adjacency)
                self.sampler = SGS(self.dataset, group_size=self.batch_size, shuffle=True)

        self.val_dataset = SODA_val(adjacency_method=self.adjacency)
        self.val_sampler = SGS(self.val_dataset, group_size=36, shuffle=False) # want to do window-by-window prediction
            
    def train_dataloader(self):
        return CustomDataLoader(custom_len=self.sampler.num_batches, dataset=self.dataset, sampler=self.sampler, batch_size=self.batch_size, num_workers=NUM_WORKERS)

    def val_dataloader(self):
        return CustomDataLoader(custom_len=self.val_sampler.num_batches, dataset=self.val_dataset, sampler=self.val_sampler, batch_size=36, num_workers=NUM_WORKERS)