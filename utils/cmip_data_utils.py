"""
This script provides some basic functionality for subsetting the merged CMIP dataset
"""

import xarray as xr
import numpy as np 
import pandas as pd 
import os
import config 

# change as necessary
enso_stats_path = os.path.join(config.DATA_DIR, "ENSO_performance_CMIP_normalized.csv")
cmip_merged = xr.open_dataset(os.path.join(config.DATA_DIR, "merged_data/CMIP_merged.nc"))


def get_model_names_dict(cmip_ds):
    """
    Returns a dict mapping model names to the number of ensemble members for each model.

    Param:
        (xr.Dataset)    cmip_ds (needs to be cmip_merged.nc)
    """

    models = {}
    for simulation in cmip_ds.simulation_id.values:
        model_name = simulation.split(":")[0]
        if model_name in models.keys():
            models[model_name] += 1
        else:
            models[model_name] = 1

    return models


def subset_cmip_ds(cmip_ds, model_name):
    """
    Returns a xr.Dataset containing only the simulations corresponding to model_name 

    Param:
        (xr.Dataset)    cmip_ds (needs to be cmip_merged.nc)
        (string)        model_name
    """

    subset = [model_name == sim_id.split(":")[0] for sim_id in cmip_ds.simulation_id.values]
    return cmip_ds.isel(simulation_id=subset)


def get_best_cmip_models(n):
    """
    Returns a DataArray of n of the best CMIP models, evaluated as averaging over the ENSO
    performance statistics given by enso_stats. 

    Citation: https://journals.ametsoc.org/view/journals/bams/102/2/BAMS-D-19-0337.1.xml#tbla3

    Param:
        (int)       n: the number of model names to return 

    Returns: xr.DataArray
    """
    if not os.path.exists(enso_stats_path):
        print(f"{enso_stats_path} does not exist!")
        return None

    enso_stats_df = pd.read_csv(enso_stats_path)
    enso_stats_df = enso_stats_df.rename(columns={"Unnamed: 0": "model_name"})

    # Clean the "model_name" column to remove extra asterisks
    enso_stats_df["model_name"] = enso_stats_df["model_name"].str.replace("*", "", regex=False).str.strip()

    # Convert to an xarray Dataset and set "model_name" as the index
    enso_stats_ds = enso_stats_df.set_index("model_name").to_xarray()

    # subset to the models we have downloaded 
    model_name_list = get_model_names_dict(cmip_merged).keys()
    enso_stats_ds = enso_stats_ds.sel(model_name=[name for name in model_name_list \
                                        if name in enso_stats_ds.model_name.values])

    enso_stats_da_list = []
    for var in enso_stats_ds.variables:
        if var[0:4] == "Enso":
            enso_stats_da_list.append(enso_stats_ds.rename({var : "enso_performance_metric"})["enso_performance_metric"])

    avg_enso_performance = xr.concat(enso_stats_da_list, dim="enso_performance_metric").mean("enso_performance_metric")

    lowest_avg_models = avg_enso_performance.sortby(avg_enso_performance).isel(model_name=slice(0, n))

    return lowest_avg_models
