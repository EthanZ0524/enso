import intake
import numpy as np
import xarray as xr 
import dask
import xesmf as xe
import time 
import os
import pickle
import pandas as pd 

from xmip.preprocessing import rename_cmip6
from xmip.preprocessing import correct_lon
from xmip.preprocessing import promote_empty_dims

import warnings
warnings.filterwarnings("ignore")

SAVE_DIRECTORY = "/scratch/users/yucli/enso_data/"

# Load in the data from the cloud 
col = intake.open_esm_datastore("https://storage.googleapis.com/cmip6/pangeo-cmip6.json")

def get_catalog():
    """
    Returns
    1) (catalog)    pangeo-cmip6 catalog subsetted for monthly ocean potential temperature (thetao) 
    2) (list)       unique_models (name of models or "source_id")
    """

    query = dict(experiment_id=['historical'], table_id=['Omon'],
                variable_id=['thetao'], grid_label=['gn'])

    cat = col.search(**query)

    df = cat.df

    # source_id is the model name
    unique_models = sorted(df["source_id"].unique())

    # skip ICON-ESM because it doesn't have xy coords 
    unique_models.remove('ICON-ESM-LR')
    # skip MCM-UA-1-0 because it has no y index 
    unique_models.remove('MCM-UA-1-0') 
    # skip AWI models because they are on a uniquely unstructured grid 
    unique_models.remove('AWI-CM-1-1-MR')
    unique_models.remove('AWI-ESM-1-1-LR')

    return cat, unique_models


def load_data_as_dset(cat, source_id):
    """
    Param:
        (pangeo-cmip6 catalog)  cat
        (string)                name of model (source_id)   

    Returns:
        (xr.Dataset)            xarray dataset for model 
    """

    def xmip_preprocessing_wrapper(ds):
        ds = ds.copy()
        ds = rename_cmip6(ds)
        ds = promote_empty_dims(ds)
        ds = correct_lon(ds)
        return ds

    cat_subset = cat.search(source_id=source_id)


    # Note 
    #
    # For some reason, the xmip_preprocessing_wrapper function causes issues for some
    # models, where rename_cmip6 does not seem to be applied correctly. A temporary
    # workaround is applying the preprocessing function after retrieving it using intake-esm

    try:
        x_kwargs = {"consolidated": True, "decode_times": True, "use_cftime": True}
        with dask.config.set(**{'array.slicing.split_large_chunks': True}):
            dset_dict = cat_subset.to_dataset_dict(xarray_open_kwargs=x_kwargs, preprocess=rename_cmip6)
    except: 
        print(f"Error retrieving dset_dict using intake for {source_id}. Skipping...")
        return None

    dset_ids = list(dset_dict.keys())

    if len(dset_ids) == 1: 
        ds = xmip_preprocessing_wrapper(dset_dict[dset_ids[0]])
        return ds
    else:
        print(f"Found multiple datasets for the given query:\n{dset_ids} \nSkipping...")
        return None


def subset_data(ds): 
    """
    From thetao (potential temperature), subset the SST (defined as the topmost depth)
    and T300 (weighted-average of top 300 m). 

    Note that xarray does lazy evaluation so no computation is actually done when you 
    call this. 

    Param:
        (xr.Dataset)    dset containing thetao as a variable 

    Returns:
        (xr.DataArray)  sst
        (xr.DataArray)  t300 
    """
    
    # get sst field 
    sst = ds.thetao.isel(lev=0)
    
    # check the units for the depth variable 'lev'
    lev_units = "m"
    if max(ds.lev.values) > 1e5: 
        lev_units = "cm"

    # select top 300 m to calculate OHC
    if lev_units == "m":
        depth_slice = slice(0,300) 
    elif lev_units == "cm":
        depth_slice = slice(0,30000)

    thetao_300m = ds.thetao.sel(lev=depth_slice)

    # calculate how thick each grid cell is to take the weighted mean over depth
    lev_midpoints = np.append([0], (ds.lev[1:].values + ds.lev[:-1].values) / 2)
    lev_thickness = np.diff(lev_midpoints)
    lev_thickness = lev_thickness[0:len(thetao_300m.lev)] 
    
    lev_weights = xr.DataArray(
        data=lev_thickness,
        dims=["lev"],
        coords=dict(
            lev=thetao_300m.lev.values # the weights live at the original levs, not the midpts
        ),
    ) 
    
    t300 = thetao_300m.weighted(lev_weights).mean("lev")

    # clean up this singleton dimension
    if "dcpp_init_year" in sst.coords: 
        sst = sst.isel(dcpp_init_year=0)
        t300 = t300.isel(dcpp_init_year=0)

    return sst, t300 


def regrid(da, model_name):
    """
    Regrid data to 5x5 grid using xesmf. 

    Param:
        (xr.DataArray)  data to regrid

    Returns:
        (xr.DataArray)  regridded 
    """

    output_grid = xe.util.grid_global(5, 5)
    start_time = time.time()
    weight_file = os.path.join(SAVE_DIRECTORY, f'grids/{model_name}_to_5x5_bilinear_weights.nc')

    if os.path.exists(weight_file):
        regridder = xe.Regridder(da, output_grid, 'bilinear', weights=weight_file, 
                                ignore_degenerate=True, reuse_weights=True, periodic=True)
    else:
        regridder = xe.Regridder(da, output_grid, 'bilinear', filename=weight_file, 
                                ignore_degenerate=True, reuse_weights=False, periodic=True)

    da_regridded = regridder(da)

    return da_regridded


def add_index_to_lat_lon(da):
    """
    Makes lat and lon indexable. This should be applied after regridding so that lat and lon
    are regular (i.e., the grid is rectilinear). 

    Param:
        (xr.DataArray)  regridded data with lat (x,y) and lon (x,y)
    
    Returns:
        (xr.DataArray)  data with indexed lat lon coordinates
    """

    lat_1d = da['lat'][:, 0] 
    lon_1d = da['lon'][0, :] 

    # Create a new DataArray with updated coordinates
    da = da.assign_coords(
        lat=lat_1d,
        lon=lon_1d
    ).rename({'x': 'lon', 'y': 'lat'})

    da = da.set_index(lat='lat', lon='lon')

    return da


def check_if_data_exists(model, member_id):        
    save_path = os.path.join(SAVE_DIRECTORY, f"cmip/{model}_{member_id}_sst_t300.nc")
    return os.path.exists(save_path)


def calculate_ONI(sst):
    # double check the longitude convention (0 to 360) or (-180 to 180)
    if min(sst.lon) < 0: 
        lon_slice = slice(-170, -120)
    elif max(sst.lon) > 180:
        lon_slice = slice(190, 240)

    # Note that this is not explicitly area averaged
    # the input is already coarsened to 5x5 deg and the averaging region is close to the equator
    # so the error should be relatively small. 
    nino34 = sst.sel(lat=slice(-5,5), lon=lon_slice).mean(dim=("lat", "lon"))

    # calculate ONI via a 3-month moving mean 
    oni = nino34.rolling(time=3, center=True).mean("time")

    return oni 


def main():
    cat, unique_models = get_catalog()

    models_to_download = unique_models

    # the state of downloaded models is stored in a dict
    download_progress_file_path = os.path.join(SAVE_DIRECTORY, f"cmip/download_progress.pkl")
    if os.path.exists(download_progress_file_path):
        with open(download_progress_file_path, 'rb') as file:
            download_progress = pickle.load(file)
    else:
        download_progress = {}
        for model in models_to_download:
            download_progress[model] = False

    # download the models 
    for model in models_to_download:
        # skip if all ensemble members have already been downloaded 
        if download_progress[model]: continue 

        # check concurrent downloads (if running this on multiple nodes)
        if "curr_downloads" not in download_progress.keys():
            download_progress["curr_downloads"] = [model]
        elif model in download_progress["curr_downloads"]:
            print(f"another instance of this script is currently downloading {model}. Skipping...")
            continue
        else:
            download_progress["curr_downloads"].append(model)

        # update the download progress file 
        with open(download_progress_file_path, 'wb') as file:
            pickle.dump(download_progress, file)

        print(f"Preprocessing data for {model}")
        ds = load_data_as_dset(cat, model)

        if ds is None: continue 

        # make sure time coordinate is a standardized format 
        time_coord = pd.date_range("1850-01-01", "2014-12-01", freq="MS")
        if len(ds.time) != 1980: 
            print(f"Skipping {model} because its timeseries has length {len(ds.time)}, not expected length of 1980")
            continue
        else:
            ds = ds.assign_coords(time=time_coord)

        # do each ensemble member separately 
        for member in ds.member_id.values:
            if check_if_data_exists(model, member): 
                print(f"Already found downloaded data for {member}. Skipping... \n")
                continue 

            print(f"Preprocessing ensemble member {member}. Subsetting...", end="")
            ds_subset = ds.sel(member_id=member)
            
            # subset and regrid
            sst, t300 = subset_data(ds_subset)

            sst = regrid(sst, model)
            t300 = regrid(t300, model)

            sst = add_index_to_lat_lon(sst)
            t300 = add_index_to_lat_lon(t300)

            # select the domain from 60S to 60N 
            sst = sst.sel(lat=slice(-60, 60))
            t300 = t300.sel(lat=slice(-60,60))

            # calculate ONI index 
            oni = calculate_ONI(sst)

            # now normalize 
            sst_normalized = sst.groupby("time.month").apply(
                lambda x: (x - x.mean("time")) / x.std("time")
            )

            t300_normalized = t300.groupby("time.month").apply(
                lambda x: (x - x.mean("time")) / x.std("time")
            )

            oni_normalized = oni.groupby("time.month").apply(
                lambda x: (x - x.mean("time")) / x.std("time")
            )

            print("done!")

            # download, compute everything, and save
            print("Downloading and regridding... ")
            start_time = time.time()
            combined_ds = xr.Dataset({"sst": sst_normalized, "t300": t300_normalized, "oni": oni_normalized})
            combined_ds = combined_ds.assign_attrs(model_name=model, member_id=member, regrid_method="bilinear")

            save_path = os.path.join(SAVE_DIRECTORY, f"cmip/{model}_{member}_sst_t300.nc")
            combined_ds.to_netcdf(save_path)
            end_time = time.time()
            elapsed_time = end_time - start_time

            print(f"done! (Time elapsed: {elapsed_time:.2f} seconds)\n")

        download_progress[model] = True
        download_progress["curr_downloads"].remove(model)
        with open(download_progress_file_path, 'wb') as file:
            pickle.dump(download_progress, file)

        print(f"\n\n")


if __name__ == "__main__":
    main()