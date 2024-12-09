import os
import requests
import xarray as xr
import numpy as np 
import xesmf as xe
import config

def download_SODA_data(base_url, start_year, end_year, spatial_dims, output_dir):
    """
    Downloads SODA data year by year from ERDDAP dataset. 

    Param:
    (string)    base_url: The base URL of the dataset (excluding the time range and spatial dimensions).
    (int)       start_year: Start year of the dataset.
    (int)       end_year: End year of the dataset.
    (string)    spatial_dims: Spatial dimensions string for the query (e.g., "[5.01:1:350][(-60):1:60][(0.25):1:359.75]").
    (string)    output_dir: Directory to save the downloaded files.

    Returns:
    If everything already downloaded, returns None 
    """
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    existing_files = os.listdir(output_dir)
    if f"SODA_thetao_upper300m_{end_year}.nc" in existing_files:
        print("Already found all files. Skipping... \n\n")
        return None
    
    for year in range(start_year, end_year + 1):
        time_range = f"[({year}-01-15):1:({year}-12-15)]"
        
        full_url = f"{base_url}{time_range}{spatial_dims}"
        
        output_file = os.path.join(output_dir, f"SODA_thetao_upper300m_{year}.nc")
        
        if os.path.exists(output_file):
            print(f"File already exists, skipping download: {output_file}")
            continue
        
        # Download the file
        try:
            print(f"Downloading SODA data for {year}...")
            response = requests.get(full_url, stream=True)
            response.raise_for_status()
            
            # Save the file in chunks 
            with open(output_file, "wb") as file:
                for chunk in response.iter_content(chunk_size=8192):
                    file.write(chunk)
            print(f"Saved: {output_file}")
        except requests.exceptions.RequestException as e:
            print(f"Failed to download data for {year}: {e}")


def download_GODAS_data(base_url, start_year, end_year, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    existing_files = os.listdir(output_dir)
    if f"GODAS_thetao_{end_year}.nc" in existing_files:
        print("Already found all files. Skipping... \n\n")
        return None
    
    for year in range(start_year, end_year + 1):        
        full_url = f"{base_url}.{year}.nc"
        
        output_file = os.path.join(output_dir, f"GODAS_thetao_{year}.nc")
        
        if os.path.exists(output_file):
            print(f"File already exists, skipping download: {output_file}")
            continue
        
        # Download the file
        try:
            print(f"Downloading GODAS data for {year}...")
            response = requests.get(full_url, stream=True)
            response.raise_for_status()
            
            # Save the file in chunks 
            with open(output_file, "wb") as file:
                for chunk in response.iter_content(chunk_size=8192):
                    file.write(chunk)
            print(f"Saved: {output_file}")
        except requests.exceptions.RequestException as e:
            print(f"Failed to download data for {year}: {e}")


def subset_data(da): 
    """
    From thetao (potential temperature), subset the SST (defined as the topmost depth)
    and T300 (weighted-average of top 300 m). 

    Note that xarray does lazy evaluation so no computation is actually done when you 
    call this. 

    Param:
        (xr.DataArray)  DataArray of ocean potential temperature 

    Returns:
        (xr.DataArray)  sst
        (xr.DataArray)  t300 
    """
    
    # get sst field 
    sst = da.isel(lev=0)

    # get top 300m temperature 
    thetao_300m = da.sel(lev=slice(0,300))

    # calculate how thick each grid cell is to take the weighted mean over depth
    lev_midpoints = np.append([0], (da.lev[1:].values + da.lev[:-1].values) / 2)
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

    return sst, t300 


def calculate_ONI(sst):
    # double check the longitude convention (0 to 360) or (-180 to 180)
    if min(sst.lon) < 0: 
        lon_slice = slice(-170, -120)
    elif max(sst.lon) > 180:
        lon_slice = slice(190, 240)

    # calculate sst anomaly
    ssta = sst.groupby("time.month").apply(
        lambda x: x - x.mean("time")
    )

    # Note that this is not explicitly area averaged
    # the input is already coarsened to 5x5 deg and the averaging region is close to the equator
    # so the error should be relatively small. 
    nino34 = ssta.sel(lat=slice(-5,5), lon=lon_slice).mean(dim=("lat", "lon"))

    # calculate ONI via a 3-month moving mean 
    oni = nino34.rolling(time=3, center=True).mean("time")

    return oni 


def regrid(da, model_name, save_dir):
    """
    Regrid data to 5x5 grid using xesmf. Saves the weight file to save_dir

    Param:
        (xr.DataArray)  data to regrid
        (string)        model_name: the name of the input grid 
        (string)        save_dir: the path to save the regridding weights 

    Returns:
        (xr.DataArray)  regridded data
    """

    output_grid = xe.util.grid_global(5, 5)
    os.makedirs(os.path.join(save_dir, "grids"), exist_ok=True)
    weight_file = os.path.join(save_dir, f'grids/{model_name}_to_5x5_bilinear_weights.nc')

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


def preprocess_SODA_data(save_dir, start_year, end_year): 
    os.makedirs(save_dir, exist_ok=True) 

    save_path = os.path.join(save_dir, "SODA_regridded.nc")
    if os.path.exists(save_path):
        print(f"Already found regridded file at {save_path}. Skipping... \n\n")
        return None

    sst_list = []
    t300_list = []
    for year in range(start_year, end_year+1):
        ds = xr.open_dataset(f"{config.DATA_DIR}/reanalysis/soda/SODA_thetao_upper300m_{year}.nc")
        da = ds["temp"]

        # rename the coordinates to standard lev/lat/lon
        da = da.rename({"LEV": "lev", 
                        "latitude": "lat",
                        "longitude": "lon"})
        
        sst, t300 = subset_data(da) 

        sst_list.append(regrid(sst, "SODA", save_dir))
        t300_list.append(regrid(t300, "SODA", save_dir))
        print(f"{year} regridding done!")

    print("normalizing and merging...")
    # merge 
    sst = xr.concat(sst_list, dim="time")
    t300 = xr.concat(t300_list, dim="time")

    sst = add_index_to_lat_lon(sst)
    t300 = add_index_to_lat_lon(t300)

    # select the domain from 60S to 60N 
    sst = sst.sel(lat=slice(-60, 60))
    t300 = t300.sel(lat=slice(-60, 60))

    # calculate oni 
    oni = calculate_ONI(sst)

    # normalize 
    sst_normalized = sst.groupby("time.month").apply(
        lambda x: (x - x.mean("time")) / x.std("time")
    )

    t300_normalized = t300.groupby("time.month").apply(
        lambda x: (x - x.mean("time")) / x.std("time")
    )

    # create the dataset and add some metadata
    combined_ds = xr.Dataset({"sst": sst_normalized, "t300": t300_normalized, "oni": oni})
    combined_ds.sst.assign_attrs(units="stdev (normalized)")
    combined_ds.t300.assign_attrs(units="stdev (normalized)")
    combined_ds.oni.assign_attrs(units="˚C")
    combined_ds.assign_attrs(regrid_method="bilinear", 
                            source="https://apdrc.soest.hawaii.edu/erddap/griddap/hawaii_soest_c71f_e12b_37f8.html")

    # save
    combined_ds.to_netcdf(save_path)

    print("done! \n\n")


def preprocess_GODAS_data(save_dir, start_year, end_year):
    os.makedirs(save_dir, exist_ok=True) 

    save_path = os.path.join(save_dir, "GODAS_regridded.nc")
    if os.path.exists(save_path):
        print(f"Already found regridded file at {save_path}. Skipping... \n\n")
        return None

    sst_list = []
    t300_list = []
    for year in range(start_year, end_year+1):
        ds = xr.open_dataset(f"{config.DATA_DIR}/reanalysis/godas/GODAS_thetao_{year}.nc")
        da = ds["pottmp"]

        # rename the coordinates to standard lev/lat/lon
        da = da.rename({"level": "lev"})
        
        sst, t300 = subset_data(da) 

        sst_list.append(regrid(sst, "GODAS", save_dir))
        t300_list.append(regrid(t300, "GODAS", save_dir))
        print(f"{year} regridding done!")

    print("normalizing and merging... ")

    # merge 
    sst = xr.concat(sst_list, dim="time")
    t300 = xr.concat(t300_list, dim="time")

    sst = add_index_to_lat_lon(sst)
    t300 = add_index_to_lat_lon(t300)

    # select the domain from 60S to 60N 
    sst = sst.sel(lat=slice(-60, 60))
    t300 = t300.sel(lat=slice(-60, 60))

    # save 
    oni = calculate_ONI(sst)

    # normalize 
    sst_normalized = sst.groupby("time.month").apply(
        lambda x: (x - x.mean("time")) / x.std("time")
    )

    t300_normalized = t300.groupby("time.month").apply(
        lambda x: (x - x.mean("time")) / x.std("time")
    )

    # create the dataset and add some metadata
    combined_ds = xr.Dataset({"sst": sst_normalized, "t300": t300_normalized, "oni": oni})
    combined_ds.sst.assign_attrs(units="stdev (normalized)")
    combined_ds.t300.assign_attrs(units="stdev (normalized)")
    combined_ds.oni.assign_attrs(units="˚C")
    combined_ds.assign_attrs(regrid_method="bilinear", 
                            source="https://psl.noaa.gov/data/gridded/data.godas.html")

    # save
    combined_ds.to_netcdf(save_path)

    print("done! \n\n")


def main():
    print("Downloading SODA data...")
    # Specify the output directory
    soda_output_dir = f"{config.DATA_DIR}/reanalysis/soda"
    soda_base_url = "https://apdrc.soest.hawaii.edu/erddap/griddap/hawaii_soest_c71f_e12b_37f8.nc?temp"
    spatial_dims = "[(5.01):1:(350)][(-60):1:(60)][(0.25):1:(359.75)]"

    # SODA 2.2.4 time span 
    start_year = 1871
    end_year = 1973

    download_SODA_data(soda_base_url, start_year, end_year, spatial_dims, soda_output_dir)
    
    print("Regridding and saving SODA data...")
    soda_regridded_dir = f"{config.DATA_DIR}/reanalysis/soda/regridded"
    preprocess_SODA_data(soda_regridded_dir, start_year, end_year) 



    print("Downloading GODAS data...")
    godas_output_dir = f"{config.DATA_DIR}/reanalysis/godas"
    godas_base_url = "https://psl.noaa.gov/thredds/fileServer/Datasets/godas/pottmp"
    
    # GODAS time span 
    start_year = 1980
    end_year = 2024

    download_GODAS_data(godas_base_url, start_year, end_year, godas_output_dir)

    print("Regridding and saving GODAS data...")
    godas_regridded_dir = f"{config.DATA_DIR}/reanalysis/godas/regridded"
    preprocess_GODAS_data(godas_regridded_dir, start_year, end_year) 




if __name__ == "__main__":
    main()