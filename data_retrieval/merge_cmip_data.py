import xarray as xr
import numpy as np 
import os
import sys

from data_retrieval.data_config import DATA_DIR


save_dir = os.path.join(DATA_DIR, "merged_data")

godas = xr.open_dataset(os.path.join(DATA_DIR, "reanalysis/godas/regridded/GODAS_regridded.nc"))
soda = xr.open_dataset(os.path.join(DATA_DIR, "reanalysis/soda/regridded/SODA_regridded.nc"))

def merge_cmip():
    """
    Merges regridded CMIP files (saved separately for different ensemble members/models)
    into a single netCDF file. Creates a new dimension called `simulation_id` that has
    format model_name:member_id. 

    Computes a land mask that unions over all data (including SODA and GODAS) so that the
    underlying grid is the same. Also, by convention the longitude ranges from -180 to 180

    Returns:
        (xr.Dataset)    merged_ds
        (xr.Dataset)    land_mask 
    """

    cmip_path = os.path.join(DATA_DIR, "cmip")
    cmip_files = os.listdir(cmip_path)
    cmip_files.remove("download_progress.pkl")

    cmip_ds_list = []
    land_mask = np.logical_or(np.isnan(soda.sst.isel(time=0)), np.isnan(godas.sst.isel(time=0)))

    for f in sorted(cmip_files):
        filepath = os.path.join(cmip_path, f)
        try:
            ds = xr.open_dataset(filepath, chunks={})
            
            # Assign model_name and member_id as coordinates
            model_name = ds.attrs.get("model_name", "unknown_model")
            member_id = ds.attrs.get("member_id", "unknown_member")
            simulation_id = f"{model_name}:{member_id}"
            ds = ds.assign_coords(simulation_id=simulation_id)

            # drop attributes and non-essential singleton vars 
            ds = ds.drop_attrs()
            for coord in ds.coords:
                if coord not in ["time", "lon", "lat", "simulation_id"]:
                    ds = ds.drop_vars(coord)
            
            if np.all(np.isnan(ds.sst.isel(time=0))):
                print(f"{simulation_id} has missing sst, skipping.")
                continue

            if np.all(np.isnan(ds.t300.isel(time=0))):
                print(f"{simulation_id} has missing t300, skipping.")
                continue
            
            # check for a weird regridding bug (TODO: fix the regridding for these models)
            # there are four models (CNRM and CMCC) that are affected
            regrid_bug_exists = np.all(np.isnan(ds.sst.isel(time=0).sel(lon=72.5)))
            if regrid_bug_exists:
                print(f"Found known regridding bug for {simulation_id}, skipping")
                continue

            cmip_ds_list.append(ds)

            # update the land mask. This is a union over all models 
            land_mask = np.logical_or(land_mask, np.isnan(ds.sst.isel(time=0))) 

        except Exception as e:
            # Log the error and skip the file
            print(f"Error opening file {filepath}: {e}")
            continue
    
    print("\n\nMerging cmip_ds_list...")
    merged_ds = xr.concat(cmip_ds_list, dim="simulation_id")
    merged_ds["sst"] = merged_ds["sst"].where(~land_mask)
    merged_ds["t300"] = merged_ds["t300"].where(~land_mask)

    land_mask = land_mask.to_dataset(name="land_mask")

    return merged_ds, land_mask

def main():
    os.makedirs(save_dir, exist_ok=True)
    if os.path.exists(os.path.join(save_dir, "CMIP_merged.nc")): 
        print("Already found existing merged CMIP file. Exiting.")
        return 

    merged_ds, land_mask = merge_cmip()

    print(f"Saving merged CMIP data and land_mask to {save_dir}...")
    
    merged_ds.to_netcdf(os.path.join(save_dir, "CMIP_merged.nc"))
    land_mask.to_netcdf(os.path.join(save_dir, "land_mask.nc"))
    print("done!")

    # Save new SODA and GODAS files since we changed the land mask 
    print(f"Saving SODA and GODAS with updated land mask to {save_dir}...")
    soda["sst"] = soda["sst"].where(~land_mask.land_mask)
    soda["t300"] = soda["t300"].where(~land_mask.land_mask)
    godas["sst"] = godas["sst"].where(~land_mask.land_mask)
    godas["t300"] = godas["t300"].where(~land_mask.land_mask)

    soda.to_netcdf(os.path.join(save_dir, "SODA.nc"))
    godas.to_netcdf(os.path.join(save_dir, "GODAS.nc"))
    print("all done!")

if __name__ == "__main__":
    main()
