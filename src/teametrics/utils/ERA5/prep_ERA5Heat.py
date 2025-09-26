import glob
import numpy as np
import os
import xarray as xr

def combine_daily_files():
    directory = '/data/arsclisys/normal/clim-hydro/TEA-Indicators/ERA5Heat/'

    zip_files = sorted(glob.glob(os.path.join(directory, 'raw', '*')))

    for file in zip_files:
        if not file.endswith('.zip'):
            continue

        # find year in the filename
        year = file.split('/')[-1].split('_')[1][:4]

        # Extract the zip file
        os.system(f'unzip {file} -d {os.path.join(directory, "raw/tmp")}')

        # combine all extracted files
        nc_files = sorted(glob.glob(os.path.join(directory, 'raw/tmp', '*.nc')))

        os.system(
            f'cdo mergetime {os.path.join(directory, "raw/tmp", f"*.nc")} '
            f'{os.path.join(directory, f"ERA5Heat_{year}.nc")}')

        # emtpy the temporary directory
        for tmp_file in nc_files:
            os.remove(tmp_file)


def reduce_file_size():
    """
    reduces the size of the ERA5Heat
    Returns:

    """
    directory = '/data/arsclisys/normal/clim-hydro/TEA-Indicators/ERA5Heat/'
    files = sorted(glob.glob(os.path.join(directory, '*')))

    for file in files:
        fname = file.split('/')[-1]
        print(fname)
        ds = xr.open_dataset(file)

        ds.to_netcdf(os.path.join(directory, 'compressed', fname),
                     encoding={'utci': {'dtype': 'float32', 'zlib': True, 'complevel': 5}},)


if __name__ == "__main__":
    # combine_daily_files()
    reduce_file_size()
