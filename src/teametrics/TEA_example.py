#!/usr/bin/env python3
"""
Example for using the TEA class
"""
import os
import requests

import xarray as xr
import matplotlib.pyplot as plt
from argparse import ArgumentParser

from teametrics.TEA import TEAIndicators

THRESHOLD = 28  # degC


def get_opts():
    """
    Get command line options.
    Returns:
        opts: Command line options.
    """
    parser = ArgumentParser(description='TEA example script.')
    
    parser.add_argument('--no-gui', dest='no_gui', action='store_true',
                        help='Run without displaying the plots on screen (default: False)')
    
    opts = parser.parse_args()
    
    return opts


def get_binary_file(filename,
                    url="https://cloud.uni-graz.at/s/xrN7pSkMPZfcJiE/download",
                    cache_dir="~/.my_package_data") -> str:
    """
    Ensures the binary file is available locally.
    Downloads it if missing.

    Parameters
    ----------
    url : str
        Direct download URL for the file.
    filename : str
        Name to save the file as.
    cache_dir : str
        Directory to store the file (default: ~/.my_package_data).

    Returns
    -------
    str
        Path to the local file.
    """
    cache_dir = os.path.expanduser(cache_dir)
    os.makedirs(cache_dir, exist_ok=True)

    file_path = os.path.join(cache_dir, filename)

    if not os.path.exists(file_path):
        print(f"Downloading {filename} ...")
        resp = requests.get(url, stream=True)
        resp.raise_for_status()

        with open(file_path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)

        print(f"Saved to {file_path}")
    else:
        print(f"Using cached file at {file_path}")

    return file_path


def dl_example_file() -> tuple[str, str]:
    ERA5_file = 'ERA5_Tx_1956-2024_CH.nc'
    example_path = get_binary_file(filename=ERA5_file)
    print(f"Binary ready at: {example_path}")
    return ERA5_file, example_path


def run():
    """
    Run the TEA example.
    Returns:

    """
    opts = get_opts()
    
    # download example file
    ERA5_file, example_path = dl_example_file()
    
    # load example data (ERA5 Switzerland, 1956-2024, daily maximum temperature)
    input_data = xr.open_dataset(example_path)
    
    # create TEA object; set threshold to THRESHOLD degC
    tea_obj = TEAIndicators(input_data=input_data.Tx, threshold=THRESHOLD, unit='degC')
    
    # calculate daily TEA indicators and save to NetCDF file
    print('Calculating daily TEA indicators...')
    tea_obj.calc_daily_basis_vars()
    ERA5_basename = ERA5_file.split('.')[0]
    if not os.path.exists('./tea_examples'):
        os.makedirs('./tea_examples')
    outpath = f'./tea_examples/{ERA5_basename}_TEA_daily_results.nc'
    print(f'Saving daily results to {outpath}...')
    tea_obj.save_daily_results(outpath)
    
    # plot Daily Threshold Exceedance Magnitude (DTEM) for 2024-08-30
    tea_obj.daily_results.DTEM.sel(time='2024-08-30').plot()
    if not opts.no_gui:
        plt.show()
    
    # calculate annual TEA indicators for warm season (WAS) and save to NetCDF file
    print('Calculating annual TEA indicators...')
    tea_obj.calc_annual_ctp_indicators(ctp='WAS', drop_daily_results=True)
    outpath = f'./tea_examples/{ERA5_basename}_TEA_annual_results.nc'
    print(f'Saving annual results to {outpath}...')
    tea_obj.save_ctp_results(outpath)
    
    # plot cumulative exceedance magnitude (temporal event extremity tEX) for 2024
    tea_obj.ctp_results.EM.sel(time='2024').plot()
    if not opts.no_gui:
        plt.show()
    
    # calculate decadal-mean TEA indicators and save to NetCDF file
    print('Calculating decadal-mean TEA indicators...')
    tea_obj.calc_decadal_indicators(calc_spread=True, drop_annual_results=True, )
    outpath = f'./tea_examples/{ERA5_basename}_TEA_decadal_results.nc'
    print(f'Saving decadal results to {outpath}...')
    tea_obj.save_decadal_results(outpath)
    
    # plot decadal-mean exceedance magnitude (EM) for 2010s
    tea_obj.decadal_results.EM.sel(time='2014').plot()
    if not opts.no_gui:
        plt.show()
    
    # calculate amplification factors and save to NetCDF file
    print('Calculating amplification factors...')
    tea_obj.calc_amplification_factors()
    outpath = f'./tea_examples/{ERA5_basename}_TEA_amplification_factors.nc'
    print(f'Saving amplification factors to {outpath}...')
    tea_obj.save_amplification_factors(outpath)
    
    # plot amplification factor for exceedance magnitude (EM) for current climate period (CC=2008-2024)
    tea_obj.amplification_factors.EM_AF_CC.plot()
    if not opts.no_gui:
        plt.show()


if __name__ == '__main__':
    run()
    
