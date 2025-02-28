"""
Example for using the TEA class
"""
import xarray as xr
import matplotlib.pyplot as plt

from TEA import TEAIndicators

THRESHOLD = 28  # degC

if __name__ == '__main__':
    
    # load example data (ERA5 Switzerland, 1956-2024, daily maximum temperature)
    ERA5_file = 'ERA5_Tx_1956-2024_CH.nc'
    input_data = xr.open_dataset(f'../examples/{ERA5_file}')
    
    # create TEA object; set threshold to THRESHOLD degC
    tea_obj = TEAIndicators(input_data_grid=input_data.Tx, threshold=THRESHOLD, unit='degC')
    
    # calculate daily TEA indicators and save to NetCDF file
    tea_obj.calc_daily_basis_vars()
    
    ERA5_basename = ERA5_file.split('.')[0]
    outpath = f'../examples/{ERA5_basename}_TEA_daily_results.nc'
    tea_obj.save_daily_results(outpath)
    
    # plot Daily Threshold Exceedance Magnitude (DTEM) for 2024-08-30
    tea_obj.daily_results.DTEM.sel(time='2024-08-30').plot()
    plt.show()
