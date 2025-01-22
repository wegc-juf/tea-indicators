import glob
import math
import numpy as np
import os
from pathlib import Path
import sys
import warnings
import xarray as xr

warnings.filterwarnings(action='ignore', message='All-NaN slice encountered')
warnings.filterwarnings(action='ignore', message='divide by zero encountered in divide')

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from scripts.general_stuff.general_functions import create_history, compare_to_ref, create_tea_history
from scripts.general_stuff.var_attrs import get_attrs
from scripts.general_stuff.TEA_logger import logger
from scripts.calc_indices.calc_daily_basis_vars import calc_daily_basis_vars
from scripts.calc_indices.general_TEA_stuff import assign_ctp_coords
from scripts.calc_indices.TEA_AGR import TEAAgr

import time


def calc_tea_lat(opts, lat, tea_agr, lons):
    if opts.full_region:
        land_frac_min = 0
    else:
        land_frac_min = 0.5
    
    if opts.precip:
        cell_size_lat = 1
    else:
        cell_size_lat = 2
    
    # step through all longitudes
    for ilon, lon in enumerate(lons):
        # this comment is necessary to suppress an unnecessary PyCharm warning for lon
        # noinspection PyTypeChecker
        logger.info(f'Processing lat {lat}, lon {lon}')
        start_time = time.time()
        tea_sub = tea_agr.select_sub_gr(lat=lat, lon=lon, cell_size_lat=cell_size_lat, land_frac_min=land_frac_min)
        if tea_sub is None:
            continue
        
        # calculate daily basis variables
        tea_sub.calc_daily_basis_vars(grid=False)
        # TODO check if this is necessary
        # dbv_results = tea_sub.get_daily_results(gr=True, grid=False).compute()
        # tea_agr.set_dbv_results(lat, lon, dbv_results)
        
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', message='invalid value encountered in multiply')
            # calculate CTP indicators
            tea_sub.calc_annual_CTP_indicators(opts.period, drop_daily_results=True)
            
            # set agr_results for lat and lon
            ctp_results = tea_sub.get_CTP_results(gr=True, grid=False).compute()
            
        tea_agr.set_ctp_results(lat, lon, ctp_results)
        end_time = time.time()
        logger.debug(f'Lat {lat}, lon {lon} processed in {end_time - start_time} seconds')

    
def regrid_data_experimental(data):
    import xarray as xr
    import numpy as np
    from rasterio.enums import Resampling
    import matplotlib.pyplot as plt
    
    data = data.isel(time=0)
    data = data.rio.write_crs('EPSG:4326')
    data = data.rio.set_spatial_dims('lon', 'lat')
    dst = data.rio.reproject('EPSG:3857',
                             # shape=(250, 250),
                             resampling=Resampling.bilinear,
                             nodata=np.nan)
    merc = data.rio.reproject('ESRI:53004',
                              resampling=Resampling.bilinear,
                              nodata=np.nan)
    sinu = data.rio.reproject('ESRI:54008',
                              resampling=Resampling.bilinear,
                              nodata=np.nan)
    pass


def calc_tea_large_gr(opts, data, masks, static):
    logger.info(f'Switching to calc_TEA_largeGR because GR > 100 areals.')
    
    # preselect region to reduce computation time (incl. some margins to avoid boundary effects)
    if opts.full_region:
        min_lat = masks.lat.min().values
        max_lat = masks.lat.max().values
    else:
        min_lat = data.lat[np.where(masks['lt1500_mask'] > 0)[0][-1]].values - 2
        if min_lat < static.area_grid.lat.min().values:
            min_lat = float(static.area_grid.lat.min().values)
        max_lat = data.lat[np.where(masks['lt1500_mask'] > 0)[0][0]].values + 2
        if max_lat > static.area_grid.lat.max().values:
            max_lat = float(static.area_grid.lat.max().values)
        if min_lat < 35:
            min_lat = 35
    if opts.dataset == 'ERA5' and opts.region == 'EUR':
        lons = np.arange(-12, 40.5, 0.5)
    else:
        lons = np.arange(9, 18, 0.5)
    min_lon = lons[0]
    max_lon = lons[-1]
    proc_data = data.sel(lat=slice(max_lat, min_lat), lon=slice(min_lon, max_lon))
    # regrid_data(proc_data)
    land_sea_mask = masks['valid_cells'].sel(lat=slice(max_lat, min_lat), lon=slice(min_lon, max_lon))
    full_mask = masks['lt1500_mask'].sel(lat=slice(max_lat, min_lat), lon=slice(min_lon, max_lon))
    proc_static = static.sel(lat=slice(max_lat, min_lat), lon=slice(min_lon, max_lon))
    
    # load agr mask
    try:
        agr_mask = xr.open_dataset(f'{opts.statpath}{opts.region}_mask_0p5_{opts.dataset}.nc')
        agr_mask = agr_mask.mask_lt1500
    except FileNotFoundError:
        agr_mask = None
    
    tea_agr = TEAAgr(input_data_grid=proc_data, threshold_grid=proc_static['threshold'],
                     area_grid=proc_static['area_grid'], mask=full_mask, min_area=1, land_sea_mask=land_sea_mask,
                     agr_mask=agr_mask)
    tea_agr.calc_daily_basis_vars()

    # define latitudes with 0.5° resolution for output
    lats = np.arange(math.ceil(min_lat / .5) / 2, math.ceil(max_lat) + 0.5, 0.5)
    
    # for testing with only one latitude or debugging
    if False:
        calc_tea_lat(opts=opts, lat=47., tea_agr=tea_agr, lons=lons)
    else:
        for llat in lats:
            calc_tea_lat(opts=opts, lat=llat, tea_agr=tea_agr, lons=lons)
    
    # save output files
    # TODO do we need these results?
    # dbv_path = (f'{opts.outpath}/daily_basis_variables/'
    #             f'DBV_{opts.param_str}_{opts.region}_{opts.period}_{opts.dataset}'
    #             f'_{opts.start}to{opts.end}.nc')
    # create_tea_history(cli_params=sys.argv, tea=tea_agr, result_type='dbv_agr')
    # tea_agr.save_dbv_results(dbv_path)
    
    ctp_path = (f'{opts.outpath}/ctp_indicator_variables/'
                f'CTP_{opts.param_str}_{opts.region}_{opts.period}_{opts.dataset}'
                f'_{opts.start}to{opts.end}.nc')
    create_tea_history(cli_params=sys.argv, tea=tea_agr, result_type='ctp_agr')
    tea_agr.apply_mask()
    # TODO: check if var attributes are correct
    tea_agr.save_ctp_results(ctp_path)
    
    ref_path = ctp_path.replace('.nc', '_ref.nc')
    ref_data = xr.open_dataset(ref_path)
    ref_data = ref_data.sel(lat=47., lon=slice(13, 16))
    ref_data = ref_data.rename({'ctp': 'time'})
    
    ref_data = ref_data.rename({vvar: vvar.replace('avg', '_avg') for vvar in ref_data.data_vars if 'avg' in vvar})
    
    compare_data = tea_agr.get_ctp_results().sel(lat=47., lon=slice(13, 16))
    compare_to_ref(compare_data, ref_data, relative=True)
    del tea_agr
    