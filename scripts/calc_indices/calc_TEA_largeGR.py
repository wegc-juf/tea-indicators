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
from scripts.general_stuff.general_functions import compare_to_ref, create_tea_history, create_history_from_cfg
from scripts.general_stuff.var_attrs import get_attrs
from scripts.general_stuff.TEA_logger import logger
from scripts.calc_indices.calc_daily_basis_vars import calc_daily_basis_vars
from scripts.calc_indices.general_TEA_stuff import assign_ctp_coords
from scripts.calc_indices.TEA_AGR import TEAAgr

import time


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
    
    if opts.precip:
        cell_size_lat = 1
    else:
        cell_size_lat = 2
    
    # preselect region to reduce computation time (incl. some margins to avoid boundary effects)
    if opts.full_region:
        land_frac_min = 0
        min_lat = masks.lat.min().values
        max_lat = masks.lat.max().values
    else:
        land_frac_min = 0.5
        min_lat = data.lat[np.where(masks['lt1500_mask'] > 0)[0][-1]].values - cell_size_lat
        if min_lat < static.area_grid.lat.min().values:
            min_lat = float(static.area_grid.lat.min().values)
        max_lat = data.lat[np.where(masks['lt1500_mask'] > 0)[0][0]].values + cell_size_lat
        if max_lat > static.area_grid.lat.max().values:
            max_lat = float(static.area_grid.lat.max().values)
        if min_lat < 35 - cell_size_lat:
            min_lat = 35 - cell_size_lat
    if opts.dataset == 'ERA5' and opts.region == 'EUR':
        lons = np.arange(-12, 40.5, 0.5)
    else:
        lons = np.arange(9, 18, 0.5)
    min_lon = lons[0] - cell_size_lat
    max_lon = lons[-1] + cell_size_lat
    proc_data = data.sel(lat=slice(max_lat, min_lat), lon=slice(min_lon, max_lon))
    # regrid_data(proc_data)
    land_sea_mask = masks['valid_cells'].sel(lat=slice(max_lat, min_lat), lon=slice(min_lon, max_lon))
    full_mask = masks['lt1500_mask'].sel(lat=slice(max_lat, min_lat), lon=slice(min_lon, max_lon))
    proc_static = static.sel(lat=slice(max_lat, min_lat), lon=slice(min_lon, max_lon))
    
    # load agr mask
    try:
        agr_mask = xr.open_dataset(f'{opts.maskpath}/{opts.region}_mask_0p5_{opts.dataset}.nc')
        agr_mask = agr_mask.mask_lt1500
    except FileNotFoundError:
        agr_mask = None
    
    tea_agr = TEAAgr(input_data_grid=proc_data, threshold_grid=proc_static['threshold'],
                     area_grid=proc_static['area_grid'], mask=full_mask, min_area=1, land_sea_mask=land_sea_mask,
                     agr_mask=agr_mask, land_frac_min=land_frac_min, cell_size_lat=cell_size_lat, ctp=opts.period,
                     unit=opts.unit, low_extreme=opts.low_extreme)
    
    if agr_mask is None:
        save_0p5_mask(opts, tea_agr.agr_mask, tea_agr.agr_area)
    
    tea_agr.calc_daily_basis_vars()

    # for testing with only one latitude or debugging
    if False:
        lons = [37]
        lat = 51
        tea_agr.calc_tea_agr(lats=[lat], lons=lons)
        res = tea_agr.get_ctp_results()
        res = res.sel(lat=lat, lon=slice(lons[0], lons[-1]))
        logger.info(res)
    else:
        tea_agr.calc_tea_agr()
    
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
    create_tea_history(cfg_params=opts, tea=tea_agr, result_type='CTP')
    tea_agr.apply_mask()
    # TODO: check if var attributes are correct
    tea_agr.save_ctp_results(ctp_path)
    
    if opts.compare_to_ref:
        ref_path = ctp_path.replace('.nc', '_ref.nc')
        ref_data = xr.open_dataset(ref_path)
        if 'ctp' in ref_data.coords:
            ref_data = ref_data.rename({'ctp': 'time'})
            ref_data = ref_data.rename({vvar: vvar.replace('avg', '_avg') for vvar in ref_data.data_vars if 'avg' in vvar})
        
        compare_data = tea_agr.get_ctp_results()
        compare_to_ref(compare_data, ref_data, relative=True)
    return tea_agr


def save_0p5_mask(opts, mask_0p5, area_0p5):
    """
    save mask on 0.5° grid to netcdf file
    Args:
        opts: CLI parameter
        mask_0p5: mask on 0.5° grid
        area_0p5: area grid on 0.5° grid
    """
    area_0p5 = create_history_from_cfg(cfg_params=opts, ds=area_0p5)
    try:
        area_0p5.to_netcdf(f'{opts.statpath}/area_grid_0p5_{opts.region}_{opts.dataset}.nc')
    except PermissionError:
        area_0p5.to_netcdf(f'{opts.outpath}/area_grid_0p5_{opts.region}_{opts.dataset}.nc')

    # save 0.5° mask
    mask_0p5 = create_history_from_cfg(cfg_params=opts, ds=mask_0p5)
    try:
        mask_0p5.to_netcdf(f'{opts.maskpath}/{opts.region}_mask_0p5_{opts.dataset}.nc')
    except PermissionError:
        mask_0p5.to_netcdf(f'{opts.outpath}/{opts.region}_mask_0p5_{opts.dataset}.nc')
