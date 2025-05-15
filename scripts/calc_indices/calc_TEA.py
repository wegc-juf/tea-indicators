#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: hst
"""

import argparse
import gc
import glob
import numpy as np
import os
import pandas as pd
from pathlib import Path
import re
import sys
import math
import warnings
import xarray as xr
import yaml
from copy import deepcopy

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from scripts.general_stuff.general_functions import (create_history_from_cfg, create_tea_history, load_opts,
                                                     compare_to_ref, get_data)
from scripts.general_stuff.TEA_logger import logger
from scripts.calc_indices.calc_decadal_indicators import calc_decadal_indicators, calc_amplification_factors
import scripts.calc_indices.calc_TEA_largeGR as largeGR
from scripts.calc_indices.TEA import TEAIndicators
from scripts.calc_indices.TEA_AGR import TEAAgr
from scripts.data_prep.create_static_files import create_threshold_grid


def load_static_files(opts, large_gr=False):
    """
    # TODO: make this function obsolete
    load GR masks and static file
    Args:
        opts: CLI parameter
        large_gr: set for large GR (> 100 areals)

    Returns:
        masks: GR masks (ds)
        static: ds with threshold, area_grid, etc.

    """

    if opts.full_region:
        full_str = '_full'
    else:
        full_str = ''
    masks = xr.open_dataset(f'{opts.maskpath}/{opts.mask_sub}/{opts.region}_masks_{opts.dataset}.nc')

    if 'LSM_EUR' in masks.data_vars:
        valid_cells = masks['lt1500_mask_EUR'].where(masks['LSM_EUR'].notnull())
        valid_cells = valid_cells.rename('valid_cells')
        masks['valid_cells'] = valid_cells
    elif opts.region == 'EUR':
        valid_cells = masks['lt1500_mask'].copy()
        valid_cells = valid_cells.rename('valid_cells')
        masks['valid_cells'] = valid_cells

    if large_gr:
        region = 'EUR'
    else:
        region = opts.region
    static = xr.open_dataset(
        f'{opts.statpath}static_{opts.param_str}_{region}_{opts.dataset}{full_str}.nc')

    return masks, static


def load_mask_file(opts):
    """
    load GR mask
    Args:
        opts: options

    Returns:
        mask: GR mask (ds)

    """
    maskpath = f'{opts.maskpath}/{opts.mask_sub}/{opts.region}_mask_{opts.dataset}.nc'
    logger.info(f'Loading mask from {maskpath}')
    mask_file = xr.open_dataset(maskpath)
    
    return mask_file.mask


def load_lsm_file(opts):
    """
    load land-sea-mask for AGR
    Args:
        opts: options

    Returns:
        mask: GR mask (ds)

    """
    # TODO: make this work outside of EUR
    new_opts = deepcopy(opts)
    new_opts.region = 'EUR'
    return load_mask_file(new_opts)


def compare_to_ctp_ref(tea, ctp_filename_ref):
    """
    compare results to reference file
    TODO: move this to test routine
    Args:
        tea: TEA object
        ctp_filename_ref: reference file
    """
    
    if os.path.exists(ctp_filename_ref):
        logger.info(f'Comparing results to reference file {ctp_filename_ref}')
        tea_ref = TEAIndicators()
        tea_ref.load_ctp_results(ctp_filename_ref)
        tea_result = tea.ctp_results
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="invalid value encountered in multiply")
            compare_to_ref(tea_result, tea_ref.ctp_results)
    else:
        logger.warning(f'Reference file {ctp_filename_ref} not found.')
    
    
def save_ctp_output(opts, tea, start, end):
    """
    save CTP results to netcdf file
    Args:
        opts: CLI parameters
        tea: TEA object
        start: start year
        end: end year
    """
    create_tea_history(cfg_params=opts, tea=tea, result_type='ctp')

    path = Path(f'{opts.outpath}/ctp_indicator_variables/')
    path.mkdir(parents=True, exist_ok=True)
    
    if 'agr' in opts:
        grg_str = 'GRG-'
    else:
        grg_str = ''
    
    outpath = (f'{opts.outpath}/ctp_indicator_variables/'
               f'CTP_{opts.param_str}_{grg_str}{opts.region}_{opts.period}_{opts.dataset}'
               f'_{start}to{end}.nc')
    
    path_ref = outpath.replace('.nc', '_ref.nc')
    
    logger.info(f'Saving CTP indicators to {outpath}')
    tea.save_ctp_results(outpath)
    
    if opts.compare_to_ref:
        compare_to_ctp_ref(tea, path_ref)
        
        
def _save_0p5_mask(opts, mask_0p5, area_0p5):
    """
    save mask on 0.5° grid to netcdf file
    Args:
        opts: CLI parameter
        mask_0p5: mask on 0.5° grid
        area_0p5: area grid on 0.5° grid
    """
    area_0p5 = create_history_from_cfg(cfg_params=opts, ds=area_0p5)
    area_grid_file = f'{opts.statpath}/area_grid_0p5_{opts.region}_{opts.dataset}.nc'
    try:
        area_0p5.to_netcdf(area_grid_file)
    except PermissionError:
        os.remove(area_grid_file)
        area_0p5.to_netcdf(area_grid_file)
    
    # save 0.5° mask
    mask_0p5 = create_history_from_cfg(cfg_params=opts, ds=mask_0p5)
    mask_file = f'{opts.maskpath}/{opts.mask_sub}/{opts.region}_mask_0p5_{opts.dataset}.nc'
    try:
        mask_0p5.to_netcdf(mask_file)
    except PermissionError:
        os.remove(mask_file)
        mask_0p5.to_netcdf(mask_file)


def calc_ctp_indicators(tea, opts, start, end):
    """
    calculate the TEA indicators for the annual climatic time period
    Args:
        tea: TEA object
        opts: CLI parameter
        start: start year
        end: end year
        lsm: land-sea mask for AGR (optional)
    """
    
    # apply criterion that DTEA_GR > DTEA_min and all GR variables use same dates,
    # dtea_min is given in areals (1 areal = 100 km2)
    dtea_min = 1  # according to equation 03
    tea.update_min_area(dtea_min)
    
    if 'agr' in opts:
        # set cell_size
        tea.cell_size_lat = opts.agr_cell_size
        
        # load static GR grid files
        # TODO: set path and put load function in TEA_AGR
        gr_grid_mask, gr_grid_areas = load_gr_grid_static(opts)
        
        # generate GR grid mask and area if necessary
        if gr_grid_mask is None or gr_grid_areas is None:
            tea.generate_gr_grid_mask()
            _save_0p5_mask(opts, tea.gr_grid_mask, tea.gr_grid_areas)
        else:
            # set GR grid mask and area grid
            tea.gr_grid_mask = gr_grid_mask
            tea.gr_grid_areas = gr_grid_areas
        
        # set land_frac_min to 0 for full region
        if opts.full_region:
            tea.land_frac_min = 0
            
    # calculate annual climatic time period indicators
    logger.info('Calculating annual CTP indicators')
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="invalid value encountered in multiply")
        tea.calc_annual_ctp_indicators(opts.period, drop_daily_results=True)

    # save output
    save_ctp_output(opts=opts, tea=tea, start=start, end=end)


def calc_lat_lon_range(cell_size_lat, data, mask):
    """
    calculate latitude and longitude range for selected region
    Args:
        cell_size_lat: size of the grid cell in latitude
        data: input data
        mask: mask grid

    Returns:
        min_lat: minimum latitude
        min_lon: minimum longitude
        max_lat: maximum latitude
        max_lon: maximum longitude

    """
    min_lat = math.floor(data.lat[np.where(mask > 0)[0][-1]].values - cell_size_lat / 2)
    if min_lat < mask.lat.min().values:
        min_lat = float(mask.lat.min().values)
    max_lat = math.ceil(data.lat[np.where(mask > 0)[0][0]].values + cell_size_lat / 2)
    if max_lat > mask.lat.max().values:
        max_lat = float(mask.lat.max().values)
    cell_size_lon = 1 / np.cos(np.deg2rad(max_lat)) * cell_size_lat
    min_lon = math.floor(data.lon[np.where(mask > 0)[1][0]].values - cell_size_lon / 2)
    if min_lon < mask.lon.min().values:
        min_lon = float(mask.lon.min().values)
    max_lon = math.ceil(data.lon[np.where(mask > 0)[1][-1]].values + cell_size_lon / 2)
    if max_lon > mask.lon.max().values:
        max_lon = float(mask.lon.max().values)
    return min_lat, min_lon, max_lat, max_lon


def reduce_region(data, mask, threshold, opts):
    """
    reduce data to the region of interest
    Args:
        data: input data
        mask: mask grid
        threshold: threshold grid
        opts: options

    Returns:
        data: reduced data
        mask: reduced mask grid
        threshold: reduced threshold grid

    """
    if opts.precip:
        cell_size_lat = 1
    else:
        cell_size_lat = 2
    
    # preselect region to reduce computation time (incl. some margins to avoid boundary effects)
    if opts.full_region:
        min_lat = mask.lat.min().values
        max_lat = mask.lat.max().values
    else:
        min_lat, min_lon, max_lat, max_lon = calc_lat_lon_range(cell_size_lat, data, mask)
        
    if opts.dataset == 'ERA5' and opts.region == 'EUR':
        lons = np.arange(-12, 40.5, 0.5)
        cell_size_lon = 1 / np.cos(np.deg2rad(max_lat)) * cell_size_lat
        min_lon = math.floor(lons[0] - cell_size_lon / 2)
        max_lon = math.ceil(lons[-1] + cell_size_lon / 2)
        # if min_lat < 35 - cell_size_lat:
        #     # TODO: get rid of this
        #     logger.warning('Region is too far south. Setting minimum latitude to 35°N.')
        #     min_lat = 35 - cell_size_lat

    proc_data = data.sel(lat=slice(max_lat, min_lat), lon=slice(min_lon, max_lon))
    proc_mask = mask.sel(lat=slice(max_lat, min_lat), lon=slice(min_lon, max_lon))
    threshold = threshold.sel(lat=slice(max_lat, min_lat), lon=slice(min_lon, max_lon))
    
    return proc_data, proc_mask, threshold


def getopts():
    """
    get arguments
    :return: command line parameters
    """
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--config-file', '-cf',
                        dest='config_file',
                        type=str,
                        default='../TEA_CFG.yaml',
                        help='TEA configuration file (default: TEA_CFG.yaml)')
    
    myopts = parser.parse_args()
    
    return myopts


def run():
    warnings.filterwarnings(action='ignore', message='All-NaN slice encountered')
    warnings.filterwarnings(action='ignore', message='divide by zero encountered in divide')
    warnings.filterwarnings(action='ignore', message='invalid value encountered in divide')
    
    cmd_opts = getopts()
    
    # load CFG parameter
    opts = load_opts(fname=__file__, config_file=cmd_opts.config_file)
    
    if 'agr' in opts and False:
        calc_tea_indicators_agr(opts)
    else:
        calc_tea_indicators(opts)


def calc_tea_indicators(opts):
    """
    calculate TEA indicators for normal GeoRegion
    Args:
        opts: options as defined in CFG-PARAMS-doc.md and TEA_CFG_DEFAULT.yaml
    """
    
    mask = None
    lsm = None
    # load mask if needed
    if 'maskpath' in opts:
        mask = load_mask_file(opts)
    
    # load threshold grid or set threshold value
    if opts.threshold_type == 'abs':
        threshold_grid = opts.threshold
    else:
        threshold_file = f'{opts.statpath}/threshold_{opts.param_str}_{opts.period}_{opts.dataset}.nc'
        if opts.recalc_threshold or not os.path.exists(threshold_file):
            logger.info('Calculating percentiles...')
            threshold_grid = create_threshold_grid(opts=opts)
            logger.info(f'Saving threshold grid to {threshold_file}')
            threshold_grid.to_netcdf(threshold_file)
        else:
            logger.info(f'Loading threshold grid from {threshold_file}')
            threshold_grid = xr.open_dataset(threshold_file).threshold
    
    # calculate annual climatic time period indicators
    if not opts.decadal_only:
        starts = np.arange(opts.start, opts.end, 10)
        ends = np.append(np.arange(opts.start + 10 - 1, opts.end, 10), opts.end)
        
        for p_start, p_end in zip(starts, ends):
            # calculate daily basis variables
            tea = calc_dbv_indicators(mask=mask, opts=opts, start=p_start, end=p_end, threshold=threshold_grid)
            
            # calculate CTP indicators
            calc_ctp_indicators(tea=tea, opts=opts, start=p_start, end=p_end)
            
            gc.collect()
            
    # calculate decadal indicators and amplification factors
    if opts.decadal or opts.decadal_only or opts.recalc_decadal:
        tea = TEAIndicators()
        
        # calculate decadal-mean ctp indicator variables
        calc_decadal_indicators(opts=opts, tea=tea)
        
        # calculate amplification factors
        calc_amplification_factors(opts=opts, tea=tea)


def calc_dbv_indicators(start, end, threshold, opts, mask=None):
    """
    calculate daily basis variables for a given time period
    Args:
        start: start year
        end: end year
        threshold: either gridded threshold values (xarray DataArray) or a constant threshold value (int, float)
        opts: options
        mask: mask grid for input data containing nan values for cells that should be masked. Fractions of 1 are
        interpreted as area fractions for the given cell. (optional)

    Returns:
        tea: TEA object with daily basis variables

    """
    # check and create output path
    dbv_outpath = f'{opts.outpath}/daily_basis_variables'
    if not os.path.exists(dbv_outpath):
        os.makedirs(dbv_outpath)
        
    logger.info(f'Calculating TEA indicators for years {start}-{end}.')
    # set filenames
    if 'agr' in opts:
        agr_str = 'AGR-'
        TEA_class_obj = TEAAgr
    else:
        agr_str = ''
        TEA_class_obj = TEAIndicators
    
    # load land-sea mask for AGR
    if 'agr' in opts and 'maskpath' in opts:
        # load land-sea mask for AGR
        lsm = load_lsm_file(opts)
    else:
        lsm = None

    # DBV can't be the same for AGR and non-AGR (AGR is always without mask and has margins)
    dbv_filename = (f'{dbv_outpath}/'
                    f'DBV_{opts.param_str}_{agr_str}{opts.region}_annual_{opts.dataset}'
                    f'_{start}to{end}.nc')
    
    # recalculate daily basis variables if needed
    if opts.recalc_daily or not os.path.exists(dbv_filename):
        
        # always calculate annual basis variables to later extract sub-annual values
        period = 'annual'
        data = get_data(start=start, end=end, opts=opts, period=period)
    
        # reduce extent of data to the region of interest
        if 'agr' in opts:
            data, mask, threshold = reduce_region(data, mask, threshold, opts)
        
        # computation of daily basis variables (Methods chapter 3)
        logger.info('Daily basis variables will be recalculated. Period set to annual.')
        
        # set min area to < 1 grid cell area so that all exceedance days are considered
        min_area = 0.0001
        
        tea = TEA_class_obj(input_data_grid=data, threshold=threshold, mask=mask,
                            min_area=min_area, low_extreme=opts.low_extreme, unit=opts.unit, land_sea_mask=lsm)
        tea.calc_daily_basis_vars()
        
        # calculate hourly indicators
        if opts.hourly:
            calc_hourly_indicators(tea=tea, opts=opts, start=start, end=end)
        
        # save results
        tea.save_daily_results(dbv_filename)
    else:
        # load existing results
        tea = TEA_class_obj(threshold=threshold, mask=mask, low_extreme=opts.low_extreme, unit=opts.unit)
        logger.info(f'Loading daily basis variables from {dbv_filename}; if you want to recalculate them, '
                    'set --recalc-daily.')
        tea.load_daily_results(dbv_filename)
    return tea


def calc_hourly_indicators(tea, opts, start, end):
    """
    calculate hourly indicators for a given time period
    Args:
        tea: TEA object with daily basis variables
        opts: options
        start: start year
        end: end year

    Returns:
        tea: TEA object with hourly indicators

    """
    # load data
    data = get_data(start=start, end=end, opts=opts, hourly=True)
    
    logger.info('Calculating hourly basis variables.')
    # calculate hourly indicators
    tea.calc_hourly_indicators(input_data=data)
    
    
def calc_tea_indicators_agr(opts):
    """
    calculate TEA indicators for aggregated GeoRegions (AGR)
    Args:
        opts: options as defined in CFG-PARAMS-doc.md and TEA_CFG_DEFAULT.yaml

    Returns:

    """
    # TODO: run only last step as AGR, all other code should be the same?
    
    # load static files
    # check if GR size is larger than 100 areals and switch to AGR if so
    gr_grid_mask, gr_grid_areas = load_gr_grid_static(opts)
    
    if opts.precip:
        cell_size_lat = 1
    else:
        cell_size_lat = 2
        
    tea = TEAAgr(gr_grid_mask=gr_grid_mask, gr_grid_areas=gr_grid_areas, cell_size_lat=cell_size_lat)
    
    if not opts.decadal_only:
        # calculate annual climatic time period indicators
        myopts = deepcopy(opts)
        starts = np.arange(myopts.start, myopts.end, 10)
        ends = np.append(np.arange(myopts.start + 10 - 1, myopts.end, 10), myopts.end)
        
        dbv_outpath = f'{opts.outpath}/daily_basis_variables'
        if not os.path.exists(dbv_outpath):
            os.makedirs(dbv_outpath)
        
        for p_start, p_end in zip(starts, ends):
            myopts.start = p_start
            myopts.end = p_end
            logger.info(f'Calculating TEA indicators for years {myopts.start}-{myopts.end}.')
            
            # use European masks
            masks, static = load_static_files(opts=myopts, large_gr=True)
            data = get_data(start=p_start, end=p_end, opts=opts, period=opts.period)
            tea = largeGR.calc_tea_large_gr(opts=myopts, data=data, masks=masks, static=static,
                                            agr_mask=gr_grid_mask, agr_area=gr_grid_areas)
            gc.collect()
            
    if opts.decadal or opts.decadal_only or opts.recalc_decadal:
        agr_str = 'AGR-'
        
        outpath_decadal = (f'{opts.outpath}/dec_indicator_variables/'
                           f'DEC_{opts.param_str}_{agr_str}{opts.region}_{opts.period}_{opts.dataset}'
                           f'_{opts.start}to{opts.end}.nc')
        outpath_ampl = (f'{opts.outpath}/dec_indicator_variables/amplification/'
                        f'AF_{opts.param_str}_{agr_str}{opts.region}_{opts.period}_{opts.dataset}'
                        f'_{opts.start}to{opts.end}.nc')
        
        # calculate decadal-mean ctp indicator variables
        calc_decadal_indicators(opts=opts, tea=tea, outpath=outpath_decadal)
        
        # calculate amplification factors
        calc_amplification_factors(opts, tea, outpath_ampl)
        
        # calculate aggregate GeoRegion means and spread estimators
        agr_lat_range_dict = {'EUR': [35, 70], 'S-EUR': [35, 44.5], 'C-EUR': [45, 55], 'N-EUR': [55.5, 70]}
        agr_lon_range_dict = {'EUR': [-11, 40], 'S-EUR': [-11, 40], 'C-EUR': [-11, 40], 'N-EUR': [-11, 40]}
        if opts.region in agr_lat_range_dict:
            agr_lat_range = agr_lat_range_dict[opts.agr]
            agr_lon_range = agr_lon_range_dict[opts.agr]
        else:
            agr_lat_range = None
            agr_lon_range = None
        if opts.region != opts.agr:
            outpath_decadal = (f'{opts.outpath}/dec_indicator_variables/'
                               f'DEC_{opts.param_str}_{agr_str}{opts.agr}_{opts.period}_{opts.dataset}'
                               f'_{opts.start}to{opts.end}.nc')
            outpath_ampl = (f'{opts.outpath}/dec_indicator_variables/amplification/'
                            f'AF_{opts.param_str}_{agr_str}{opts.agr}_{opts.period}_{opts.dataset}'
                            f'_{opts.start}to{opts.end}.nc')
        tea.calc_agr_vars(lat_range=agr_lat_range, lon_range=agr_lon_range)
        logger.info(f'Saving AGR decadal results to {outpath_decadal}')
        # remove outpath_decadal if it exists
        if os.path.exists(outpath_decadal):
            os.remove(outpath_decadal)
        tea.save_decadal_results(outpath_decadal)
        logger.info(f'Saving AGR amplification factors to {outpath_ampl}')
        tea.save_amplification_factors(outpath_ampl)


def load_gr_grid_static(opts):
    """
    load grid of GRs mask and area grid for AGR calculation
    
    Args:
        opts: options as defined in CFG-PARAMS-doc.md and TEA_CFG_DEFAULT.yaml

    Returns:
        gr_grid_areas: area grid (xarray DataArray)
        gr_grid_mask: mask grid (xarray DataArray)

    """
    gr_grid_mask_file = f'{opts.maskpath}/{opts.mask_sub}/{opts.region}_mask_0p5_{opts.dataset}.nc'
    try:
        gr_grid_mask = xr.open_dataset(gr_grid_mask_file)
        gr_grid_mask = gr_grid_mask.mask_lt1500
    except FileNotFoundError:
        if opts.decadal_only:
            logger.warning(f'No GR mask found at {gr_grid_mask_file}.')
        gr_grid_mask = None
    gr_grid_areas_file = f'{opts.statpath}/area_grid_0p5_{opts.region}_{opts.dataset}.nc'
    try:
        gr_grid_areas = xr.open_dataset(gr_grid_areas_file)
        gr_grid_areas = gr_grid_areas.area_grid
    except FileNotFoundError:
        if opts.decadal_only:
            # TODO: make AGR code work without area grid (assuming all grid cells have same area)
            raise FileNotFoundError(f'No GR area grid found at {gr_grid_areas_file}. GR area grid is needed for '
                                    f'AGR calculations.')
        gr_grid_areas = None
    return gr_grid_mask, gr_grid_areas


if __name__ == '__main__':
    run()
