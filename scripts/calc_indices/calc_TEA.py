#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: hst, juf
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
                                                     compare_to_ref, get_gridded_data, get_csv_data,
                                                     create_threshold_grid)
from scripts.general_stuff.TEA_logger import logger
from scripts.calc_indices.calc_decadal_indicators import (calc_decadal_indicators, calc_amplification_factors,
                                                          _get_decadal_outpath, _get_amplification_outpath)
from scripts.calc_indices.TEA import TEAIndicators
from scripts.calc_indices.TEA_AGR import TEAAgr

# TODO: move this to config file
region_def_lat_ = {'EUR': [35, 70], 'S-EUR': [35, 44.5], 'C-EUR': [45, 55], 'N-EUR': [55.5, 70]}
region_def_lon_ = {'EUR': [-11, 40], 'S-EUR': [-11, 40], 'C-EUR': [-11, 40], 'N-EUR': [-11, 40]}


def calc_tea_indicators(opts):
    """
    calculate TEA indicators as defined in https://doi.org/10.48550/arXiv.2504.18964 and
    Methods as defined in
    Kirchengast, G., Haas, S. J. & Fuchsberger, J. Compound event metrics detect and explain ten-fold
    increase of extreme heat over Europe—Supplementary Note: Detailed methods description for
    computing threshold-exceedance-amount (TEA) indicators. Supplementary Information (SI) to
    Preprint – April 2025. 40 pp. Wegener Center, University of Graz, Graz, Austria, 2025.
    
    Args:
        opts: options as defined in CFG-PARAMS-doc.md and TEA_CFG_DEFAULT.yaml
    """
    
    # load mask if needed
    if 'maskpath' in opts:
        mask = _load_mask_file(opts)
    else:
        mask = None
    
    # calculate daily and annual climatic time period indicators
    if not opts.decadal_only:
        
        # load threshold grid or set threshold value
        threshold_grid = _get_threshold(opts)
        
        # do calcs in chunks of 10 years
        starts = np.arange(opts.start, opts.end, 10)
        ends = np.append(np.arange(opts.start + 10 - 1, opts.end, 10), opts.end)
        
        for p_start, p_end in zip(starts, ends):
            # calculate daily basis variables
            tea = calc_dbv_indicators(mask=mask, opts=opts, start=p_start, end=p_end, threshold=threshold_grid)
            
            # for aggregate GeoRegion calculation, load GR grid files
            if 'agr' in opts:
                _load_or_generate_gr_grid(opts, tea)
            
            # calculate CTP indicators
            calc_annual_ctp_indicators(tea=tea, opts=opts, start=p_start, end=p_end)
            
            # collect garbage
            gc.collect()
    
    # calculate decadal indicators and amplification factors
    if opts.decadal or opts.decadal_only or opts.recalc_decadal:
        if 'agr' in opts:
            tea = TEAAgr()
        else:
            tea = TEAIndicators()
        
        # calculate decadal-mean ctp indicator variables
        calc_decadal_indicators(opts=opts, tea=tea)
        
        # calculate amplification factors
        calc_amplification_factors(opts=opts, tea=tea)
        
        # calculate AGR variables
        if 'agr' in opts:
            _load_or_generate_gr_grid(opts, tea)
            _calc_agr_mean_and_spread(opts=opts, tea=tea)


def calc_dbv_indicators(start, end, threshold, opts, mask=None, gridded=True):
    """
    calculate daily basis variables for a given time period
    Args:
        start: start year
        end: end year
        threshold: either gridded threshold values (xarray DataArray) or a constant threshold value (int, float)
        opts: options as defined in CFG-PARAMS-doc.md and TEA_CFG_DEFAULT.yaml
        mask: mask grid for input data containing nan values for cells that should be masked. Fractions of 1 are
        interpreted as area fractions for the given cell. (optional)
        gridded: if True, load gridded data, else load station timeseries (default: True)

    Returns:
        tea: TEA object with daily basis variables

    """
    # check and create output path
    dbv_outpath = f'{opts.outpath}/daily_basis_variables'
    if not os.path.exists(dbv_outpath):
        os.makedirs(dbv_outpath)
    
    logger.info(f'Calculating TEA indicators for years {start}-{end}.')
    
    # use either TEAIndicators or TEAAgr class depending on the options
    if 'agr' in opts:
        agr_str = 'AGR-'
        TEA_class_obj = TEAAgr
    else:
        agr_str = ''
        TEA_class_obj = TEAIndicators
    
    # load land-sea mask for AGR
    if 'agr' in opts and 'maskpath' in opts:
        # load land-sea mask for AGR
        lsm = _load_lsm_file(opts)
    else:
        lsm = None
    
    # DBV can't be the same for AGR and non-AGR (AGR is always without mask and has margins) so optionally add agr_str
    if gridded:
        name = opts.region
    else:
        name = opts.station
    dbv_filename = (f'{dbv_outpath}/'
                    f'DBV_{opts.param_str}_{agr_str}{name}_annual_{opts.dataset}'
                    f'_{start}to{end}.nc')
    
    # recalculate daily basis variables if needed
    if opts.recalc_daily or not os.path.exists(dbv_filename):
        
        # always calculate annual basis variables to later extract sub-annual values
        period = 'annual'
        if gridded:
            data = get_gridded_data(start=start, end=end, opts=opts, period=period)
        else:
            data = get_csv_data(opts)
            threshold = create_threshold_grid(opts, data=data)
        
        # reduce extent of data to the region of interest
        if 'agr' in opts:
            data, mask, threshold = _reduce_region(opts, data, mask, threshold)
        
        logger.info('Daily basis variables will be recalculated. Period set to annual.')
        
        # set min area to < 1 grid cell area so that all exceedance days are considered
        min_area = 0.0001
        
        # initialize TEA object
        tea = TEA_class_obj(input_data=data, threshold=threshold, mask=mask,
                            min_area=min_area, low_extreme=opts.low_extreme, unit=opts.unit, land_sea_mask=lsm)
        
        # computation of daily basis variables (Methods chapter 3)
        if gridded:
            gr = opts.hourly
        else:
            gr = False
        tea.calc_daily_basis_vars(gr=gr)
        
        # calculate hourly indicators
        if opts.hourly:
            _calc_hourly_indicators(tea=tea, opts=opts, start=start, end=end)
        
        # save results
        tea.save_daily_results(dbv_filename)
    else:
        # load existing results
        tea = TEA_class_obj(threshold=threshold, mask=mask, low_extreme=opts.low_extreme, unit=opts.unit,
                            land_sea_mask=lsm)
        logger.info(f'Loading daily basis variables from {dbv_filename}; if you want to recalculate them, '
                    'set --recalc-daily.')
        tea.load_daily_results(dbv_filename)
    return tea


def calc_annual_ctp_indicators(tea, opts, start, end):
    """
    calculate the TEA indicators for the annual climatic time period
    Args:
        tea: TEA object
        opts: options as defined in CFG-PARAMS-doc.md and TEA_CFG_DEFAULT.yaml
        start: start year
        end: end year
    """
    
    # apply criterion that DTEA_GR > DTEA_min and all GR variables use same dates,
    # dtea_min is given in areals (1 areal = 100 km2)
    dtea_min = 1  # according to equation 03
    tea.update_min_area(dtea_min)
    
    if 'agr' in opts:
        # set land_frac_min to 0 for full region
        if opts.full_region:
            tea.land_frac_min = 0
    
    # calculate annual climatic time period indicators
    logger.info('Calculating annual CTP indicators')
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="invalid value encountered in multiply")
        tea.calc_annual_ctp_indicators(opts.period, drop_daily_results=True)
    
    # save output
    _save_ctp_output(opts=opts, tea=tea, start=start, end=end)


def _get_threshold(opts):
    """
    load threshold grid or set threshold value
    Args:
        opts: options as defined in CFG-PARAMS-doc.md and TEA_CFG_DEFAULT.yaml

    Returns:
        threshold_grid: threshold grid (xarray DataArray) or constant threshold value (int, float)

    """
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
    return threshold_grid


def _load_mask_file(opts):
    """
    load GR mask
    Args:
        opts: options as defined in CFG-PARAMS-doc.md and TEA_CFG_DEFAULT.yaml

    Returns:
        mask: GR mask (Xarray DataArray)

    """
    maskpath = f'{opts.maskpath}/{opts.mask_sub}/{opts.region}_mask_{opts.dataset}.nc'
    logger.info(f'Loading mask from {maskpath}')
    mask_file = xr.open_dataset(maskpath)
    
    return mask_file.mask


def _load_lsm_file(opts):
    """
    load land-sea-mask for AGR
    Args:
        opts: options as defined in CFG-PARAMS-doc.md and TEA_CFG_DEFAULT.yaml

    Returns:
        mask: mask (Xarray DataArray)

    """
    # TODO: make this work outside of EUR
    new_opts = deepcopy(opts)
    new_opts.region = 'EUR'
    return _load_mask_file(new_opts)


def _compare_to_ctp_ref(tea, ctp_filename_ref):
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
    
    
def _save_ctp_output(opts, tea, start, end):
    """
    save annual CTP results to netcdf file
    Args:
        opts: options as defined in CFG-PARAMS-doc.md and TEA_CFG_DEFAULT.yaml
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
    
    if 'region' in opts:
        name = opts.region
    else:
        name = opts.station
    outpath = (f'{opts.outpath}/ctp_indicator_variables/'
               f'CTP_{opts.param_str}_{grg_str}{name}_{opts.period}_{opts.dataset}'
               f'_{start}to{end}.nc')
    
    path_ref = outpath.replace('.nc', '_ref.nc')
    
    logger.info(f'Saving CTP indicators to {outpath}')
    tea.save_ctp_results(outpath)
    
    if opts.compare_to_ref:
        _compare_to_ctp_ref(tea, path_ref)
        
        
def _save_0p5_mask(opts, mask_0p5, area_0p5):
    """
    save mask on 0.5° grid to netcdf file
    Args:
        opts: CLI parameter
        mask_0p5: mask on 0.5° grid
        area_0p5: area grid on 0.5° grid
    """
    # TODO: allow arbitrary resolutions
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


def _load_or_generate_gr_grid(opts, tea):
    """
    load or generate grid of GRs mask and area grid for AGR calculation
    Args:
        opts: options as defined in CFG-PARAMS-doc.md and TEA_CFG_DEFAULT.yaml
        tea: TEA object

    Returns:

    """
    # load static GR grid files
    gr_grid_mask, gr_grid_areas = _load_gr_grid_static(opts)
    # generate GR grid mask and area if necessary
    if gr_grid_mask is None or gr_grid_areas is None:
        tea.generate_gr_grid_mask()
        _save_0p5_mask(opts, tea.gr_grid_mask, tea.gr_grid_areas)
    else:
        # set GR grid mask and area grid
        tea.gr_grid_mask = gr_grid_mask
        tea.gr_grid_areas = gr_grid_areas
        
    # set cell_size
    tea.cell_size_lat = opts.agr_cell_size


def _calc_lat_lon_range(cell_size_lat, data, mask):
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


def _reduce_region(opts, data, mask, threshold=None, full_region=False):
    """
    reduce data to the region of interest
    Args:
        opts: options as defined in CFG-PARAMS-doc.md and TEA_CFG_DEFAULT.yaml
        data: input data
        mask: mask grid
        threshold: threshold grid
        full_region: if True, use the full region (default: opts.full_region)

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
    if opts.full_region or full_region:
        min_lat = mask.lat.min().values
        max_lat = mask.lat.max().values
        min_lon = mask.lon.min().values
        max_lon = mask.lon.max().values
    else:
        min_lat, min_lon, max_lat, max_lon = _calc_lat_lon_range(cell_size_lat, data, mask)
        
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
    if threshold is not None:
        threshold = threshold.sel(lat=slice(max_lat, min_lat), lon=slice(min_lon, max_lon))
    
    return proc_data, proc_mask, threshold


def _getopts():
    """
    get command line arguments
    
    Returns:
        opts: command line parameters
    """
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--config-file', '-cf',
                        dest='config_file',
                        type=str,
                        default='../TEA_CFG.yaml',
                        help='TEA configuration file (default: TEA_CFG.yaml)')
    
    myopts = parser.parse_args()
    
    return myopts


def _calc_hourly_indicators(tea, opts, start, end):
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
    data = get_gridded_data(start=start, end=end, opts=opts, hourly=True)
    
    if 'agr' in opts:
        # reduce data to the region of interest
        data, _, _ = _reduce_region(opts, data, tea.mask, full_region=True)
    
    if not _check_data_extent(data, tea.input_data):
        logger.warning('Hourly data extent is not the same as daily data extent. '
                       'Please check your data and the region you want to calculate.')
    
    logger.info('Calculating hourly basis variables.')
    # calculate hourly indicators
    tea.calc_hourly_indicators(input_data=data)
    
    
def _check_data_extent(data, ref_data):
    """
    check if data extent is the same as the TEA data extent
    Args:
        data: input data
        ref_data: reference data

    Returns:
        True if data extent is the same, False otherwise

    """
    if not np.array_equal(data.lat.values, ref_data.lat.values) or not np.array_equal(data.lon.values,
                                                                                      ref_data.lon.values):
        return False
    else:
        return True
    
    
def _calc_agr_mean_and_spread(opts, tea):
    """
    calculate aggregate GeoRegion means and spread estimators
    
    Args:
        opts: options as defined in CFG-PARAMS-doc.md and TEA_CFG_DEFAULT.yaml
        tea: teaAgr object

    Returns:

    """
    if opts.region in region_def_lat_:
        agr_lat_range = region_def_lat_[opts.agr]
        agr_lon_range = region_def_lon_[opts.agr]
    else:
        agr_lat_range = None
        agr_lon_range = None
        
    tea.calc_agr_vars(lat_range=agr_lat_range, lon_range=agr_lon_range)
    
    # save results
    outpath_decadal = _get_decadal_outpath(opts, opts.agr)
    outpath_ampl = _get_amplification_outpath(opts, opts.agr)
    logger.info(f'Saving AGR decadal results to {outpath_decadal}')
    # remove outpath_decadal if it exists
    if os.path.exists(outpath_decadal):
        os.remove(outpath_decadal)
    tea.save_decadal_results(outpath_decadal)
    logger.info(f'Saving AGR amplification factors to {outpath_ampl}')
    tea.save_amplification_factors(outpath_ampl)


def _load_gr_grid_static(opts):
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


def _run():
    """
    run the script
    Returns:

    """
    
    # suppress warnings
    warnings.filterwarnings(action='ignore', message='All-NaN slice encountered')
    warnings.filterwarnings(action='ignore', message='divide by zero encountered in divide')
    warnings.filterwarnings(action='ignore', message='invalid value encountered in divide')
    
    # get command line parameters
    cmd_opts = _getopts()
    
    # load CFG parameters
    opts = load_opts(fname=__file__, config_file=cmd_opts.config_file)
    
    # calculate TEA indicators
    calc_tea_indicators(opts)


if __name__ == '__main__':
    _run()
