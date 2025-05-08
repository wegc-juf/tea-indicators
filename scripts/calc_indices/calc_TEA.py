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
    
    mask_file = xr.open_dataset(f'{opts.maskpath}/{opts.mask_sub}/{opts.region}_mask_{opts.dataset}.nc')
    
    return mask_file.mask


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
        tea_ref.load_CTP_results(ctp_filename_ref)
        tea_result = tea.CTP_results
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="invalid value encountered in multiply")
            compare_to_ref(tea_result, tea_ref.CTP_results)
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
    create_tea_history(cfg_params=opts, tea=tea, result_type='CTP')

    path = Path(f'{opts.outpath}/ctp_indicator_variables/supplementary/')
    path.mkdir(parents=True, exist_ok=True)
    
    outpath = (f'{opts.outpath}/ctp_indicator_variables/'
               f'CTP_{opts.param_str}_{opts.region}_{opts.period}_{opts.dataset}'
               f'_{start}to{end}.nc')
    
    path_ref = outpath.replace('.nc', '_ref.nc')
    
    logger.info(f'Saving CTP indicators to {outpath}')
    tea.save_CTP_results(outpath)
    
    if opts.compare_to_ref:
        compare_to_ctp_ref(tea, path_ref)


def calc_ctp_indicators(tea, opts, start, end):
    """
    calculate the TEA indicators for the annual climatic time period
    Args:
        tea: TEA object
        opts: CLI parameter
        start: start year
        end: end year
    """
    
    # apply criterion that DTEA_GR > DTEA_min and all GR variables use same dates,
    # dtea_min is given in areals (1 areal = 100 km2)
    dtea_min = 1  # according to equation 03
    tea.update_min_area(dtea_min)
    
    # calculate annual climatic time period indicators
    logger.info('Calculating annual CTP indicators')
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="invalid value encountered in multiply")
        tea.calc_annual_CTP_indicators(opts.period, drop_daily_results=True)

    # save output
    save_ctp_output(opts=opts, tea=tea, start=start, end=end)


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
    
    if 'agr' in opts:
        calc_tea_indicators_agr(opts)
    else:
        calc_tea_indicators(opts)


def calc_tea_indicators(opts):
    """
    calculate TEA indicators for normal GeoRegion
    Args:
        opts: options as defined in CFG-PARAMS-doc.md and TEA_CFG_DEFAULT.yaml
    """
    
    # load mask if needed
    if 'maskpath' in opts:
        mask = load_mask_file(opts)
    else:
        mask = None
    
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
            
            # calculate hourly indicators
            if opts.hourly:
                calc_hourly_indicators(tea=tea, opts=opts, start=p_start, end=p_end)
            
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
    dbv_filename = (f'{dbv_outpath}/'
                    f'DBV_{opts.param_str}_{opts.region}_annual_{opts.dataset}'
                    f'_{start}to{end}.nc')
    
    # recalculate daily basis variables if needed
    if opts.recalc_daily or not os.path.exists(dbv_filename):
        
        # always calculate annual basis variables to later extract sub-annual values
        period = 'annual'
        data = get_data(start=start, end=end, opts=opts, period=period)
        
        # computation of daily basis variables (Methods chapter 3)
        logger.info('Daily basis variables will be recalculated. Period set to annual.')
        tea = TEAIndicators(input_data_grid=data, threshold=threshold, mask=mask,
                            # set min area to < 1 grid cell area so that all exceedance days are considered
                            min_area=0.0001, low_extreme=opts.low_extreme, unit=opts.unit)
        
        tea.calc_daily_basis_vars()
        
        # save results
        logger.info(f'Saving daily basis variables to {dbv_filename}')
        tea.save_daily_results(dbv_filename)
    else:
        # load existing results
        tea = TEAIndicators(threshold=threshold, mask=mask, low_extreme=opts.low_extreme, unit=opts.unit)
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
    
    # calculate hourly indicators
    tea.calc_hourly_indicators(input_data=data)
    
    
def calc_tea_indicators_agr(opts):
    # TODO: run only last step as AGR, all other code should be the same?
    # load static files
    gr_grid_mask = None
    gr_grid_areas = None
    # check if GR size is larger than 100 areals and switch to AGR if so
    gr_grid_areas, gr_grid_mask = load_gr_grid_static(gr_grid_areas, gr_grid_mask, opts)
    
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
            
            # set filenames
            dbv_filename = (f'{dbv_outpath}/'
                            f'DBV_{myopts.param_str}_{myopts.region}_annual_{myopts.dataset}'
                            f'_{myopts.start}to{myopts.end}.nc')
            
            # check if GR size is larger than 100 areals and switch to calc_TEA_largeGR if so
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


def load_gr_grid_static(gr_grid_areas, gr_grid_mask, opts):
    # load agr mask
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
    return gr_grid_areas, gr_grid_mask


if __name__ == '__main__':
    run()
