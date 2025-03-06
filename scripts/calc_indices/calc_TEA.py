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
                                                     compare_to_ref)
from scripts.general_stuff.TEA_logger import logger
from scripts.calc_indices.calc_decadal_indicators import calc_decadal_indicators, calc_amplification_factors
import scripts.calc_indices.calc_TEA_largeGR as largeGR
from scripts.calc_indices.TEA import TEAIndicators
from scripts.calc_indices.TEA_AGR import TEAAgr

DS_PARAMS = {'SPARTACUS': {'xname': 'x', 'yname': 'y'},
             'ERA5': {'xname': 'lon', 'yname': 'lat'},
             'ERA5Land': {'xname': 'lon', 'yname': 'lat'}}


def get_data(opts):
    """
    loads data
    :param opts: input parameter
    :return: dataset of daily maximum temperature or precipitation
    """

    param_str = ''
    if opts.dataset == 'SPARTACUS' and not opts.precip:
        param_str = f'{opts.parameter}'
    elif opts.dataset == 'SPARTACUS' and opts.precip:
        param_str = 'RR'

    # select only files of interest, if chosen period is 'seasonal' append one year in the
    # beginning to have the first winter fully included
    filenames = []
    if opts.period == 'seasonal' and opts.start != '1961':
        yrs = np.arange(opts.start - 1, opts.end + 1)
    else:
        yrs = np.arange(opts.start, opts.end + 1)
    for iyrs in yrs:
        year_files = sorted(glob.glob(
            f'{opts.inpath}*{param_str}_{iyrs}*.nc'))
        filenames.extend(year_files)

    # load relevant years
    try:
        ds = xr.open_mfdataset(filenames, combine='by_coords')
    except ValueError:
        ds = xr.open_dataset(filenames[0])

    # select only times of interest
    if opts.period == 'seasonal' and opts.start != '1961':
        start = f'{opts.start - 1}-12-01'
        end = f'{opts.end}-11-30'
        ds = ds.sel(time=slice(start, end))
    elif opts.period == 'seasonal' and opts.start == '1961':
        # if first year is first year of record, exclude first winter (data of Dec 1960 missing)
        start = f'{opts.start - 1}-03-01'
        end = f'{opts.end}-11-30'
        ds = ds.sel(time=slice(start, end))

    if opts.period in ['ESS', 'WAS', 'JJA']:
        months = {'ESS': np.arange(5, 10), 'WAS': np.arange(4, 11), 'JJA': np.arange(6, 9)}
        season = ds['time'].dt.month.isin(months[opts.period])
        ds = ds.sel(time=season)

    # select variable
    if opts.dataset == 'SPARTACUS' and opts.parameter == 'P24h_7to7':
        ds = ds.rename({'RR': opts.parameter})
    data = ds[opts.parameter]

    if opts.dataset == 'SPARTACUS':
        data = data.drop('lambert_conformal_conic')

    return data


def load_static_files(opts, large_gr=False):
    """
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
    masks = xr.open_dataset(f'{opts.maskpath}{opts.region}_masks_{opts.dataset}.nc')

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
    
    
def save_ctp_output(opts, tea):
    """
    save CTP results to netcdf file
    Args:
        opts: CLI parameters
        tea: TEA object
    """
    create_tea_history(cfg_params=opts, tea=tea, result_type='CTP')

    path = Path(f'{opts.outpath}/ctp_indicator_variables/supplementary/')
    path.mkdir(parents=True, exist_ok=True)
    
    outpath = (f'{opts.outpath}/ctp_indicator_variables/'
               f'CTP_{opts.param_str}_{opts.region}_{opts.period}_{opts.dataset}'
               f'_{opts.start}to{opts.end}.nc')
    
    path_ref = outpath.replace('.nc', '_ref.nc')
    
    logger.info(f'Saving CTP indicators to {outpath}')
    tea.save_CTP_results(outpath)
    
    if opts.compare_to_ref:
        compare_to_ctp_ref(tea, path_ref)


def calc_daily_basis_vars(data, threshold_grid, mask=None, area_grid=None, unit='', low_extreme=False):
    """
    compute daily basis variables following chapter 3 of TEA methods
    Args:
        data: input data grid
        threshold_grid: threshold grid for computing TEA metrics
        mask: mask grid for masking out certain regions (optional)
        area_grid: area grid with individual grid cell areas (optional)
        unit: unit of the input data (default: '')
        low_extreme: set to True if low extremes are to be calculated (default: False)

    Returns:
        TEA: TEA object
    """
    # create TEA object
    tea = TEAIndicators(input_data_grid=data, threshold=threshold_grid, area_grid=area_grid, mask=mask,
                        # set min area to < 1 grid cell area so that all exceedance days are considered
                        min_area=0.0001,
                        low_extreme=low_extreme, unit=unit)
    
    tea.calc_daily_basis_vars()
    
    return tea


def calc_ctp_indicators(tea, opts):
    """
    calculate the TEA indicators for the annual climatic time period
    Args:
        tea: TEA object
        opts: CLI parameter
    """
    # apply criterion that DTEA_GR > DTEA_min and all GR variables use same dates,
    # dtea_min is given in areals (1 areal = 100 km2)
    dtea_min = 1
    tea.update_min_area(dtea_min)
    
    # calculate annual climatic time period indicators
    logger.info('Calculating annual CTP indicators')
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="invalid value encountered in multiply")
        tea.calc_annual_CTP_indicators(opts.period, drop_daily_results=True)

    # save output
    save_ctp_output(opts=opts, tea=tea)


def run():
    warnings.filterwarnings(action='ignore', message='All-NaN slice encountered')
    warnings.filterwarnings(action='ignore', message='divide by zero encountered in divide')
    warnings.filterwarnings(action='ignore', message='invalid value encountered in divide')
    
    # load CFG parameter
    opts = load_opts(fname=__file__)

    # load static files
    masks, static = load_static_files(opts=opts)
    threshold_grid = static['threshold']
    area_grid = static['area_grid']
    
    gr_grid_mask = None
    gr_grid_areas = None
    # check if GR size is larger than 100 areals and switch to AGR if so
    if 'ERA' in opts.dataset and static['GR_size'] > 100:
        # load agr mask
        try:
            gr_grid_mask_file = f'{opts.maskpath}/{opts.region}_mask_GRG_0p5_{opts.dataset}.nc'
            gr_grid_mask = xr.open_dataset(gr_grid_mask_file)
            gr_grid_mask = gr_grid_mask.mask_lt1500
        except FileNotFoundError:
            logger.warning(f'No GR mask found at {gr_grid_mask_file}.')
            gr_grid_mask = None
        
        try:
            gr_grid_areas_file = f'{opts.statpath}/area_grid_GRG_0p5_{opts.region}_{opts.dataset}.nc'
            gr_grid_areas = xr.open_dataset(gr_grid_areas_file)
            gr_grid_areas = gr_grid_areas.area_grid
        except FileNotFoundError:
            logger.warning(f'No GR area grid found at {gr_grid_areas_file}.')
            gr_grid_areas = None
        
        tea = TEAAgr(gr_grid_mask=gr_grid_mask, gr_grid_areas=gr_grid_areas)
        agr = True
    else:
        tea = TEAIndicators()
        agr = False
    
    if not opts.decadal_only:
        # calculate annual climatic time period indicators
        myopts = deepcopy(opts)
        starts = np.arange(myopts.start, myopts.end, 10)
        ends = np.append(np.arange(myopts.start + 10 - 1, myopts.end, 10), myopts.end)
        
        dbv_outpath = f'{opts.outpath}/daily_basis_variables'
        if not os.path.exists(dbv_outpath):
            os.makedirs(dbv_outpath)
            
        for pstart, pend in zip(starts, ends):
            myopts.start = pstart
            myopts.end = pend
            logger.info(f'Calculating TEA indicators for years {myopts.start}-{myopts.end}.')
            
            # set filenames
            dbv_filename = (f'{dbv_outpath}/'
                            f'DBV_{myopts.param_str}_{myopts.region}_annual_{myopts.dataset}'
                            f'_{myopts.start}to{myopts.end}.nc')
            
            # check if GR size is larger than 100 areals and switch to calc_TEA_largeGR if so
            if 'ERA' in myopts.dataset and static['GR_size'] > 100:
                # use European masks
                masks, static = load_static_files(opts=myopts, large_gr=True)
                data = get_data(opts=myopts)
                tea = largeGR.calc_tea_large_gr(opts=myopts, data=data, masks=masks, static=static,
                                                agr_mask=gr_grid_mask, agr_area=gr_grid_areas)
            else:
                # create mask array
                mask = masks['lt1500_mask'] * masks['mask']
                
                if myopts.recalc_daily or not os.path.exists(dbv_filename):
                    # always calculate annual basis variables to later extract sub-annual values
                    old_period = myopts.period
                    myopts.period = 'annual'
                    data = get_data(opts=myopts)
                    
                    # computation of daily basis variables (Methods chapter 3)
                    logger.info('Daily basis variables will be recalculated. Period set to annual.')
                    tea = calc_daily_basis_vars(data=data, threshold_grid=threshold_grid, area_grid=area_grid,
                                                mask=mask, unit=myopts.unit, low_extreme=myopts.low_extreme)
                    
                    # save results
                    logger.info(f'Saving daily basis variables to {dbv_filename}')
                    tea.save_daily_results(dbv_filename)
                    myopts.period = old_period
                else:
                    # load existing results
                    tea = TEAIndicators(area_grid=area_grid)
                    logger.info(f'Loading daily basis variables from {dbv_filename}; if you want to recalculate them, '
                                'set --recalc-daily.')
                    tea.load_daily_results(dbv_filename)
                
                # calculate CTP indicators
                calc_ctp_indicators(tea=tea, opts=myopts)
                
            gc.collect()

    if opts.decadal or opts.decadal_only or opts.recalc_decadal:
        if agr:
            agr_str = 'AGR-'
        else:
            agr_str = ''
        
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
        if agr:
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


if __name__ == '__main__':
    run()
