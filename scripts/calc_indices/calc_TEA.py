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

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from scripts.general_stuff.general_functions import create_history_from_cfg, load_opts, compare_to_ref
from scripts.general_stuff.TEA_logger import logger
from scripts.calc_indices.calc_daily_basis_vars import calc_daily_basis_vars
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
    create_tea_history(cli_params=sys.argv, tea=tea, result_type='CTP')

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


def calc_ctp_indicators(opts, masks, static, agr_mask=None, agr_area=None):
    """
    calculate the TEA indicators for the annual climatic time period
    Args:
        opts: CLI parameter
        masks: mask files
        static: static files
        agr_mask: mask for AGR
        agr_area: area for AGR

    Returns:

    """
    
    # check if GR size is larger than 100 areals and switch to calc_TEA_largeGR if so
    if 'ERA' in opts.dataset and static['GR_size'] > 100:
        # use European masks
        masks, static = load_static_files(opts=opts, large_gr=True)
        data = get_data(opts=opts)
        tea_agr = largeGR.calc_tea_large_gr(opts=opts, data=data, masks=masks, static=static, agr_mask=agr_mask,
                                            agr_area=agr_area)
        return tea_agr

    # computation of daily basis variables (Methods chapter 3)
    if opts.recalc_daily:
        data = get_data(opts=opts)
        
        # load GR masks and static file
        logger.info('Daily basis variables will be recalculated. Period set to annual.')
        old_period = opts.period
        opts.period = 'annual'
        
        # create mask array
        mask = masks['lt1500_mask'] * masks['mask']
        
        tea = calc_daily_basis_vars(opts=opts, static=static, data=data, mask=mask)
        opts.period = old_period
    else:
        tea = TEAIndicators()
        
        dbv_filename = (f'{opts.outpath}/daily_basis_variables/DBV_{opts.param_str}_{opts.region}_annual'
                            f'_{opts.dataset}_{opts.start}to{opts.end}.nc')

        logger.info(f'Loading daily basis variables from {dbv_filename}; if you want to recalculate them, '
                    'set --recalc-daily.')
        tea.load_daily_results(dbv_filename)

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
    return tea


def run():
    warnings.filterwarnings(action='ignore', message='All-NaN slice encountered')
    warnings.filterwarnings(action='ignore', message='divide by zero encountered in divide')
    warnings.filterwarnings(action='ignore', message='invalid value encountered in divide')
    
    # load CFG parameter
    opts = load_opts(fname=__file__)

    # check length of input time span
    start = opts.start
    end = opts.end
    
    # load static files
    masks, static = load_static_files(opts=opts)
    agr_mask = None
    agr_area = None
    # check if GR size is larger than 100 areals and switch to AGR if so
    if 'ERA' in opts.dataset and static['GR_size'] > 100:
        # load agr mask
        try:
            agr_mask = xr.open_dataset(f'{opts.maskpath}/{opts.region}_mask_0p5_{opts.dataset}.nc')
            agr_mask = agr_mask.mask_lt1500
        except FileNotFoundError:
            agr_mask = None
        
        try:
            agr_area = xr.open_dataset(f'{opts.statpath}/area_grid_0p5_{opts.region}_{opts.dataset}.nc')
            agr_area = agr_area.area_grid
        except FileNotFoundError:
            agr_area = None
        
        tea = TEAAgr(agr_mask=agr_mask, agr_area=agr_area)
        agr = True
    else:
        tea = TEAIndicators()
        agr = False
    
    if not opts.decadal_only:
        # calculate annual climatic time period indicators
        if end - start > 10 - 1:
            starts = np.arange(start, end, 10)
            ends = np.append(np.arange(start + 10 - 1, end, 10), end)
            for pstart, pend in zip(starts, ends):
                opts.start = pstart
                opts.end = pend
                logger.info(f'Calculating TEA indicators for years {opts.start}-{opts.end}.')
                tea = calc_ctp_indicators(opts=opts, masks=masks, static=static, agr_mask=agr_mask, agr_area=agr_area)
                gc.collect()
        else:
            tea = calc_ctp_indicators(opts=opts, masks=masks, static=static, agr_mask=agr_mask, agr_area=agr_area)

    if opts.decadal or opts.decadal_only or opts.recalc_decadal:
        opts.start, opts.end = start, end
        
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
    
        if agr:
            tea.calc_agr_mean()
            tea.save_decadal_results(outpath_decadal)
            tea.save_amplification_factors(outpath_ampl)


if __name__ == '__main__':
    run()
