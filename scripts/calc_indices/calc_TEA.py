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

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from scripts.general_stuff.general_functions import create_history, extend_tea_opts
from scripts.general_stuff.var_attrs import get_attrs
from scripts.general_stuff.TEA_logger import logger
from scripts.calc_indices.calc_daily_basis_vars import calc_daily_basis_vars, save_event_count
from scripts.calc_indices.calc_ctp_indicator_variables import (calc_event_frequency,
                                                               calc_supplementary_event_vars,
                                                               calc_event_duration,
                                                               calc_exceedance_magnitude,
                                                               calc_exceedance_area_tex_sev)
from scripts.calc_indices.calc_decadal_indicators import calc_decadal_indicators
from scripts.calc_indices.general_TEA_stuff import assign_ctp_coords, validate_period
import scripts.calc_indices.calc_TEA_largeGR as largeGR
from scripts.calc_indices.TEA import TEAIndicators

DS_PARAMS = {'SPARTACUS': {'xname': 'x', 'yname': 'y'},
             'ERA5': {'xname': 'lon', 'yname': 'lat'},
             'ERA5Land': {'xname': 'lon', 'yname': 'lat'}}


def getopts():
    """
    get arguments
    :return: command line parameters
    """

    def dir_path(path):
        if os.path.isdir(path):
            return path
        else:
            raise argparse.ArgumentTypeError(f'{path} is not a valid path.')

    def float_1pcd(value):
        if not re.match(r'^\d+(\.\d{1})?$', value):
            raise argparse.ArgumentTypeError('Threshold value must have at most one digit after '
                                             'the decimal point.')
        return float(value)

    parser = argparse.ArgumentParser()

    parser.add_argument('--start',
                        default=1961,
                        type=int,
                        help='Start of the interval to be processed [default: 1961].')

    parser.add_argument('--end',
                        default=pd.to_datetime('today').year,
                        type=int,
                        help='End of the interval to be processed [default: current year].')

    parser.add_argument('--period',
                        dest='period',
                        default='WAS',
                        type=str,
                        choices=['monthly', 'seasonal', 'annual', 'WAS', 'ESS', 'JJA', 'EWS'],
                        help='Climatic time period (CTP) of interest. '
                             'Options: monthly, seasonal, WAS, ESS, EWS, JJA, and  annual [default].')

    parser.add_argument('--region',
                        default='AUT',
                        type=str,
                        help='GeoRegion. Options: EUR, AUT (default), SAR, SEA, FBR, '
                             'Austrian state, or ISO2 code of a european country.')

    parser.add_argument('--parameter',
                        default='Tx',
                        type=str,
                        help='Parameter for which the TEA indices should be calculated'
                             '[default: Tx].')

    parser.add_argument('--unit',
                        default='degC',
                        type=str,
                        help='Physical unit of chosen parameter.')

    parser.add_argument('--precip',
                        action='store_true',
                        help='Set if chosen parameter is a precipitation parameter.')

    parser.add_argument('--threshold',
                        default=99,
                        type=float_1pcd,
                        help='Threshold in degrees Celsius, mm, or as percentile [default: 99].')

    parser.add_argument('--threshold-type',
                        dest='threshold_type',
                        type=str,
                        choices=['perc', 'abs'],
                        default='perc',
                        help='Pass "perc" (default) if percentiles should be used as thresholds or '
                             '"abs" for absolute thresholds.')

    parser.add_argument('--low-extreme',
                        dest='low_extreme',
                        default=False,
                        action='store_true',
                        help='Set if values should be lower than threshold to classify as extreme, '
                             'e.g. for cold extremes.')

    parser.add_argument('--inpath',
                        default='/data/arsclisys/normal/clim-hydro/TEA-Indicators/SPARTACUS/',
                        type=dir_path,
                        help='Path of folder where data is located.')

    parser.add_argument('--outpath',
                        default='/data/users/hst/TEA-clean/TEA/',
                        help='Path of folder where output data should be saved.')

    parser.add_argument('--statpath',
                        type=dir_path,
                        default='/data/arsclisys/normal/clim-hydro/TEA-Indicators/static/',
                        help='Path of folder where static file is located.')

    parser.add_argument('--maskpath',
                        type=dir_path,
                        default='/data/arsclisys/normal/clim-hydro/TEA-Indicators/masks/',
                        help='Path of folder where mask file is located.')

    parser.add_argument('--tmppath',
                        type=dir_path,
                        default='/home/hst/tmp_data/TEAclean/largeGR/',
                        help='Path of folder where tmp files should be stored. '
                             'Only relevant if large GR (> 100 areals) are processed with '
                             'ERA5(-Land) data.')

    parser.add_argument('--dataset',
                        dest='dataset',
                        default='SPARTACUS',
                        type=str,
                        choices=['SPARTACUS', 'ERA5', 'ERA5Land'],
                        help='Input dataset. Options: SPARTACUS (default), ERA5, ERA5Land.')

    parser.add_argument('--decadal',
                        dest='decadal',
                        default=False,
                        action='store_true',
                        help='Set if decadal TEA indicators should also be calculated. '
                             'Only possible if end - start >= 10.')

    parser.add_argument('--spreads',
                        dest='spreads',
                        default=False,
                        action='store_true',
                        help='Set if spread estimators of decadal TEA indicators should also '
                             'be calculated. Default: False.')

    parser.add_argument('--decadal-only',
                        dest='decadal_only',
                        default=False,
                        action='store_true',
                        help='Set if ONLY decadal TEA indicators should be calculated. '
                             'Only possible if CTP vars already calculated.')
    
    parser.add_argument('--recalc-daily',
                        dest='recalc_daily',
                        default=False,
                        action='store_true',
                        help='Set if daily basis variables should be recalculated. Default: False - read from file.')
    
    parser.add_argument('--compare-to-ref',
                        dest='compare_to_ref',
                        default=False,
                        action='store_true',
                        help='Set if results should be compared to reference file. Default: False.')
    
    parser.add_argument('--save-old',
                        dest='save_old',
                        default=False,
                        action='store_true',
                        help='Set if old output files should be saved. Default: False.')

    myopts = parser.parse_args()

    return myopts


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


def load_static_files(opts):
    """
    load GR masks and static file
    Args:
        opts: CLI parameter

    Returns:
        masks: GR masks (ds)
        static: ds with threshold, area_grid, etc.

    """

    masks = xr.open_dataset(f'{opts.maskpath}{opts.region}_masks_{opts.dataset}.nc')

    if 'LSM_EUR' in masks.data_vars:
        valid_cells = masks['lt1500_mask_EUR'].where(masks['LSM_EUR'].notnull())
        valid_cells = valid_cells.rename('valid_cells')
        masks['valid_cells'] = valid_cells
    elif opts.region == 'EUR':
        valid_cells = masks['lt1500_mask'].copy()
        valid_cells = valid_cells.rename('valid_cells')
        masks['valid_cells'] = valid_cells

    param_str = f'{opts.parameter}{opts.threshold:.1f}p'
    if opts.threshold_type == 'abs':
        param_str = f'{opts.parameter}{opts.threshold:.1f}{opts.unit}'

    static = xr.open_dataset(f'{opts.statpath}static_{param_str}_{opts.region}_{opts.dataset}.nc')

    return masks, static


def compare_to_ref(tea, ctp_filename_ref):
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
        for vvar in tea.CTP_results.data_vars:
            if vvar in tea_ref.CTP_results.data_vars:
                diff = tea.CTP_results[vvar] - tea_ref.CTP_results[vvar]
                max_diff = diff.max(skipna=True).values
                if max_diff > 1e-6:
                    logger.warning(f'Maximum difference in {vvar} is {max_diff}')
            else:
                logger.warning(f'{vvar} not found in reference file.')
    else:
        logger.warning(f'Reference file {ctp_filename_ref} not found.')


def save_output(opts, tea, masks):
    """
    save data arrays to output datasets
    Args:
        opts: CLI parameter
        ef: EF da
        ed: ED da
        em: EM da
        ea: EA da
        svars: suppl. vars da
        em_suppl: suppl. EM da
        masks: masks

    Returns:

    """
    # combine to output dataset
    ds_out = tea.CTP_results.copy()
    
    # set all values to 0 if EF is 0
    for vvar in ds_out.data_vars:
        if 'GR' in vvar:
            ds_out[vvar] = ds_out[vvar].where(tea.CTP_results.EF_GR != 0, 0)
        else:
            ds_out[vvar] = ds_out[vvar].where(tea.CTP_results.EF != 0, 0)

    mask = masks['lt1500_mask'] * masks['mask']
    # apply masks to grid data again (sum etc. result in 0 outside of region)
    for vvar in ds_out.data_vars:
        if 'GR' not in vvar:
            ds_out[vvar] = ds_out[vvar].where(mask == 1)

    ds_out = create_history(cli_params=sys.argv, ds=ds_out)

    path = Path(f'{opts.outpath}ctp_indicator_variables/supplementary/')
    path.mkdir(parents=True, exist_ok=True)
    
    outpath = (f'{opts.outpath}ctp_indicator_variables/'
               f'CTP_{opts.param_str}_{opts.region}_{opts.period}_{opts.dataset}'
               f'_{opts.start}to{opts.end}.nc')
    
    if opts.save_old:
        logger.info(f'Saving CTP indicators to {outpath}')
        ds_out.to_netcdf(outpath)
        
        # save supplementary variables
        logger.info('Saving supplementary variables')
        ds_out.to_netcdf(f'{opts.outpath}ctp_indicator_variables/supplementary/'
                         f'CTPsuppl_{opts.param_str}_{opts.region}_{opts.period}_{opts.dataset}'
                         f'_{opts.start}to{opts.end}.nc')
    
    outpath_new = outpath.replace('.nc', '_new.nc')
    path_ref = outpath.replace('.nc', '_new_ref.nc')
    
    logger.info(f'Saving CTP indicators to {outpath_new}')
    tea.save_CTP_results(outpath_new)
    
    if opts.compare_to_ref:
        compare_to_ref(tea, path_ref)


def calc_indicators(opts):
    """
    calculate the TEA indicators
    Args:
        opts: CLI parameter

    Returns:

    """

    data = get_data(opts=opts)

    # load GR masks and static file
    masks, static = load_static_files(opts=opts)

    # check if GR size is larger than 100 areals and switch to calc_TEA_largeGR if so
    if 'ERA' in opts.dataset and static['GR_size'] > 100:
        largeGR.calc_tea_large_gr(opts=opts, data=data, masks=masks, static=static)
        return

    # apply mask to data
    data = data * (masks['lt1500_mask'] * masks['mask'])

    # computation of daily basis variables (Methods chapter 3)
    if opts.recalc_daily:
        logger.info('Daily basis variables will be recalculated. Period set to annual.')
        old_period = opts.period
        opts.period = 'annual'
        tea = calc_daily_basis_vars(opts=opts, static=static, data=data)
        opts.period = old_period
    else:
        tea = TEAIndicators(input_data_grid=data, threshold_grid=static['threshold'], area_grid=static['area_grid'])
        
    dbv_filename_new = (f'{opts.outpath}/daily_basis_variables/DBV_{opts.param_str}_{opts.region}_annual'
                        f'_{opts.dataset}_{opts.start}to{opts.end}_new.nc')

    if not opts.recalc_daily:
        logger.info(f'Loading daily basis variables from {dbv_filename_new}; if you want to recalculate them, '
                    'set --recalc-daily.')
        tea.load_daily_results(dbv_filename_new)

    # apply criterion that DTEA_GR > DTEA_min and all GR variables use same dates,
    # dtea_min is given in areals (1 areal = 100 km2)
    dtea_min = 1
    tea.update_min_area(dtea_min)
    
    # calculate annual climatic time period indicators
    logger.info('Calculating CTP indicators')
    tea.calc_annual_CTP_indicators(opts.period)

    # save output
    save_output(opts=opts, tea=tea, masks=masks)
    return tea


def run():
    warnings.filterwarnings(action='ignore', message='All-NaN slice encountered')
    warnings.filterwarnings(action='ignore', message='divide by zero encountered in divide')
    warnings.filterwarnings(action='ignore', message='invalid value encountered in divide')

    # load CLI parameter
    opts = getopts()

    # add necessary strings to opts
    opts = extend_tea_opts(opts)

    # check length of input time span
    start = opts.start
    end = opts.end

    validate_period(opts)

    if not opts.decadal_only:
        if end - start > 10 - 1:
            starts = np.arange(start, end, 10)
            ends = np.append(np.arange(start + 10 - 1, end, 10), end)
            for pstart, pend in zip(starts, ends):
                opts.start = pstart
                opts.end = pend
                logger.info(f'Calculating TEA indicators for years {opts.start}-{opts.end}.')
                tea = calc_indicators(opts=opts)
                gc.collect()
        else:
            tea = calc_indicators(opts=opts)
    else:
        tea = TEAIndicators()

    if opts.decadal or opts.decadal_only:
        opts.start, opts.end = start, end
        logger.info(f'Calculating decadal-mean primary variables.')
        calc_decadal_indicators(opts=opts, tea=tea)


if __name__ == '__main__':
    run()
