#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: hst
"""

import argparse
from datetime import timedelta
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

from scripts.general_stuff.general_functions import create_history, extend_tea_opts
from scripts.general_stuff.var_attrs import get_attrs
from scripts.general_stuff.TEA_logger import logger
from scripts.calc_indices.calc_daily_basis_vars import calc_daily_basis_vars, calculate_event_count
from scripts.calc_indices.calc_ctp_indicator_variables import (calc_event_frequency,
                                                               calc_supplementary_event_vars,
                                                               calc_event_duration,
                                                               calc_exceedance_magnitude,
                                                               calc_exceedance_area_tex_sev)
from scripts.calc_indices.calc_decadal_indicators import calc_decadal_indicators
import scripts.calc_indices.calc_TEA_largeGR as largeGR

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
            raise argparse.ArgumentTypeError(f'{path} is not a valid path')

    def float_1pcd(value):
        if not re.match(r'^\d+(\.\d{1})?$', value):
            raise argparse.ArgumentTypeError('Threshold value must have at most one digit after '
                                             'the decimal point')
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
                        choices=['monthly', 'seasonal', 'annual', 'WAS', 'ESS', 'JJA'],
                        help='Climatic time period (CTP) of interest. '
                             'Options: monthly, seasonal, WAS, ESS, JJA, and  annual [default].')

    parser.add_argument('--region',
                        default='AUT',
                        type=str,
                        help='GeoRegion. Options: EUR, AUT (default), SAR, SEA, FBR, '
                             'Austrian state, or ISO2 code of a european country.')

    parser.add_argument('--parameter',
                        default='T',
                        type=str,
                        choices=['T', 'P'],
                        help='Parameter for which the TEA indices should be calculated '
                             'Options: T (= temperature, default), P (= precipitation).')

    parser.add_argument('--precip_var',
                        default='Px1h_7to7',
                        type=str,
                        choices=['Px1h', 'P24h', 'Px1h_7to7', 'P24h_7to7'],
                        help='Precipitation variable used.'
                             '[Px1h, P24h, Px1h_7to7 (default), P24h_7to7]')

    parser.add_argument('--threshold',
                        default=99,
                        type=float_1pcd,
                        help='Threshold in degrees Celsius, mm, or as percentile [default: 99].')

    parser.add_argument('--threshold_type',
                        type=str,
                        choices=['perc', 'abs'],
                        default='perc',
                        help='Pass "perc" (default) if percentiles should be used as thresholds or '
                             '"abs" for absolute thresholds.')

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

    parser.add_argument('--decadal_only',
                        dest='decadal_only',
                        default=False,
                        action='store_true',
                        help='Set if ONLY decadal TEA indicators should be calculated. '
                             'Only possible if CTP vars already calculated.')

    myopts = parser.parse_args()

    return myopts


def validate_period(opts):
    valid_dec_periods = ['annual', 'WAS', 'ESS', 'JJA']
    if opts.decadal and opts.period not in valid_dec_periods:
        raise AttributeError(f'For decadal output, please select from {valid_dec_periods} as '
                             f'period! {opts.period} was passed instead.')

    if opts.decadal or opts.decadal_only:
        if opts.end - opts.start < 9:
            raise AttributeError(f'For decadal output, please pass more at least 10 years! '
                                 f'{(opts.end - opts.start) + 1} years were passed instead.')


def get_data(opts):
    """
    loads data
    :param opts: input parameter
    :return: dataset of daily maximum temperature or precipitation
    """

    params = {'SPARTACUS': {'T': 'Tx', 'P': 'RR'}, 'ERA5': {'T': '', 'P': ''},
              'ERA5Land': {'T': '', 'P': ''}}

    # select only files of interest, if chosen period is 'seasonal' append one year in the
    # beginning to have the first winter fully included
    filenames = []
    if opts.period == 'seasonal' and opts.start != '1961':
        yrs = np.arange(opts.start - 1, opts.end + 1)
    else:
        yrs = np.arange(opts.start, opts.end + 1)
    for iyrs in yrs:
        year_files = sorted(glob.glob(
            f'{opts.inpath}*{params[opts.dataset][opts.parameter]}_{iyrs}*.nc'))
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
    if opts.parameter == 'T':
        var = 'Tx'
    else:
        if opts.dataset == 'SPARTACUS':
            var = 'RR'
        else:
            var = opts.precip_var
    data = ds[var]
    data = data.rename(time='days')

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

    pstr = opts.parameter
    if opts.parameter == 'P':
        pstr = f'{opts.precip_var}_'

    param_str = f'{pstr}{opts.threshold}p'
    if opts.threshold_type == 'abs':
        unit_str = 'degC'
        if opts.parameter == 'P':
            unit_str = 'mm'
        param_str = f'{pstr}{opts.threshold}{unit_str}'
    static = xr.open_dataset(f'{opts.statpath}static_{param_str}_{opts.region}_{opts.dataset}.nc')

    return masks, static


def assign_ctp_coords(opts, data):
    """
    create dictionary of all start & end dates, the chosen frequency and period
    Args:
        opts: CLI parameter
        data: data array

    Returns:

    """

    freqs = {'annual': 'AS', 'seasonal': '3MS', 'WAS': 'AS-APR', 'ESS': 'AS-MAY', 'JJA': 'AS-JUN',
             'monthly': 'MS'}
    freq = freqs[opts.period]

    pstarts = pd.date_range(data.days[0].values, data.days[-1].values,
                            freq=freq).to_series()
    if opts.period == 'WAS':
        pends = pd.date_range(data.days[0].values, data.days[-1].values,
                              freq='A-OCT').to_series()
    elif opts.period == 'ESS':
        pends = pd.date_range(data.days[0].values, data.days[-1].values,
                              freq='A-SEP').to_series()
    else:
        pends = pstarts - timedelta(days=1)
        pends[0:-1] = pends[1:]
        pends.iloc[-1] = data.days[-1].values

    # add ctp as coordinates to enable using groupby later
    # map the 'days' coordinate to 'ctp'
    def map_to_ctp(dy, starts, ends):
        for start, end, ctp in zip(starts, ends, starts):
            if start <= dy <= end:
                return ctp
        return np.nan

    days_to_ctp = []
    for day in data.days.values:
        ctp_dy = map_to_ctp(dy=day, starts=pstarts, ends=pends)
        days_to_ctp.append(ctp_dy)

    data.coords['ctp'] = ('days', days_to_ctp)

    # group into CTPs
    data_per = data.groupby('ctp')

    return data, data_per


def save_output(opts, ef, ed, em, ea, svars, em_suppl, masks):
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
    ds_out = xr.merge([ef, ed, em, ea])
    ds_out_suppl = xr.merge([svars, em_suppl])

    ctp_attrs = get_attrs(opts=opts, vname='ctp')

    ds_out['ctp'] = ds_out['ctp'].assign_attrs(ctp_attrs)
    ds_out_suppl['ctp'] = ds_out_suppl['ctp'].assign_attrs(ctp_attrs)

    mask = masks['lt1500_mask'] * masks['mask']
    # apply masks to grid data again (sum etc. result in 0 outside of region)
    for vvar in ds_out.data_vars:
        if 'GR' not in vvar:
            ds_out[vvar] = ds_out[vvar].where(mask == 1)
    for vvar in ds_out_suppl.data_vars:
        if 'GR' not in vvar:
            ds_out_suppl[vvar] = ds_out_suppl[vvar].where(mask == 1)

    ds_out = create_history(cli_params=sys.argv, ds=ds_out)
    ds_out_suppl = create_history(cli_params=sys.argv, ds=ds_out_suppl)

    path = Path(f'{opts.outpath}ctp_indicator_variables/supplementary/')
    path.mkdir(parents=True, exist_ok=True)
    ds_out.to_netcdf(f'{opts.outpath}ctp_indicator_variables/'
                     f'CTP_{opts.param_str}_{opts.region}_{opts.period}_{opts.dataset}'
                     f'_{opts.start}to{opts.end}.nc')
    ds_out_suppl.to_netcdf(f'{opts.outpath}ctp_indicator_variables/supplementary/'
                           f'CTPsuppl_{opts.param_str}_{opts.region}_{opts.period}_{opts.dataset}'
                           f'_{opts.start}to{opts.end}.nc')


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
    calc_daily_basis_vars(opts=opts, static=static, data=data)
    dbv = xr.open_dataset(
        f'{opts.outpath}daily_basis_variables/'
        f'DBV_{opts.param_str}_{opts.region}_{opts.period}_{opts.dataset}'
        f'_{opts.start}to{opts.end}.nc')

    # apply criterion that DTEA_GR > DTEA_min and all GR variables use same dates,
    # dtea_min is given in areals (1 areal = 100 km2)
    dtea_min = 1
    for vvar in dbv.data_vars:
        if vvar == 'DTEEC_GR':
            # Amin criterion sometimes splits up events --> run DTEEC_GR detection again
            dbv[vvar] = calculate_event_count(opts=opts, dtec=dbv['DTEC_GR'], da_out=True, cstr='')
        elif 'GR' in vvar:
            dbv[vvar] = dbv[vvar].where(dbv['DTEA_GR'] > dtea_min)

    # get dates for climatic time periods (CTP) and assign coords to dbv
    dbv, dbv_per = assign_ctp_coords(opts, data=dbv)

    # calculate EF and corresponding supplementary variables
    ef = calc_event_frequency(pdata=dbv_per)
    svars = calc_supplementary_event_vars(data=dbv)

    # calculate ED
    ed = calc_event_duration(pdata=dbv_per, ef=ef)

    # calculate EM
    em, em_suppl = calc_exceedance_magnitude(opts=opts, pdata=dbv_per, ed=ed)

    # calculate EA
    ea = calc_exceedance_area_tex_sev(opts=opts, data=dbv, ed=ed, em=em)

    # save output
    save_output(opts=opts, ef=ef, ed=ed, em=em, ea=ea, svars=svars, em_suppl=em_suppl, masks=masks)


def run():
    warnings.filterwarnings(action='ignore', message='All-NaN slice encountered')
    warnings.filterwarnings(action='ignore', message='divide by zero encountered in divide')

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
                calc_indicators(opts=opts)
                gc.collect()
        else:
            calc_indicators(opts=opts)

    if opts.decadal or opts.decadal_only:
        opts.start, opts.end = start, end
        logger.info(f'Calculating decadal-mean primary variables.')
        calc_decadal_indicators(opts=opts, suppl=False)
        logger.info(f'Calculating decadal-mean supplementary variables.')
        calc_decadal_indicators(opts=opts, suppl=True)


if __name__ == '__main__':
    run()
