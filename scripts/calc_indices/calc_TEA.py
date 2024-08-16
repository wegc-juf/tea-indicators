#!/opt/virtualenv3.7/bin/python3
# -*- coding: utf-8 -*-
"""
@author: hst
@reviewer: juf, 2024-07-29
"""

import argparse
import os

import cftime as cft
from datetime import timedelta
import glob
import numpy as np
import pandas as pd
from pathlib import Path
import warnings
import xarray as xr

from calc_daily_basis_vars import calc_daily_basis_vars

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
                        type=float,
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

    parser.add_argument('--decpath',
                        type=dir_path,
                        default='/home/hst/tmp_data/decadal_input/TEA-clean/',
                        help='Path of folder where decadal output data should be saved.')

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

    parser.add_argument('--dataset',
                        dest='dataset',
                        default='SPARTACUS',
                        type=str,
                        choices=['SPARTACUS', 'ERA5', 'ERA5Land'],
                        help='Input dataset. Options: SPARTACUS (default), ERA5, ERA5Land.')

    parser.add_argument('--gr-only',
                        dest='gr',
                        default=False,
                        action='store_true',
                        help='Set if output shall only contain GR variables. [default: False].')

    parser.add_argument('--decadal',
                        dest='decadal',
                        default=False,
                        action='store_true',
                        help='Set if output is meant for decadal index calculation '
                             '[default: False]. Only possible if period annual, WAS, JJA, or ESS '
                             'is chosen.')

    myopts = parser.parse_args()

    return myopts


def extend_opts(opts):
    """
    add strings that are often needed to opts
    Args:
        opts: CLI parameter

    Returns:

    """
    unit, unit_str = '°C', 'degC'
    if opts.parameter == 'P':
        unit, unit_str = 'mm', 'mm'

    pstr = opts.parameter
    if opts.parameter == 'P':
        pstr = opts.precip_var

    param_str = f'{pstr}{opts.threshold}p'
    if opts.threshold_type == 'abs':
        param_str = f'{pstr}{opts.threshold}{unit_str}'

    opts.unit = unit
    opts.unit_str = unit_str
    opts.param_str = param_str

    return opts


def validate_period(opts):
    valid_dec_periods = ['annual', 'WAS', 'ESS', 'JJA']
    if opts.decadal and opts.period not in valid_dec_periods:
        raise AttributeError(f'For decadal output, please select from {valid_dec_periods} as '
                             f'period! {opts.period} was passed instead.')


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
        pstr = opts.precip_var

    param_str = f'{pstr}{opts.threshold}p'
    if opts.threshold_type == 'abs':
        unit_str = 'degC'
        if opts.parameter == 'P':
            unit_str = 'mm'
        param_str = f'{pstr}{opts.threshold}{unit_str}'
    static = xr.open_dataset(f'{opts.statpath}static_{param_str}_{opts.region}_{opts.dataset}.nc')

    return masks, static


def resample_time(opts, dys):
    """
    create dictionary of all start & end dates, the chosen frequency and period
    Args:
        opts: CLI parameter
        dys: days array

    Returns:

    """

    freqs = {'annual': 'AS', 'seasonal': '3MS', 'WAS': 'AS-APR', 'ESS': 'AS-MAY', 'JJA': 'AS_JUN',
             'monthly': 'MS'}
    freq = freqs[opts.period]

    pstarts = pd.date_range(dys[0].values, dys[-1].values, freq=freq).to_series()
    if opts.period == 'WAS':
        pends = pd.date_range(dys[0].values, dys[-1].values, freq='A-OCT').to_series()
    elif opts.period == 'ESS':
        pends = pd.date_range(dys[0].values, dys[-1].values, freq='A-SEP').to_series()
    else:
        pends = pstarts - timedelta(days=1)
        pends[0:-1] = pends[1:]
        pends.iloc[-1] = dys[-1].values

    periods = {'start': pstarts, 'end': pends, 'freq': freq, 'period': opts.period}

    return periods


def calc_event_frequency(opts, periods, dteecs):
    """
    calculate event frequency (Eq. 11 & 12)
    Args:
        opts: CLI parameter
        periods: start and end dates of periods
        dteecs: daily threshold exceedance event count (gridded and GR)

    Returns:
        ef: event frequency
    """

    # TODO: create empty da for ef and ef_gr and fill it later
    ef = xr.DataArray(data=np.zeros((len(periods['start']), len(dteecs.y), len(dteecs.x))),
                      coords={'periods': (['periods'], periods['start']),
                              'x': (['x'], dteecs.x.data),
                              'y': (['y'], dteecs.y.data)})

    for iper, per in enumerate(periods['start']):
        pdata = dteecs.sel(days=slice(per, periods['end'][iper]))
        print()

    print()


def calc_indicators(opts):
    """
    calculate the TEA indicators
    Args:
        opts: CLI parameter

    Returns:

    """

    # data = get_data(opts=opts)

    # load GR masks and static file
    # masks, static = load_static_files(opts=opts)
    #
    # # apply mask to data
    # data = data * (masks['lt1500_mask'] * masks['mask'])

    # computation of daily basis variables (Methods chapter 3)
    # TODO: uncomment again later
    # calc_daily_basis_vars(opts=opts, static=static, data=data)
    dbv = xr.open_dataset(
        f'{opts.outpath}daily_basis_variables/'
        f'DBV_{opts.param_str}_{opts.region}_{opts.dataset}_{opts.start}to{opts.end}.nc')

    # apply criterion that DTEA_GR > DTEA_min and all GR variables use same dates
    dtea_min = 1
    for vvar in dbv.data_vars:
        if 'GR' in vvar:
            dbv[vvar] = dbv[vvar].where(dbv['DTEA_GR'] > dtea_min)

    # get dates for climatic time periods (CTP)
    pdates = resample_time(opts, dys=dbv.days)

    # calculate EF
    ef = calc_event_frequency(opts=opts, periods=pdates, dteecs=dbv[['DTEEC', 'DTEEC_GR']])


def run():
    warnings.filterwarnings(action='ignore', message='All-NaN slice encountered')
    # warnings.filterwarnings(action='ignore', message='invalid value encountered in true_divide')
    # warnings.filterwarnings(action='ignore', message='divide by zero encountered in true_divide')
    # warnings.filterwarnings(action='ignore', message='invalid value encountered in multiply')

    # load CLI parameter
    opts = getopts()

    # add necessary strings to opts
    opts = extend_opts(opts)

    # check length of input time span
    start = opts.start
    end = opts.end

    validate_period(opts)

    if end - start > 10 - 1:
        starts = np.arange(start, end, 10)
        ends = np.append(np.arange(start + 10 - 1, end, 10), end)
        for start, end in zip(starts, ends):
            opts.start = start
            opts.end = end
            print(f'{opts.start}-{opts.end}')
            calc_indicators(opts=opts)
    else:
        calc_indicators(opts=opts)


if __name__ == '__main__':
    run()
