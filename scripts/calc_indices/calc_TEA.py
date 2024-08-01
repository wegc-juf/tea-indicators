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
import warnings
import xarray as xr


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
                        default='/data/users/hst/TEA-clean/SPARTACUS/',
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
                        default='/data/users/hst/TEA-clean/static/',
                        help='Path of folder where static files are located.')

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
            f'{opts.inpath}{params[opts.dataset][opts.parameter]}{iyrs}*.nc'))
        filenames.extend(year_files)

    # load relevant years
    try:
        ds = xr.open_mfdataset(filenames, combine='by_coords')
    except ValueError:
        ds = xr.open_dataset(filenames[0])

    # select only times of interest
    if opts.period == 'seasonal' and opts.start != '1961':
        start = f'{opts.start-1}-12-01'
        end = f'{opts.end}-11-30'
        ds = ds.sel(time=slice(start, end))
    elif opts.period == 'seasonal' and opts.start == '1961':
        # if first year is first year of record, exclude first winter (data of Dec 1960 missing)
        start = f'{opts.start-1}-03-01'
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

    return data


def calc_indicators(opts):
    """
    calculate the TEA indicators
    Args:
        opts: CLI parameter

    Returns:

    """

    data = get_data(opts=opts)
    print()


def run():
    # warnings.filterwarnings(action='ignore', message='All-NaN slice encountered')
    # warnings.filterwarnings(action='ignore', message='invalid value encountered in true_divide')
    # warnings.filterwarnings(action='ignore', message='divide by zero encountered in true_divide')
    # warnings.filterwarnings(action='ignore', message='invalid value encountered in multiply')

    # load CLI parameter
    opts = getopts()

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