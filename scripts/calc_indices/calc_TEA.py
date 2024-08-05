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


def calc_daily_basis_vars(opts, static, data):
    """
    compute daily basis variables following chapter 3 of TEA methods
    Args:
        opts: CLI parameter
        static: static ds
        data: data

    Returns:
        basic_vars: ds with daily basis variables (DTEC, DTEM, DTEA) both gridded and for GR
        dtem_max: da daily maximum threshold exceedance magnitude

    """

    if opts.parameter == 'T':
        data_unit = '°C'
    else:
        data_unit = 'mm'

    # set minimum tea (unit: 100 km2)
    tea_min = 1

    # calculate DTEM
    # equation 07
    dtem = data - static['threshold']
    dtem = dtem.where(dtem > 0)
    dtem = dtem.rename('DTEM')
    dtem.attrs = {'long_name': 'daily threshold exceedance magnitude', 'units': data_unit}

    # equation 01
    # store DTEM for all DTEC == 1
    dtec = dtem.where(dtem.isnull(), 1)
    dtec = dtec.rename('DTEC')
    dtec.attrs = {'long_name': 'daily threshold exceedance count', 'units': '1'}

    # equation 02_1 not needed (cells with TEC == 0 are already nan in tem)
    # equation 02_2
    dtea = dtec * static['area_grid']

    # equation 06
    # calculate DTEA_GR
    dtea_gr = dtea.sum(axis=(1, 2), skipna=True)
    dtea_gr = dtea_gr.rename('DTEA_GR')
    dtea_gr = dtea_gr.assign_attrs({'long_name': 'daily threshold exceedance area',
                                    'units': 'areals'})

    # equation 03
    # if DTEA < 1, set DTEC and DTEM to nan --> exceedance area needs to be greater
    # than 100 km2 (1 areal) in order to keep the day as an exceedance day also replace 0 in
    # dtea_gr by nan
    dtec = dtec.where(dtea_gr > tea_min)
    dtem = dtem.where(dtea_gr > tea_min)
    dtea_gr = dtea_gr.where(dtea_gr > tea_min)
    area_frac = (dtea_gr / static['GR_size']) * 100
    area_frac = area_frac.rename('DTEA_frac')

    # calculate dtec_gr (continues equation 03)
    dtec_gr = dtec.notnull().any(dim=static['threshold'].dims)
    dtec_gr = dtec_gr.where(dtec_gr == True)
    dtec_gr = dtec_gr.rename(f'{dtec.name}_GR')
    dtec_gr = dtec_gr.assign_attrs({'long_name': 'daily threshold exceedance count (GR)',
                                    'units': '1'})

    # equation 08
    # calculate dtem_gr (area weighted DTEM)
    area_fac = static['area_grid'] / dtea_gr.T
    dtem_gr = (dtem * area_fac).sum(axis=(1, 2), skipna=True)
    dtem_gr = dtem_gr.rename(f'{dtem.name}_GR')
    dtem_gr = dtem_gr.assign_attrs({'long_name': 'daily threshold exceedance magnitude (GR)',
                                    'units': data_unit})

    # equation 09
    # save maximum DTEM
    dtem_max = dtem.max(dim=static['threshold'].dims)
    dtem_max = dtem_max.assign_attrs({'long_name': 'daily maximum grid cell exceedance magnitude',
                                     'units': data_unit})

    # equations 4 and 5
    # calculate DTEEC(_GR)
    dteec = calculate_event_count(dtec=dtec)
    dteec_gr = calculate_event_count(dtec=dtec_gr)

    # combine all basic variables (except DTEM_max) into one ds
    basic_vars = xr.merge((dtec, dtec_gr, dteec, dteec_gr, dtem, dtem_gr, dtea_gr, area_frac))

    return basic_vars, dtem_max


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


def calc_dteec_1d(dtec_cell):
    # Convert to a NumPy array and change NaN to 0
    dtec_np = np.nan_to_num(dtec_cell, nan=0)

    # Find the starts and ends of sequences (change NaNs to 0 before the diff operation)
    change = np.diff(np.concatenate(
        ([np.zeros((1,) + dtec_np.shape[1:]), dtec_np, np.zeros((1,) + dtec_np.shape[1:])]),
        axis=0), axis=0)
    starts = np.where(change == 1)
    ends = np.where(change == -1)

    # Calculate the middle points (as flat indices)
    middle_indices = (starts[0] + ends[0] - 1) // 2

    # Create an output array filled with NaNs
    events_np = np.full(dtec_cell.shape, np.nan)

    # Set the middle points to 1 (use flat indices to index into the 3D array)
    events_np[middle_indices] = 1

    return events_np


def calculate_event_count(dtec):
    """
    calculate DTEEC(_GR) according to equations 4 and 5
    Args:
        dtec: daily threshold exceedance count

    Returns:

    """

    if 'GR' in dtec.name:
        dteec_np = calc_dteec_1d(dtec_cell=dtec.values)
        dteec = xr.DataArray(dteec_np, coords=dtec.coords, dims=dtec.dims)
        gr_str, gr_var_str = ' (GR)', '_GR'
    else:
        dteec = xr.full_like(dtec, np.nan)
        dtec_3d = dtec.values
        # loop through all rows and calculate DTEEC
        for iy in range(len(dtec_3d[0, :, 0])):
            dtec_row = dtec_3d[:, iy, :]
            dteec_row = np.apply_along_axis(calc_dteec_1d, axis=0, arr=dtec_row)
            dteec[:, iy, :] = dteec_row
        gr_str, gr_var_str = '', ''

    dteec = dteec.rename(f'DTEEC{gr_var_str}')
    dteec.attrs = {'long_name': f'daily threshold exceedance event count{gr_str}', 'units': '1'}

    return dteec


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

    # apply mask to data
    data = data * (masks['lt1500_mask'] * masks['mask'])

    # computation of daily basis variables (Methods chapter 3)
    dbv, dtem_max = calc_daily_basis_vars(opts=opts, static=static, data=data)

    # get dates for periods
    pdates = resample_time(opts, dys=dbv.days)

    # calculate EF


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
