#!/opt/virtualenv3.11/bin/python3
# -*- coding: utf-8 -*-
"""
@author: hst

"""

import argparse
import os
import copy

import glob
import logging
import pandas as pd
from pathlib import Path
import sys
import xarray as xr

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

sys.path.append('/home/hst/tea-indicators/scripts/misc/')
from general_functions import create_history, ref_cc_params
from calc_TEA import extend_opts, assign_ctp_coords
from calc_daily_basis_vars import calc_dteec_1d

PARAMS = ref_cc_params()

def getopts():
    """
    get arguments
    :return: command line parameters
    """

    def dir_path(path):
        if os.path.isdir(path):
            return path
        else:
            raise argparse.ArgumentTypeError(f'{path} is not a valid path!')

    parser = argparse.ArgumentParser()

    parser.add_argument('--period',
                        dest='period',
                        default='WAS',
                        type=str,
                        choices=['monthly', 'seasonal', 'annual', 'WAS', 'ESS', 'JJA'],
                        help='Climatic time period (CTP) of interest. '
                             'Options: monthly, seasonal, WAS, ESS, JJA, and  annual [default].')

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

    parser.add_argument('--station',
                        default='Graz',
                        type=str,
                        choices=['Graz', 'Innsbruck', 'Salzburg', 'Kremsmuenster', 'Wien',
                                 'BadGleichenberg', 'Deutschlandsberg'],
                        help='Station to use.')

    parser.add_argument('--parameter',
                        default='T',
                        type=str,
                        choices=['T', 'P'],
                        help='Parameter for which the TEA indices should be calculated '
                             'Options: T (= temperature, default), P (= precipitation).')

    parser.add_argument('--inpath',
                        default='/data/users/hst/cdrDPS/station_data/',
                        type=dir_path,
                        help='Path of folder where data is located.')

    parser.add_argument('--outpath',
                        default='/data/users/hst/TEA-clean/TEA/',
                        type=dir_path,
                        help='Path of folder where output data should be saved.')

    myopts = parser.parse_args()

    return myopts


def load_data(opts):
    """
    load station data
    Args:
        opts: CLI parameter

    Returns:
        data: interpolated station data

    """

    if opts.parameter == 'T':
        pstr = 'Tmax'
        rename_dict = {'tmax': 'T'}
    else:
        pstr = 'RR'
        rename_dict = {'nied': 'P'}

    # read csv file of station data and set time as index of df
    try:
        filenames = f'{opts.inpath}{pstr}_{opts.station}*18770101*.csv'
        file = glob.glob(filenames)
    except FileNotFoundError:
        filenames = f'{opts.inpath}{pstr}_{opts.station}*.csv'
        file = glob.glob(filenames)
    data = pd.read_csv(file[0])
    data['time'] = pd.to_datetime(data['time'])
    data = data.set_index('time')

    # rename columns
    data = data.rename(columns=rename_dict)

    # only select the years 1877-2022
    data = data.loc['1877-01-01':'2023-01-01']

    if opts.period == 'WAS':
        data = data[(data.index.month >= 4) & (data.index.month <= 10)]
    elif opts.period == 'ESS':
        data = data[(data.index.month >= 5) & (data.index.month <= 9)]

    # interpolate missing data
    data = interpolate_gaps(opts=opts, data=data)

    return data


def interpolate_gaps(opts, data):
    """
    interpolates data gaps with average of missing day from other years
    Args:
        opts: CLI parameter
        data: station data

    Returns:
        data: interpolated data
    """

    non_nan = data.loc[data[opts.parameter].notnull(), :]
    start_yr = non_nan.index[0]

    gaps = data[data[opts.parameter].isnull()]
    for igap in gaps.index:
        if igap < start_yr:
            continue
        # select all values from that day of year
        day_data = data[data.index.month == igap.month]
        day_data = day_data[day_data.index.day == igap.day]
        # calculate mean
        fill_val = day_data[opts.parameter].mean(skipna=True)
        # fill gap with fill value
        data.at[igap, opts.parameter] = fill_val

    return data


def calc_thresh(opts):
    opts_ref = copy.deepcopy(opts)
    opts_ref.period = 'annual'

    data = load_data(opts=opts_ref)
    ref_data = data[(data.index.year >= int(PARAMS['REF']['start'][:4]))
                    & (data.index.year <= int(PARAMS['REF']['end'][:4]))]

    if opts.threshold_type == 'abs':
        thresh = opts.threshold
    else:
        thresh = ref_data[opts.parameter].quantile(opts.threshold / 100)

    return thresh


def calc_basis(opts, data):
    """
    calculate daily basis variables
    Args:
        opts: CLI parameter
        data: daily station data

    Returns:
        basics: daily basis variables
        data: updated data

    """

    thresh = calc_thresh(opts)

    # equation 01
    data['dtec'] = 0.0
    data.loc[data[opts.parameter] > thresh, 'dtec'] = 1.0
    # equation 07
    data['dtem'] = 0.0
    data.loc[data['dtec'] == 1, 'dtem'] = data[opts.parameter] - thresh

    dtem = data['dtem'].values

    dtec = data['dtec'].where(data['dtec'] > 0).values

    # calc dteec
    dteec = calc_dteec_1d(dtec_cell=dtec)

    basics = xr.Dataset(data_vars={'DTEC': (['days'], dtec),
                                   'DTEM': (['days'], dtem),
                                   'DTEEC': (['days'], dteec)},
                        coords={'days': (['days'], data.index.values)})

    return basics, data


def calc_ctp_indicators(opts, data):

    pdata = data.sum('days')

    ef = pdata.DTEEC
    ed = pdata['DTEC']
    ed_avg = ed / ef
    if opts.parameter == 'T':
        em_avg = pdata['DTEM'] / ed
        data_unit = '°C'
    else:
        em_avg = data.median('days')['DTEM']
        data_unit = 'mm'

    # add attributes and combine to one dataset
    ef = ef.rename('EF')
    ef = ef.assign_attrs({'long_name': 'event frequency', 'units': '1'})

    ed_avg = ed_avg.rename('EDavg')
    ed_avg.attrs = {'long_name': 'average events duration', 'units': 'dys'}

    em_avg = em_avg.rename('EMavg')
    em_avg.attrs = {'long_name': 'average exceedance magnitude', 'units': data_unit}

    ctp = xr.merge([ef, ed_avg, em_avg])

    return ctp


def run():
    opts = getopts()
    if opts.parameter == 'P':
        opts.precip_var = 'P24h_7to7'
    opts = extend_opts(opts=opts)

    data = load_data(opts=opts)

    # calc daily basis variables
    dbv, data = calc_basis(opts=opts, data=data)
    dbv, dbv_per = assign_ctp_coords(opts=opts, data=dbv)

    # calc CTP variables
    ctp = calc_ctp_indicators(opts=opts, data=dbv_per)

    # save output
    ds_out = create_history(cli_params=sys.argv, ds=ctp)
    path = Path(f'{opts.outpath}station_indices/')
    path.mkdir(parents=True, exist_ok=True)
    ds_out.to_netcdf(f'{opts.outpath}station_indices/CTP_{opts.param_str}_{opts.station}.nc')


if __name__ == '__main__':
    run()
