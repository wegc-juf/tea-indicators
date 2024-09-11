#!/opt/virtualenv3.7/bin/python3
# -*- coding: utf-8 -*-
"""
@author: hst
"""

import argparse
from datetime import timedelta
import gc
import glob
import logging
import numpy as np
import os
import pandas as pd
from pathlib import Path
import re
import sys
import warnings
import xarray as xr

sys.path.append('/home/hst/tea-indicators/scripts/misc/')
from general_functions import create_history
from calc_TEA import extend_opts

logging.basicConfig(
    filename='LOGFILE_calc_TEA.log',
    encoding='utf-8',
    filemode='a',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

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

    parser.add_argument('--agr',
                        default='EUR',
                        type=str,
                        choices=['EUR', 'S-EUR', 'C-EUR', 'N-EUR'],
                        help='Aggregate GeoRegion. Options: EUR, S-EUR, C-EUR, and N-EUR.')

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
                        default='/data/users/hst/TEA-clean/TEA/dec_indicator_variables/',
                        type=dir_path,
                        help='Path of folder where data is located.')

    parser.add_argument('--outpath',
                        default='/data/users/hst/TEA-clean/TEA/dec_indicator_variables/',
                        help='Path of folder where output data should be saved.')

    parser.add_argument('--dataset',
                        dest='dataset',
                        default='SPARTACUS',
                        type=str,
                        choices=['SPARTACUS', 'ERA5', 'ERA5Land'],
                        help='Input dataset. Options: SPARTACUS (default), ERA5, ERA5Land.')

    parser.add_argument('--spreads',
                        dest='spreads',
                        default=False,
                        action='store_true',
                        help='Set if spread estimators of decadal TEA indicators should also '
                             'be calculated. Default: False.')

    myopts = parser.parse_args()

    return myopts


def load_data(opts):
    """
    load dec data and select AGR lats
    Args:
        opts: CLI parameter

    Returns:
        ds: decadal data of AGR
    """

    file = (f'{opts.inpath}DEC_{opts.paramstr}_EUR_{opts.period}_{opts.dataset}'
            f'_{opts.start}to{opts.end}.nc')
    ds = xr.open_dataset(file)

    agr_lims = {'EUR': [35, 70], 'S-EUR': [35, 44.5], 'C-EUR': [45, 55], 'N-EUR': [55.5, 70]}

    ds = ds.sel(lat=slice(agr_lims[opts.agr][1], agr_lims[opts.agr][0]))

    return ds


def run():
    opts = getopts()

    # add necessary strings to opts
    opts = extend_opts(opts)

    data = load_data(opts=opts)


if __name__ == '__main__':
    run()
