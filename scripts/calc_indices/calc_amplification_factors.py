#!/opt/virtualenv3.7/bin/python3
# -*- coding: utf-8 -*-
"""
@author: hst
"""

import argparse
from datetime import timedelta
import glob
import logging
import numpy as np
import os
import pandas as pd
from pathlib import Path
import sys
import warnings
import xarray as xr

sys.path.append('/home/hst/tea-indicators/scripts/misc/')
from general_functions import create_history
from calc_daily_basis_vars import calc_daily_basis_vars, calculate_event_count
from calc_ctp_indicator_variables import (calc_event_frequency, calc_supplementary_event_vars,
                                          calc_event_duration, calc_exceedance_magnitude,
                                          calc_exceedance_area_tex_sev)
from calc_decadal_indicators import calc_decadal_indicators

logging.basicConfig(
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
                        default='/data/users/hst/TEA-clean/TEA/',
                        type=dir_path,
                        help='Path of folder where TEA data is located.')

    parser.add_argument('--outpath',
                        default='/data/users/hst/TEA-clean/TEA/',
                        help='Path of folder where output data should be saved.')

    parser.add_argument('--dataset',
                        dest='dataset',
                        default='SPARTACUS',
                        type=str,
                        choices=['SPARTACUS', 'ERA5', 'ERA5Land'],
                        help='Input dataset. Options: SPARTACUS (default), ERA5, ERA5Land.')

    myopts = parser.parse_args()

    return myopts



if __name__ == '__main__':
    run()
