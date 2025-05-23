#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: hst

"""

import copy
import logging
import os
from datetime import timedelta

import numpy as np
import pandas as pd
from pathlib import Path
import sys
import xarray as xr

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from scripts.general_stuff.general_functions import (load_opts, create_history_from_cfg,
                                                     ref_cc_params, get_csv_data)
from scripts.general_stuff.var_attrs import get_attrs
from scripts.calc_indices.calc_TEA import _getopts, calc_dbv_indicators, _save_ctp_output

PARAMS = ref_cc_params()


def calc_station_tea_indicators(opts):
    """
    calculate TEA indicators for station data
    Args:
        opts: options as defined in CFG-PARAMS-doc.md and TEA_CFG_DEFAULT.yaml

    Returns:

    """
    tea = calc_dbv_indicators(start=opts.start, end=opts.end, opts=opts, gridded=False, threshold=None)
    tea.calc_annual_ctp_indicators(opts.period, drop_daily_results=True)
    _save_ctp_output(opts=opts, tea=tea, start=opts.start, end=opts.end)


def run():
    # get command line parameters
    cmd_opts = _getopts()
    
    # load CFG parameters
    opts = load_opts(fname=__file__, config_file=cmd_opts.config_file)

    if opts.parameter == 'T':
        opts.param_str = f'Tx{opts.threshold:.1f}p'
    
    calc_station_tea_indicators(opts)


if __name__ == '__main__':
    run()
