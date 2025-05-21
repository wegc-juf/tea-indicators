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
from scripts.calc_indices.calc_TEA import _getopts

PARAMS = ref_cc_params()


def assign_ctp_coords(opts, data):
    """
    create dictionary of all start & end dates, the chosen frequency and period
    Args:
        opts: CLI parameter
        data: data array

    Returns:

    """
    
    pd_major, pd_minor = pd.__version__.split('.')[:2]
    if int(pd_major) > 2 or (int(pd_major) == 2 and int(pd_minor) >= 2):
        freqs = {'annual': 'YS', 'seasonal': '3MS', 'WAS': 'YS-APR', 'ESS': 'YS-MAY', 'JJA': 'YS-JUN',
                 'monthly': 'MS', 'EWS': 'YS-NOV'}
    else:
        freqs = {'annual': 'AS', 'seasonal': '3MS', 'WAS': 'AS-APR', 'ESS': 'AS-MAY', 'JJA': 'AS-JUN',
                 'monthly': 'MS', 'EWS': 'AS-NOV'}
    
    freq = freqs[opts.period]
    
    pstarts = pd.date_range(data.time[0].values, data.time[-1].values,
                            freq=freq).to_series()
    if opts.period == 'WAS':
        pends = pd.date_range(data.time[0].values, data.time[-1].values,
                              freq='A-OCT').to_series()
    elif opts.period == 'ESS':
        pends = pd.date_range(data.time[0].values, data.time[-1].values,
                              freq='A-SEP').to_series()
    else:
        pends = pstarts - timedelta(days=1)
        pends[0:-1] = pends[1:]
        pends.iloc[-1] = data.time[-1].values
    
    # add ctp as coordinates to enable using groupby later
    # map the 'time' coordinate to 'ctp'
    def map_to_ctp(dy, starts, ends):
        for start, end, ctp in zip(starts, ends, starts):
            if start <= dy <= end:
                return ctp
        return np.nan
    
    days_to_ctp = []
    for day in data.time.values:
        ctp_dy = map_to_ctp(dy=day, starts=pstarts, ends=pends)
        days_to_ctp.append(ctp_dy)
    
    data.coords['ctp'] = ('time', days_to_ctp)
    
    # group into CTPs
    data_per = data.groupby('ctp')
    
    return data, data_per


def calc_thresh(opts):
    opts_ref = copy.deepcopy(opts)
    opts_ref.period = 'annual'

    data = get_csv_data(opts=opts_ref)
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

    basics['DTEM'] = basics['DTEM'].where(basics['DTEM'] > 0)

    return basics


def calc_ctp_indicators(opts, data):

    pdata = data.sum('days')

    ef = pdata.DTEEC
    ed = pdata['DTEC']
    ed_avg = ed / ef
    if opts.parameter == 'T':
        em_avg = pdata['DTEM'] / ed
    else:
        em_avg = data.median('days')['DTEM']
        em_avg = em_avg.interpolate_na(dim='ctp')

    # add attributes and combine to one dataset
    ef = ef.rename('EF')
    ef = ef.assign_attrs(get_attrs(vname='EF'))

    ed_avg = ed_avg.rename('EDavg')
    ed_avg.attrs = get_attrs(vname='EDavg')

    em_avg = em_avg.rename('EMavg')
    em_avg.attrs = get_attrs(vname='EMavg')

    ctp = xr.merge([ef, ed_avg, em_avg])

    return ctp


def calc_station_tea_indicators(opts):
    """
    calculate CTP indicators for station data
    Args:
        opts: options as defined in CFG-PARAMS-doc.md and TEA_CFG_DEFAULT.yaml

    Returns:

    """
    # tea = calc_dbv_indicators(start=opts.start, end=opts.end, threshold=25., opts=opts, gridded=False)
    
    data = get_csv_data(opts=opts)
    # calc daily basis variables
    dbv = calc_basis(opts=opts, data=data)
    dbv, dbv_per = assign_ctp_coords(opts=opts, data=dbv)
    # calc CTP variables
    ctp = calc_ctp_indicators(opts=opts, data=dbv_per)
    # set all other vars to 0 if EF is 0
    ctp = ctp.where(ctp.EF != 0, 0)
    # save output
    ds_out = create_history_from_cfg(cfg_params=opts, ds=ctp)
    path = Path(f'{opts.outpath}station_indices/')
    path.mkdir(parents=True, exist_ok=True)
    ds_out.to_netcdf(f'{opts.outpath}station_indices/CTP_{opts.param_str}_{opts.station}.nc')


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
