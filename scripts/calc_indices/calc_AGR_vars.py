#!/opt/virtualenv3.7/bin/python3
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

sys.path.append('/home/hst/tea-indicators/scripts/misc/')
from general_functions import create_history, extend_tea_opts, ref_cc_params
from calc_amplification_factors import calc_ref_cc_mean
from TEA_AGR_logger import logger

DS_PARAMS = {'SPARTACUS': {'xname': 'x', 'yname': 'y'},
             'ERA5': {'xname': 'lon', 'yname': 'lat'},
             'ERA5Land': {'xname': 'lon', 'yname': 'lat'}}

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

    parser.add_argument('--statpath',
                        type=dir_path,
                        default='/data/arsclisys/normal/clim-hydro/TEA-Indicators/static/',
                        help='Path of folder where static file is located.')

    parser.add_argument('--dataset',
                        dest='dataset',
                        default='ERA5',
                        type=str,
                        choices=['ERA5', 'ERA5Land'],
                        help='Input dataset. Options: ERA5 (default) or ERA5Land.')

    parser.add_argument('--spreads',
                        dest='spreads',
                        default=False,
                        action='store_true',
                        help='Set if spread estimators of decadal TEA indicators should also '
                             'be calculated. Default: False.')

    myopts = parser.parse_args()

    return myopts


def load_data(opts, suppl=False):
    """
    load dec data and select AGR lats
    Args:
        opts: CLI parameter
        suppl: set to true if supplementary data should be loaded

    Returns:
        ds: decadal data of AGR
        lat_lims: latitude limits of AGR
    """

    sdir = ''
    sstr = ''
    biv = ['EF', 'EDavg', 'EMavg', 'EAavg']
    if suppl:
        sdir = 'supplementary/'
        sstr = 'suppl'
        biv = ['delta_y']

    file = (f'{opts.inpath}{sdir}DEC{sstr}_{opts.param_str}_EUR_{opts.period}_{opts.dataset}'
            f'_{opts.start}to{opts.end}.nc')
    ds = xr.open_dataset(file)

    # select relevant vars
    if 'EMavg_Md' in ds.data_vars:
        biv.append('EMavg_Md')
    ds = ds[biv]

    agr_lims = {'EUR': [35, 70], 'S-EUR': [35, 44.5], 'C-EUR': [45, 55], 'N-EUR': [55.5, 70]}

    ds = ds.sel(lat=slice(agr_lims[opts.agr][0], agr_lims[opts.agr][1]))

    return ds, agr_lims[opts.agr]


def add_attrs(opts, nvar, da):
    """
    add attributes to data array
    Args:
        opts: CLI parameter
        nvar: name of variable
        da: data array

    Returns:
        da: data array with attributes

    """

    attrs = {'EF': {'long_name': 'event frequency (AGR)', 'units': '1'},
             'EDavg': {'long_name': 'average event duration (AGR)', 'units': 'dys'},
             'EMavg': {'long_name': 'average event magnitude (AGR)', 'units': opts.unit},
             'EMavg_Md': {'long_name': 'average daily-median exceedance magnitude (AGR)',
                          'units': opts.unit},
             'EAavg': {'long_name': 'average event area (AGR)', 'units': 'areals'},
             'delta_y': {'long_name': 'annual exposure period (AGR)', 'units': 'dys'}}

    da = da.rename(f'{nvar}_AGR')
    da.attrs = attrs[nvar]

    return da


def calc_agr(opts, vdata, awgts):

    # calc mean of ref period (Eq. 26)
    ref_ds = vdata.sel(ctp=slice(PARAMS['REF']['start_cy'], PARAMS['REF']['end_cy']))
    ref_db = (1 / len(ref_ds.ctp)) * (np.log10(ref_ds)).sum(dim='ctp')
    vdata_ref = 10 ** ref_db

    # calc X_Ref^AGR and X_s^AGR (Eq. 34_1 and 34_2)
    x_ref_agr = (awgts * vdata_ref).sum()
    xt_s_agr = (awgts * vdata).sum(dim=('lat', 'lon'))

    # calc Xt_ref_db and Xt_ref_agr (Eq. 34_3)
    # TODO: when data for full ref period available, check if there are any time steps in ref
    #  period with 0 entries that lead to -inf in log
    x_s_agr_ref = xt_s_agr.sel(ctp=slice(PARAMS['REF']['start_cy'], PARAMS['REF']['end_cy']))
    x_s_agr_ref = x_s_agr_ref.where((x_s_agr_ref > 0).compute(), drop=True)
    xt_ref_db = (10/21) * np.log10(x_s_agr_ref).sum()
    xt_ref_agr = 10 ** (xt_ref_db / 10)

    # calculate X_s_AGR (Eq. 34_4)
    x_s_agr = (x_ref_agr / xt_ref_agr) * xt_s_agr

    # add attributes
    x_s_agr = add_attrs(opts=opts, nvar=vdata.name, da=x_s_agr)

    return x_ref_agr, x_s_agr


def add_compound_attrs(opts, da):
    """
    add attributes to da of compound metrics
    Args:
        opts: CLI parameter
        da: da of variable

    Returns:
        da: da with attributes

    """
    attrs = {'EM_AGR': {'long_name': 'cumulative event magnitude (AGR)', 'units': opts.unit},
             'ESavg_AGR': {'long_name': 'average event severity (AGR)',
                           'units': f'areal {opts.unit} dys'},
             'TEX_AGR': {'long_name': 'total events extremity (AGR)',
                         'units': f'areal {opts.unit} dys'},
             'EM_Md_AGR': {'long_name': 'cumulative daily-median exceedance magnitude (AGR)',
                          'units': opts.unit},
             'H_AEHC_AGR': {'long_name': 'cumulative atmospheric boundary layer exceedance '
                                         'heat content (AGR)', 'units': 'PJ'}}

    da.attrs = attrs[da.name]

    return da

def calc_compound_vars(opts, agr, suppl, refs):
    """
    calc compound AGR variables (Eq. 35)
    Args:
        opts: CLI parameter
        agr: AGR vars
        suppl: supplementary AGR vars
        refs: AGR ref vals (from Eq. 34_1)

    Returns:

    """

    # approximate atmospheric boundary layer daily exceedance heat energy uptake capacity
    # [PJ/(areal °C day)]
    ct_abl = 0.1507

    # REF variables
    m_ref_agr = refs['EF_AGR'] * refs['EDavg_AGR'] * refs['EMavg_AGR']
    savg_ref_agr = refs['EDavg_AGR'] * refs['EMavg_AGR'] * refs['EAavg_AGR']
    tex_ref_agr = refs['EF_AGR'] * savg_ref_agr
    mmd_ref_agr = refs['EF_AGR'] * refs['EDavg_AGR'] * refs['EMavg_Md_AGR']
    haehc_ref_agr = ct_abl * tex_ref_agr

    # Time series
    m_s_agr = agr['EF_AGR'] * agr['EDavg_AGR'] * agr['EMavg_AGR']
    savg_s_agr = agr['EDavg_AGR'] * agr['EMavg_AGR'] * agr['EAavg_AGR']
    tex_s_agr = agr['EF_AGR'] * savg_s_agr
    haehc_s_agr = ct_abl * tex_s_agr
    if 'EMavg_Md_GR' in agr:
        mmd_s_agr = agr['EF_AGR'] * agr['EDavg_AGR'] * agr['EMavg_Md_AGR']
    else:
        mmd_s_agr = agr['EF_AGR'] * agr['EDavg_AGR'] * suppl['EMavg_Md_AGR']

    # add vars to ds
    refs['EM_AGR'] = m_ref_agr
    refs['ESavg_AGR'] = savg_ref_agr
    refs['TEX_AGR'] = tex_ref_agr
    refs['EM_Md_AGR'] = mmd_ref_agr
    refs['H_AEHC_AGR'] = haehc_ref_agr

    suppl['EM_AGR'] = m_s_agr
    agr['ESavg_AGR'] = savg_s_agr
    agr['TEX_AGR'] = tex_s_agr
    suppl['EM_Md_AGR'] = mmd_s_agr
    suppl['H_AEHC_AGR'] = haehc_s_agr

    for vvar in ['ESavg_AGR', 'TEX_AGR']:
        agr[vvar] = add_compound_attrs(opts=opts, da=agr[vvar])

    for vvar in ['EM_AGR', 'EM_Md_AGR', 'H_AEHC_AGR']:
        agr[vvar] = add_compound_attrs(opts=opts, da=suppl[vvar])

    return refs, agr, suppl


def run():
    opts = getopts()

    # add necessary strings to opts
    opts = extend_tea_opts(opts)

    # load area grid (0.5°)
    areas = xr.open_dataarray(f'{opts.statpath}area_grid_0p5_EUR_{opts.dataset}.nc')

    # load TEA data
    data, lat_lims = load_data(opts=opts)
    data_suppl, _ = load_data(opts=opts, suppl=True)

    # slice area grid
    areas = areas.sel(lat=slice(lat_lims[0], lat_lims[1]))

    # calc area weights
    wgts = areas / areas.sum()

    # calc AGR vars
    refs = xr.Dataset()
    agrs = xr.Dataset(coords=data.coords)
    for vvar in data.data_vars:
        ref_agr, var_agr = calc_agr(opts=opts, vdata=data[vvar], awgts=wgts)
        agrs[f'{vvar}_AGR'] = var_agr
        refs[f'{vvar}_AGR'] = ref_agr

    # calc AGR supplementary vars
    agrs_suppl = xr.Dataset(coords=data.coords)
    for vvar in data_suppl.data_vars:
        ref_agr, var_agr = calc_agr(opts=opts, vdata=data_suppl[vvar], awgts=wgts)
        agrs_suppl[f'{vvar}_AGR'] = var_agr
        refs[f'{vvar}_AGR'] = ref_agr

    # calc compound AGR vars
    refs, agrs, agrs_suppl = calc_compound_vars(opts=opts, agr=agrs, suppl=agrs_suppl, refs=refs)

    print()


if __name__ == '__main__':
    run()
