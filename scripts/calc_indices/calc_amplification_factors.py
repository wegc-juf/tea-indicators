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
from general_functions import create_history, ref_cc_params, extend_tea_opts

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

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
                        default='/data/users/hst/TEA-clean/TEA/dec_indicator_variables/',
                        type=dir_path,
                        help='Path of folder where TEA data is located.')

    parser.add_argument('--outpath',
                        default='/data/users/hst/TEA-clean/TEA/',
                        help='Path of folder where output data should be saved.')

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

    myopts = parser.parse_args()

    return myopts


def load_data(opts):
    """
    load decadal-mean data
    Args:
        opts: CLI parameter

    Returns:

    """

    file = (f'{opts.inpath}'
            f'DEC_{opts.param_str}_{opts.region}_{opts.dataset}_{opts.start}to{opts.end}.nc')
    data = xr.open_dataset(file)

    return data


def calc_ref_cc_mean(data):
    """
    calc mean of Reference and Current Climate period (Eq. 26)
    Args:
        data: dec TEA ds

    Returns:
        ref: mean values of reference period
        cc: mean values of current climate period
    """

    ref_ds = data.sel(ctp=slice(PARAMS['REF']['start_cy'], PARAMS['REF']['end_cy']))
    cc_ds = data.sel(ctp=slice(PARAMS['CC']['start_cy'], PARAMS['CC']['end_cy']))

    ref_db = (1 / len(ref_ds.ctp)) * (np.log10(ref_ds)).sum(dim='ctp')
    cc_db = (1 / len(cc_ds.ctp)) * (np.log10(cc_ds)).sum(dim='ctp')

    ref = 10 ** ref_db
    cc = 10 ** cc_db

    for vvar in ref.data_vars:
        if 'long_name' in ref[vvar].attrs:
            ref[vvar].attrs['long_name'] += ' REF mean'
            ref[vvar].attrs['long_name'] += ' CC mean'

    return ref, cc


def calc_basis_amplification_factors(data, ref, cc):
    """
    calcualte amplification factors of basis variables (Eq. 28 & 29)
    Args:
        data: decadal-mean data
        ref: mean of REF
        cc: mean of CC

    Returns:
        af_ds: AF time series ds
        af_cc_ds: AF CC ds

    """

    af_ds = data / ref
    af_cc_ds = cc / ref

    # update attributes
    for vvar in af_ds.data_vars:
        af_ds[vvar].attrs = data[vvar].attrs
        af_cc_ds[vvar].attrs = data[vvar].attrs
        if 'long_name' in af_ds[vvar].attrs:
            af_ds[vvar].attrs['long_name'] += ' amplification'
            af_cc_ds[vvar].attrs['long_name'] += ' CC amplification'
        if 'units' in af_ds[vvar].attrs:
            af_ds[vvar].attrs['units'] = '1'
            af_cc_ds[vvar].attrs['units'] = '1'

    # rename vars
    rename_dict_af_ds = {vvar: f'{vvar}_AF' for vvar in af_ds.data_vars}
    rename_dict_af_cc_ds = {vvar: f'{vvar}_AF_CC' for vvar in af_cc_ds.data_vars}
    af_ds = af_ds.rename(rename_dict_af_ds)
    af_cc_ds = af_cc_ds.rename(rename_dict_af_cc_ds)

    return af_ds, af_cc_ds


def calc_compound_amplification_factors(opts, af, af_cc):
    """
    calculate amplification factors of compound variables (Eq. 30)
    Args:
        opts: CLI parameter
        af: amplification factor time series
        af_cc: CC amplification factors

    Returns:
        af: amplification factor time series with compound AF added
        af_cc: CC amplification factors with compound AF added

    """

    em_var = 'EMavg_GR_AF'
    if opts.parameter == 'P':
        em_var = 'EMavg_Md_GR_AF'

    # tEX
    af_tEX = af['EF_GR_AF'] * af['EDavg_GR_AF'] * af[em_var]
    af_tEX = af_tEX.rename('tEX_GR_AF')
    af_tEX = af_tEX.assign_attrs(
        {'long_name': 'decadal-mean temporal events extremity (GR) amplification', 'units': '1'})
    af_cc_tEX = af_cc['EF_GR_AF_CC'] * af_cc['EDavg_GR_AF_CC'] * af_cc[f'{em_var}_CC']
    af_cc_tEX = af_cc_tEX.rename('tEX_GR_AF_CC')
    af_cc_tEX = af_cc_tEX.assign_attrs(
        {'long_name': 'decadal-mean temporal events extremity (GR) CC amplification', 'units': '1'})

    # ES
    af_es = af['EDavg_GR_AF'] * af[em_var] * af['EAavg_GR_AF']
    af_es = af_es.rename('ES_GR_AF')
    af_es = af_es.assign_attrs(
        {'long_name': 'decadal-mean event severity (GR) amplification', 'units': '1'})
    af_cc_es = af_cc['EDavg_GR_AF_CC'] * af_cc[f'{em_var}_CC'] * af_cc['EAavg_GR_AF_CC']
    af_cc_es = af_cc_es.rename('ES_GR_AF_CC')
    af_cc_es = af_cc_es.assign_attrs(
        {'long_name': 'decadal-mean event severity (GR) CC amplification', 'units': '1'})

    # TEX
    af_TEX = af['EF_GR_AF'] * af_es
    af_TEX = af_TEX.rename('TEX_GR_AF')
    af_TEX = af_TEX.assign_attrs(
        {'long_name': 'decadal-mean total events extremity (GR) amplification', 'units': '1'})
    af_cc_TEX = af_cc['EF_GR_AF_CC'] * af_cc_es
    af_cc_TEX = af_cc_TEX.rename('TEX_GR_AF_CC')
    af_cc_TEX = af_cc_TEX.assign_attrs(
        {'long_name': 'decadal-mean total events extremity (GR) CC amplification', 'units': '1'})

    af = xr.merge([af, af_tEX, af_es, af_TEX])
    af_cc = xr.merge([af_cc, af_cc_tEX, af_cc_es, af_cc_TEX])

    return af, af_cc

def save_output(opts, af, af_cc):
    """
    save amplification data to nc files
    Args:
        opts: CLI parameter
        af: amplification factors ds
        af_cc: CC amplification factors ds

    Returns:

    """

    ds_out = xr.merge([af, af_cc])

    # apply masks to grid data again (sum etc. result in 0 outside of region)
    masks = xr.open_dataset(f'{opts.maskpath}{opts.region}_masks_{opts.dataset}.nc')
    mask = masks['lt1500_mask'] * masks['mask']
    for vvar in ds_out.data_vars:
        if 'GR' not in vvar:
            ds_out[vvar] = ds_out[vvar].where(mask == 1)

    ds_out = create_history(cli_params=sys.argv, ds=ds_out)
    path = Path(f'{opts.outpath}amplification/')
    path.mkdir(parents=True, exist_ok=True)
    ds_out.to_netcdf(f'{opts.outpath}amplification/'
                     f'AF_{opts.param_str}_{opts.region}_{opts.period}_{opts.dataset}'
                     f'_{opts.start}to{opts.end}.nc')

def run():
    opts = getopts()
    opts = extend_tea_opts(opts)

    # load DEC TEA data
    ds = load_data(opts=opts)

    # calc mean of REF and CC periods
    ref_avg, cc_avg = calc_ref_cc_mean(data=ds)

    # calc amplification factors of basis variables
    bvars = [vvar for vvar in ds.data_vars if vvar not in ['TEX_GR', 'ESavg_GR']]
    af, af_cc = calc_basis_amplification_factors(data=ds[bvars], ref=ref_avg[bvars],
                                                 cc=cc_avg[bvars])

    # calc amplification factors of compound variables
    af, af_cc = calc_compound_amplification_factors(opts=opts, af=af, af_cc=af_cc)

    # save output
    save_output(opts=opts, af=af, af_cc=af_cc)


if __name__ == '__main__':
    run()
