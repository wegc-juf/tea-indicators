#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: hst
"""

import logging
import numpy as np
import os
from pathlib import Path
import sys
import xarray as xr

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from scripts.general_stuff.general_functions import (create_history_from_cfg, ref_cc_params,
                                                     load_opts)
from scripts.general_stuff.var_attrs import get_attrs

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

PARAMS = ref_cc_params()

def load_data(opts):
    """
    load decadal-mean data
    Args:
        opts: CLI parameter

    Returns:

    """

    file = (f'{opts.inpath}'
            f'DEC_{opts.param_str}_{opts.region}_{opts.period}_{opts.dataset}'
            f'_{opts.start}to{opts.end}.nc')
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

    ref_ds = data.sel(time=slice(PARAMS['REF']['start_cy'], PARAMS['REF']['end_cy']))
    cc_ds = data.sel(time=slice(PARAMS['CC']['start_cy'], PARAMS['CC']['end_cy']))

    ref_db = (1 / len(ref_ds.time)) * (np.log10(ref_ds)).sum(dim='time')
    cc_db = (1 / len(cc_ds.time)) * (np.log10(cc_ds)).sum(dim='time')

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
    # only use data with positive values in reference period to avoid inf-values later on
    ref = ref.where(ref > 0)

    af_ds = data / ref
    af_cc_ds = cc / ref

    # update attributes
    for vvar in af_ds.data_vars:
        af_ds[vvar].attrs = data[vvar].attrs
        af_cc_ds[vvar].attrs = data[vvar].attrs
        af_ds[vvar].attrs = get_attrs(vname=f'{vvar}_AF')
        af_cc_ds[vvar].attrs = get_attrs(vname=f'{vvar}_AF_CC')

    # rename vars
    rename_dict_af_ds = {vvar: f'{vvar}_AF' for vvar in af_ds.data_vars}
    rename_dict_af_cc_ds = {vvar: f'{vvar}_AF_CC' for vvar in af_cc_ds.data_vars}
    af_ds = af_ds.rename(rename_dict_af_ds)
    af_cc_ds = af_cc_ds.rename(rename_dict_af_cc_ds)

    return af_ds, af_cc_ds


def calc_compound_amplification_factors(opts, af, af_cc, dm=False):
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

    gr_str = '_GR'
    if 'agr' in opts:
        gr_str = ''

    em_var = f'EM_avg{gr_str}_AF'
    if opts.precip:
        em_var = f'EM_avg_Md{gr_str}_AF'

    # tEX
    af_tEX = af[f'EF{gr_str}_AF'] * af[f'ED_avg{gr_str}_AF'] * af[em_var]
    af_tEX = af_tEX.rename(f'tEX{gr_str}_AF')
    af_tEX = af_tEX.assign_attrs(get_attrs(vname=f'tEX{gr_str}_AF'))
    af_cc_tEX = af_cc[f'EF{gr_str}_AF_CC'] * af_cc[f'ED_avg{gr_str}_AF_CC'] * af_cc[f'{em_var}_CC']
    af_cc_tEX = af_cc_tEX.rename(f'tEX{gr_str}_AF_CC')
    af_cc_tEX = af_cc_tEX.assign_attrs(get_attrs(vname=f'tEX{gr_str}_AF_CC'))

    # ES
    af_es = af[f'ED_avg{gr_str}_AF'] * af[em_var] * af[f'EA_avg{gr_str}_AF']
    af_es = af_es.rename(f'ES_avg{gr_str}_AF')
    af_es = af_es.assign_attrs(get_attrs(vname=f'ES_avg{gr_str}_AF'))
    af_cc_es = af_cc[f'ED_avg{gr_str}_AF_CC'] * af_cc[f'{em_var}_CC'] * af_cc[f'EA_avg{gr_str}_AF_CC']
    af_cc_es = af_cc_es.rename(f'ES_avg{gr_str}_AF_CC')
    af_cc_es = af_cc_es.assign_attrs(get_attrs(vname=f'ES_avg{gr_str}_AF_CC'))

    # TEX
    af_TEX = af[f'EF{gr_str}_AF'] * af_es
    af_TEX = af_TEX.rename(f'TEX{gr_str}_AF')
    af_TEX = af_TEX.assign_attrs(get_attrs(vname=f'TEX{gr_str}_AF'))
    af_cc_TEX = af_cc[f'EF{gr_str}_AF_CC'] * af_cc_es
    af_cc_TEX = af_cc_TEX.rename(f'TEX{gr_str}_AF_CC')
    af_cc_TEX = af_cc_TEX.assign_attrs(get_attrs(vname=f'TEX{gr_str}_AF_CC'))

    af_vars = [af, af_tEX, af_es, af_TEX]
    af_cc_vars = [af_cc, af_cc_tEX, af_cc_es, af_cc_TEX]

    if dm:
        af_dm = af[f'ED_avg{gr_str}_AF'] * af[em_var]
        af_dm = af_dm.rename(f'DM{gr_str}_AF')
        af_dm = af_dm.assign_attrs(get_attrs(vname=f'DM{gr_str}_AF'))
        af_cc_dm = af_cc[f'ED_avg{gr_str}_AF_CC'] * af_cc[f'{em_var}_CC']
        af_cc_dm = af_cc_dm.rename(f'DM{gr_str}_AF_CC')
        af_cc_dm = af_cc_dm.assign_attrs(get_attrs(vname=f'DM{gr_str}_AF_CC'))
        af_vars = [af, af_tEX, af_es, af_TEX, af_dm]
        af_cc_vars = [af_cc, af_cc_tEX, af_cc_es, af_cc_TEX, af_cc_dm]

    af = xr.merge(af_vars)
    af_cc = xr.merge(af_cc_vars)

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

    ds_out = create_history_from_cfg(cfg_params=opts, ds=ds_out)
    # ds_out = create_history(cli_params=sys.argv, ds=ds_out)
    path = Path(f'{opts.outpath}amplification/')
    path.mkdir(parents=True, exist_ok=True)
    ds_out.to_netcdf(f'{opts.outpath}amplification/'
                     f'AF_{opts.param_str}_{opts.region}_{opts.period}_{opts.dataset}'
                     f'_{opts.start}to{opts.end}.nc')

def run():
    # load CFG parameter
    opts = load_opts(fname=__file__)

    # load DEC TEA data
    ds = load_data(opts=opts)

    # calc mean of REF and CC periods
    ref_avg, cc_avg = calc_ref_cc_mean(data=ds)

    # calc amplification factors of basis variables
    bvars = [vvar for vvar in ds.data_vars if vvar not in ['TEX_GR', 'ES_avg_GR']]
    af, af_cc = calc_basis_amplification_factors(data=ds[bvars], ref=ref_avg[bvars],
                                                 cc=cc_avg[bvars])

    # calc amplification factors of compound variables
    af, af_cc = calc_compound_amplification_factors(opts=opts, af=af, af_cc=af_cc)

    # save output
    save_output(opts=opts, af=af, af_cc=af_cc)


if __name__ == '__main__':
    run()
