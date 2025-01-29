#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: hst
"""

import copy
import numpy as np
import os
import sys
import xarray as xr

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from scripts.general_stuff.general_functions import (create_history_from_cfg, ref_cc_params,
                                                     load_opts)
from scripts.general_stuff.var_attrs import get_attrs
from scripts.calc_indices.calc_amplification_factors import (calc_ref_cc_mean,
                                                             calc_basis_amplification_factors,
                                                             calc_compound_amplification_factors)

DS_PARAMS = {'SPARTACUS': {'xname': 'x', 'yname': 'y'},
             'ERA5': {'xname': 'lon', 'yname': 'lat'},
             'ERA5Land': {'xname': 'lon', 'yname': 'lat'}}

PARAMS = ref_cc_params()


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
    if suppl:
        sdir = 'supplementary/'
        sstr = 'suppl'

    if 'EUR' in opts.agr:
        ag_reg = 'EUR'
    else:
        ag_reg = opts.agr

    file = (f'{opts.inpath}{sdir}DEC{sstr}_{opts.param_str}_{ag_reg}_{opts.period}_{opts.dataset}'
            f'_{opts.start}to{opts.end}.nc')
    ds = xr.open_dataset(file)

    agr_lims = {'EUR': [35, 70], 'S-EUR': [35, 44.5], 'C-EUR': [45, 55], 'N-EUR': [55.5, 70]}
    if opts.agr not in agr_lims.keys():
        lat_min = ds.lat.min().values
        lat_max = ds.lat.max().values
    else:
        lat_min = agr_lims[opts.agr][0]
        lat_max = agr_lims[opts.agr][1]
    ds = ds.sel(lat=slice(lat_max, lat_min))

    lims = [lat_max, lat_min]

    return ds, lims


def calc_grid_afacs(opts, data):
    # adjust opts for calc_amplification_factors.py functions
    af_opts = copy.deepcopy(opts)
    af_opts.maskpath = '/data/arsclisys/normal/clim-hydro/TEA-Indicators/masks/'

    # calc mean of REF and CC periods
    ref_avg, cc_avg = calc_ref_cc_mean(data=data)

    # calc amplification factors of basis variables
    bvars = [vvar for vvar in data.data_vars if vvar not in ['TEX', 'ES_avg']]
    af, af_cc = calc_basis_amplification_factors(data=data[bvars], ref=ref_avg[bvars],
                                                 cc=cc_avg[bvars])

    # calc amplification factors of compound variables
    af, af_cc = calc_compound_amplification_factors(opts=af_opts, af=af, af_cc=af_cc)

    # combine ds into one
    af_grid = xr.merge([af, af_cc])

    # apply masks to grid data again (sum etc. result in 0 outside of region)
    mask = xr.open_dataarray(f'{af_opts.maskpath}{af_opts.region}_mask_0p5_{af_opts.dataset}.nc')
    for vvar in af_grid.data_vars:
        af_grid[vvar] = af_grid[vvar].where(mask == 1)

    return af_grid


def calc_agr(opts, vdata, awgts):
    """
    calculate AGR variables
    Args:
        opts: CLI parameter
        vdata: data of variable
        awgts: area weights

    Returns:
        x_ref_agr: Ref value of AGR
        x_s_agr: time series of AGR

    """

    # calc mean of ref period (Eq. 26)
    ref_ds = vdata.sel(ctp=slice(PARAMS['REF']['start_cy'], PARAMS['REF']['end_cy']))
    ref_db = (1 / len(ref_ds.ctp)) * (np.log10(ref_ds)).sum(dim='ctp')
    vdata_ref = 10 ** ref_db

    # calc X_Ref^AGR and X_s^AGR (Eq. 34_1 and 34_2)
    x_ref_agr = (awgts * vdata_ref).sum()
    xt_s_agr = (awgts * vdata).sum(dim=('lat', 'lon'))

    # calc Xt_ref_db and Xt_ref_agr (Eq. 34_3)
    x_s_agr_ref = xt_s_agr.sel(ctp=slice(PARAMS['REF']['start_cy'], PARAMS['REF']['end_cy']))
    x_s_agr_ref = x_s_agr_ref.where((x_s_agr_ref > 0).compute(), drop=True)
    xt_ref_db = (10 / 21) * np.log10(x_s_agr_ref).sum()
    xt_ref_agr = 10 ** (xt_ref_db / 10)

    if len(x_s_agr_ref) < 21:
        raise Warning('There are 0-values in the ref period that lead to -inf in the logarithm!')

    # calculate X_s_AGR (Eq. 34_4)
    x_s_agr = (x_ref_agr / xt_ref_agr) * xt_s_agr

    # add attributes
    x_s_agr = x_s_agr.rename(f'{vdata.name}_AGR')
    x_s_agr.attrs = get_attrs(opts=opts, vname=f'{vdata.name}_AGR')

    return x_ref_agr, x_s_agr


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
        agr[vvar].attrs = get_attrs(opts=opts, vname=vvar)

    for vvar in ['EM_AGR', 'EM_Md_AGR', 'H_AEHC_AGR']:
        suppl[vvar].attrs = get_attrs(opts=opts, vname=vvar)

    return refs, agr, suppl


def calc_cc_mean(da):
    """
    calculate mean of recent climate period (Eq. 36)
    Args:
        da: AGR data array

    Returns:

    """
    cc_da = da.sel(ctp=slice(PARAMS['CC']['start_cy'], PARAMS['CC']['end_cy']))
    cc_db = (1 / len(cc_da.ctp)) * (np.log10(cc_da)).sum(dim='ctp')
    da_cc = 10 ** cc_db

    return da_cc


def calc_ampl_facs(ref, cc, data, data_suppl):
    """
    Calculate amplification factors (Eq. 37)
    Args:
        ref: AGR REF vals
        cc: AGR CC vals
        data: AGR time series
        data_suppl: AGR suppl time series

    Returns:
        af: ds with amplification factor time series and A_CC

    """

    ampl_facs = xr.Dataset()
    ampl_facs_suppl = xr.Dataset()

    for vvar in data.data_vars:
        a_cc = cc[vvar] / ref[vvar]
        a_s = data[vvar] / ref[vvar]
        ampl_facs[f'{vvar}_AF'] = a_s
        ampl_facs[f'{vvar}_AF'].attrs = get_attrs(vname=f'{vvar}_AF')
        ampl_facs[f'{vvar}_AF_CC'] = a_cc
        ampl_facs[f'{vvar}_AF_CC'].attrs = get_attrs(vname=f'{vvar}_AF_CC')

    for vvar in data_suppl.data_vars:
        a_cc = cc[vvar] / ref[vvar]
        a_s = data_suppl[vvar] / ref[vvar]
        ampl_facs_suppl[f'{vvar}_AF'] = a_s
        ampl_facs_suppl[f'{vvar}_AF'].attrs = get_attrs(vname=f'{vvar}_AF')
        ampl_facs_suppl[f'{vvar}_AF_CC'] = a_cc
        ampl_facs_suppl[f'{vvar}_AF_CC'].attrs = get_attrs(vname=f'{vvar}_AF_CC')

    return ampl_facs, ampl_facs_suppl


def calc_compound_amplification_factors_grid(opts, af, af_cc):
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

    em_var = 'EMavg_AF'
    if opts.precip:
        em_var = 'EMavg_Md_AF'

    # tEX
    af_tEX = af['EF_AF'] * af['EDavg_AF'] * af[em_var]
    af_tEX = af_tEX.rename('tEX_AF')
    af_tEX = af_tEX.assign_attrs(get_attrs(opts=opts, vname='tEX_AF'))
    af_cc_tEX = af_cc['EF_AF_CC'] * af_cc['EDavg_AF_CC'] * af_cc[f'{em_var}_CC']
    af_cc_tEX = af_cc_tEX.rename('tEX_AF_CC')
    af_cc_tEX = af_cc_tEX.assign_attrs(get_attrs(opts=opts, vname='tEX_AF_CC'))

    # ESavg
    af_es = af['EDavg_AF'] * af[em_var] * af['EAavg_AF']
    af_es = af_es.rename('ESavg_AF')
    af_es = af_es.assign_attrs(get_attrs(opts=opts, vname='ESavg_AF'))
    af_cc_es = af_cc['EDavg_AF_CC'] * af_cc[f'{em_var}_CC'] * af_cc['EAavg_AF_CC']
    af_cc_es = af_cc_es.rename('ESavg_AF_CC')
    af_cc_es = af_cc_es.assign_attrs(get_attrs(opts=opts, vname='ESavg_AF_CC'))

    # TEX
    af_TEX = af['EF_AF'] * af_es
    af_TEX = af_TEX.rename('TEX_AF')
    af_TEX = af_TEX.assign_attrs(get_attrs(opts=opts, vname='TEX_AF'))
    af_cc_TEX = af_cc['EF_AF_CC'] * af_cc_es
    af_cc_TEX = af_cc_TEX.rename('TEX_AF_CC')
    af_cc_TEX = af_cc_TEX.assign_attrs(get_attrs(opts=opts, vname='TEX_AF_CC'))

    af = xr.merge([af, af_tEX, af_es, af_TEX])
    af_cc = xr.merge([af_cc, af_cc_tEX, af_cc_es, af_cc_TEX])

    return af, af_cc


def calc_ampl_facs_grid(opts, data, suppl=False):
    ref_avg, cc_avg = calc_ref_cc_mean(data=data)

    bvars = [vvar for vvar in data.data_vars if vvar not in ['TEX', 'ESavg']]
    af, af_cc = calc_basis_amplification_factors(data=data[bvars], ref=ref_avg[bvars],
                                                 cc=cc_avg[bvars])
    if not suppl:
        af, af_cc = calc_compound_amplification_factors_grid(opts=opts, af=af, af_cc=af_cc)

    return af


def save_output(opts, data):
    outpaths = {'AGR': f'{opts.outpath}dec_indicator_variables/'
                       f'DEC_{opts.param_str}_AGR-{opts.agr}_{opts.period}_{opts.dataset}'
                       f'_{opts.start}to{opts.end}.nc',
                'AGRsuppl': f'{opts.outpath}dec_indicator_variables/supplementary/'
                            f'DECsuppl_{opts.param_str}_AGR-{opts.agr}_{opts.period}_{opts.dataset}'
                            f'_{opts.start}to{opts.end}.nc',
                'AF': f'{opts.outpath}amplification/'
                      f'AF_{opts.param_str}_AGR-{opts.agr}_{opts.period}_{opts.dataset}'
                      f'_{opts.start}to{opts.end}.nc',
                'AF_us': f'{opts.outpath}amplification/'
                         f'AF_sUPP_{opts.param_str}_AGR-{opts.agr}_{opts.period}_{opts.dataset}'
                         f'_{opts.start}to{opts.end}.nc',
                'AF_ls': f'{opts.outpath}amplification/'
                         f'AF_sLOW_{opts.param_str}_AGR-{opts.agr}_{opts.period}_{opts.dataset}'
                         f'_{opts.start}to{opts.end}.nc',
                'AF_suppl': f'{opts.outpath}amplification/supplementary/'
                            f'AFsuppl_{opts.param_str}_AGR-{opts.agr}_{opts.period}_{opts.dataset}'
                            f'_{opts.start}to{opts.end}.nc',
                'AF_suppl_us': f'{opts.outpath}amplification/supplementary/'
                               f'AFsuppl_sUPP_{opts.param_str}_AGR-{opts.agr}_{opts.period}'
                               f'_{opts.dataset}_{opts.start}to{opts.end}.nc',
                'AF_suppl_ls': f'{opts.outpath}amplification/supplementary/'
                               f'AFsuppl_sLOW_{opts.param_str}_AGR-{opts.agr}_{opts.period}'
                               f'_{opts.dataset}_{opts.start}to{opts.end}.nc',
                'AGR_us': f'{opts.outpath}dec_indicator_variables/'
                          f'DEC_sUPP_{opts.param_str}_AGR-{opts.agr}_{opts.period}'
                          f'_{opts.dataset}_{opts.start}to{opts.end}.nc',
                'AGR_ls': f'{opts.outpath}dec_indicator_variables/'
                          f'DEC_sLOW_{opts.param_str}_AGR-{opts.agr}_{opts.period}'
                          f'_{opts.dataset}_{opts.start}to{opts.end}.nc',
                'AGRsuppl_us': f'{opts.outpath}dec_indicator_variables/supplementary/'
                               f'DECsupp_sUPP_{opts.param_str}_AGR-{opts.agr}_{opts.period}'
                               f'_{opts.dataset}_{opts.start}to{opts.end}.nc',
                'AGRsuppl_ls': f'{opts.outpath}dec_indicator_variables/supplementary/'
                               f'DECsupp_sLOW_{opts.param_str}_AGR-{opts.agr}_{opts.period}'
                               f'_{opts.dataset}_{opts.start}to{opts.end}.nc'}

    for vvars in data.keys():
        ds = data[vvars]
        ds = create_history_from_cfg(cfg_params=opts, ds=ds)
        ds.to_netcdf(outpaths[vvars])


def calc_spread_estimates(gdata, data, areas, afacs=False):
    """
    calculate spread estimates of AGR variables and AFs (Eq. 38)
    Args:
        gdata: grid data
        data: ds
        areas: area grid
        afacs: set if AF are passed

    Returns:
        s_upp: upper spread ds
        s_low: lower spread ds
    """

    su = xr.Dataset(coords={'ctp': (['ctp'], data.ctp.values)})
    sl = xr.Dataset(coords={'ctp': (['ctp'], data.ctp.values)})

    for vvar in gdata.data_vars:
        data_var = f'{vvar}_AGR'
        if afacs:
            data_var = vvar.split('_AGR_AF')[0] + '_AF'

        if data_var == 'EM_AF':
            continue

        if not afacs:
            if f'{vvar}_AGR' not in data.data_vars:
                continue
        c_upp = xr.full_like(gdata[vvar], 1)
        c_upp = c_upp.where(gdata[vvar] >= data[data_var], 0)

        # calc upper spread (Eq. 38_2, 38_5)
        wgt_fac_u = 1 / (c_upp * areas).sum(dim=('lat', 'lon'))
        sum_term_u = ((c_upp * areas) * (gdata[vvar] - data[data_var]) ** 2).sum(dim=('lat', 'lon'))
        s_upp = np.sqrt(wgt_fac_u * sum_term_u)
        s_upp = s_upp.rename(f'{data_var}_supp')

        # calc lower spread (Eq. 38_3, 38_6)
        wgt_fac_l = 1 / ((1 - c_upp) * areas).sum(dim=('lat', 'lon'))
        sum_term_l = (((1 - c_upp) * areas) * (gdata[vvar] - data[data_var]) ** 2).sum(
            dim=('lat', 'lon'))
        s_low = np.sqrt(wgt_fac_l * sum_term_l)
        s_low = s_low.rename(f'{data_var}_slow')

        # save to ds
        su[f'{s_upp.name}'] = s_upp
        sl[f'{s_low.name}'] = s_low

        su[f'{s_upp.name}'].attrs = get_attrs(vname=vvar, spread='upper')
        sl[f'{s_low.name}'].attrs = get_attrs(vname=vvar, spread='lower')

    return su, sl


def run():
    # load CFG parameter
    opts = load_opts(fname=__file__)

    # load area grid (0.5°)
    if 'EUR' in opts.agr:
        ag_reg = 'EUR'
    else:
        ag_reg = opts.agr
    areas = xr.open_dataarray(f'{opts.statpath}area_grid_0p5_{ag_reg}_{opts.dataset}.nc')

    # load TEA data
    data, lat_lims = load_data(opts=opts)
    data_suppl, _ = load_data(opts=opts, suppl=True)

    # calc AF on grid
    af_grid = calc_grid_afacs(opts=opts, data=data)

    # slice area grid
    areas = areas.sel(lat=slice(lat_lims[0], lat_lims[1]),
                      lon=slice(data.lon[0].values, data.lon[-1].values))

    # calc area weights (w_l)
    wgts = areas / areas.sum()

    # calc AGR vars
    refs = xr.Dataset()
    agrs = xr.Dataset(coords=data.coords)

    # define basis vars for which AGR vars should be calculated
    biv = ['EF', 'EDavg', 'EMavg', 'EAavg']
    biv_suppl = ['delta_y']
    if 'EMavg_Md' in data.data_vars:
        biv.append('EMavg_Md')
    else:
        biv_suppl.append('EMavg_Md')

    for vvar in biv:
        ref_agr, var_agr = calc_agr(opts=opts, vdata=data[vvar], awgts=wgts)
        agrs[f'{vvar}_AGR'] = var_agr
        refs[f'{vvar}_AGR'] = ref_agr

    # calc AGR supplementary vars
    agrs_suppl = xr.Dataset(coords=data.coords)
    for vvar in biv_suppl:
        ref_agr, var_agr = calc_agr(opts=opts, vdata=data_suppl[vvar], awgts=wgts)
        agrs_suppl[f'{vvar}_AGR'] = var_agr
        refs[f'{vvar}_AGR'] = ref_agr

    # calc compound AGR vars
    refs, agrs, agrs_suppl = calc_compound_vars(opts=opts, agr=agrs, suppl=agrs_suppl, refs=refs)

    # calc CC mean
    ccs = xr.Dataset()
    for vvar in agrs.data_vars:
        cc = calc_cc_mean(da=agrs[vvar])
        ccs[vvar] = cc

    for vvar in agrs_suppl.data_vars:
        cc = calc_cc_mean(da=agrs_suppl[vvar])
        ccs[vvar] = cc

    # calc amplification factors
    af, af_suppl = calc_ampl_facs(ref=refs, cc=ccs, data=agrs, data_suppl=agrs_suppl)

    # calc AF_sl^X,GR
    data['H_AEHC'] = 0.1507 * data['TEX']
    af_sl = calc_ampl_facs_grid(opts=opts, data=data, suppl=False)
    af_sl_suppl = calc_ampl_facs_grid(opts=opts, data=data_suppl, suppl=True)

    # add H_AEHC to af_sl_suppl and remove it from af_sl
    af_sl_suppl['H_AEHC_AF'] = af_sl['H_AEHC_AF']
    af_sl_suppl['H_AEHC_AF'].attrs = get_attrs(vname='H_AEHC_AF')
    af_sl = af_sl.drop_vars('H_AEHC_AF')

    # set 0 values to nan
    agrs = agrs.where(agrs > 0)
    agrs_suppl = agrs_suppl.where(agrs_suppl > 0)
    af = af.where(af > 0)

    # calc spreads
    agrs_us, agrs_ls = calc_spread_estimates(gdata=data, data=agrs, areas=areas)
    agrs_suppl_us, agrs_suppl_ls = calc_spread_estimates(gdata=data_suppl, data=agrs_suppl,
                                                         areas=areas)
    af_us, af_ls = calc_spread_estimates(gdata=af, data=af_sl, areas=areas, afacs=True)
    af_suppl_us, af_suppl_ls = calc_spread_estimates(gdata=af_suppl, data=af_sl_suppl,
                                                     areas=areas, afacs=True)

    af = xr.merge([af, af_grid])

    datasets = {'AGR': agrs, 'AGRsuppl': agrs_suppl,
                'AF': af, 'AF_us': af_us, 'AF_ls': af_ls,
                'AF_suppl': af_suppl, 'AF_suppl_us': af_suppl_us, 'AF_suppl_ls': af_suppl_ls,
                'AGR_us': agrs_us, 'AGR_ls': agrs_ls,
                'AGRsuppl_us': agrs_suppl_us, 'AGRsuppl_ls': agrs_suppl_ls}

    save_output(opts=opts, data=datasets)


if __name__ == '__main__':
    run()
