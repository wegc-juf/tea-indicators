#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: hst

"""
import os
import glob
import numpy as np
import pandas as pd
from pathlib import Path
import sys
import xarray as xr

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from scripts.calc_indices.calc_amplification_factors import (calc_ref_cc_mean,
                                                             calc_basis_amplification_factors,
                                                             calc_compound_amplification_factors)
from scripts.calc_indices.calc_decadal_indicators import rolling_decadal_mean

from scripts.general_stuff.general_functions import ref_cc_params, load_opts

PARAMS = ref_cc_params()


def load_data(opts):
    """
    load TEA station data and apply decadal moving average
    :return: data (station data)
    :return: ampl (amplification factors)
    """

    files = sorted(glob.glob(f'{opts.inpath}station_indices/*{opts.param_str}*.nc'))

    if opts.region == 'AUT':
        stations = ['GRAZ', 'INNS', 'KREM', 'SALZ', 'WIEN']
    else:
        stations = ['GRAZ', 'BADG', 'DEUT']
    ef = pd.DataFrame()
    ed = pd.DataFrame()
    em = pd.DataFrame()
    ampl = pd.DataFrame(index=stations, columns=['EF', 'EDavg', 'EMavg', 'DM', 'tEX'])

    for ifile, file in enumerate(files):
        basename = os.path.basename(file)
        station_abbr = basename.split('_')[-1].split('.nc')[0].upper()[:4]
        # if station_abbr != 'WIEN':
        #     continue
        if station_abbr not in stations:
            continue
        # load data
        data = xr.open_dataset(file)

        # fill nans in ED and EM with 0
        data['EDavg'] = data['EDavg'].where(data['EDavg'].notnull(), 0)
        data['EMavg'] = data['EMavg'].where(data['EMavg'].notnull(), 0)

        # calculate decadal means
        data = rolling_decadal_mean(data=data)

        # calculate reference and cc values
        ref, cc = calc_ref_cc_mean(data=data)

        # calc amplification factors
        ampl_facs, ampl_cc_facs = calc_basis_amplification_factors(data=data, ref=ref, cc=cc)

        for vvar in ampl.columns:
            if vvar in ['tEX', 'DM']:
                continue
            ampl.loc[station_abbr, vvar] = ampl_cc_facs[f'{vvar}_AF_CC'].values
        ampl.loc[station_abbr, 'DM'] = (ampl.loc[station_abbr, 'EDavg']
                                        * ampl.loc[station_abbr, 'EMavg'])
        ampl.loc[station_abbr, 'tEX'] = (ampl.loc[station_abbr, 'EF']
                                         * ampl.loc[station_abbr, 'EDavg']
                                         * ampl.loc[station_abbr, 'EMavg'])

        tef_stat = pd.DataFrame(index=data.ctp.values,
                                data=ampl_facs.EF_AF.values,
                                columns=[station_abbr])
        ed_stat = pd.DataFrame(index=data.ctp.values,
                               data=ampl_facs.EDavg_AF.values,
                               columns=[station_abbr])
        em_stat = pd.DataFrame(index=data.ctp.values,
                               data=ampl_facs.EMavg_AF.values,
                               columns=[station_abbr])

        ef = pd.concat([ef, tef_stat], axis=1)
        ed = pd.concat([ed, ed_stat], axis=1)
        em = pd.concat([em, em_stat], axis=1)

    # calc compound indicators
    dm = ed * em
    tEX = ef * ed * em

    # set values for KREM to nan (before 1920)
    if 'KREM' in stations:
        ef['year'] = ef.index.year
        ef.loc[ef['year'] < 1920, 'KREM'] = np.nan
        ef = ef.drop(['year'], axis=1)

        ed['year'] = ed.index.year
        ed.loc[ed['year'] < 1920, 'KREM'] = np.nan
        ed = ed.drop(['year'], axis=1)

        em['year'] = em.index.year
        em.loc[em['year'] < 1920, 'KREM'] = np.nan
        em = em.drop(['year'], axis=1)

    # combine in dict
    data = {'EF': ef, 'ED': ed, 'EM': em, 'DM': dm, 'tEX': tEX}

    return data, ampl

def rename_juf_data(data):
    """
    rename juf according to hst convention
    Args:
        data: dataset

    Returns:

    """

    dvars = data.data_vars
    for var in dvars:
        if '_avg' in var:
            data = data.rename({var: var.replace('_avg', 'avg')})

    data = data.rename({'EM_GR_Md': 'EM_Md_GR', 'EMavg_GR_Md': 'EMavg_Md_GR'})

    return data


def get_gr_vals(opts):
    """
    load SPARTACUS DEC TEA indicators for GR
    Args:
        opts: CLI parameter

    Returns:

    """

    if not opts.precip:
        em_var = 'EMavg_GR'
    else:
        em_var = 'EMavg_Md_GR'

    ref_data = xr.open_dataset(f'{opts.inpath}dec_indicator_variables/'
                               f'DEC_{opts.param_str}_{opts.region}_WAS_SPARTACUS_1961to2024.nc')

    ref_data = rename_juf_data(data=ref_data)

    vkeep = ['EF_GR', 'EDavg_GR', em_var, 'EAavg_GR']
    vdrop = [vvar for vvar in ref_data.data_vars if vvar not in vkeep]
    ref_data = ref_data.drop_vars(vdrop)

    ref_vals, cc_vals = calc_ref_cc_mean(data=ref_data)
    ampl, cc_ampl = calc_basis_amplification_factors(data=ref_data, ref=ref_vals, cc=cc_vals)
    ampl, cc_ampl = calc_compound_amplification_factors(opts=opts, af=ampl, af_cc=cc_ampl, dm=True)

    # add combined indicator variables
    ref_vals['DM'] = ref_vals['EDavg_GR'] * ref_vals[em_var]
    ref_vals['tEX'] = ref_vals['EF_GR'] * ref_vals['EDavg_GR'] * ref_vals[em_var]
    cc_vals['DM'] = cc_vals['EDavg_GR'] * cc_vals[em_var]
    cc_vals['tEX'] = cc_vals['EF_GR'] * cc_vals['EDavg_GR'] * cc_vals[em_var]
    cc_ampl['DM'] = cc_ampl['EDavg_GR_AF_CC'] * cc_ampl[f'{em_var}_AF_CC']
    cc_ampl['tEX'] = cc_ampl['EF_GR_AF_CC'] * cc_ampl['EDavg_GR_AF_CC'] * cc_ampl[f'{em_var}_AF_CC']

    # calc std of ref period
    try:
        s_ref = ampl.sel(ctp=slice(PARAMS['REF']['start_cy'], PARAMS['REF']['end_cy'])).std()
    except KeyError:
        s_ref = ampl.sel(time=slice(PARAMS['REF']['start_cy'], PARAMS['REF']['end_cy'])).std()

    return ref_vals, cc_vals, ampl, cc_ampl, s_ref


def calc_factors(opts, st_data, s_ref_gr):
    """
    Eq. 32_4 and 32_5 first term
    Args:
        opts: CLI parameter
        st_data: station data
        s_ref_gr: stddevs of GR data in ref period

    Returns:
        factors: ds with factors
    """

    if not opts.precip:
        em_var = 'EMavg'
    else:
        em_var = 'EMavg_Md'

    vnames = {'EF': 'EF_GR_AF', 'ED': 'EDavg_GR_AF', 'EM': f'{em_var}_GR_AF',
              'DM': 'DM_GR_AF', 'tEX': 'tEX_GR_AF'}

    factors = pd.DataFrame(columns=['EF', 'EDavg', em_var, 'DM', 'tEX'])

    for vvar in st_data.keys():
        vdata = st_data[vvar].loc[PARAMS['REF']['start_cy']:PARAMS['REF']['end_cy']]
        s_ref_k = vdata.std()
        fac = s_ref_gr[vnames[vvar]] / np.sqrt((s_ref_k ** 2).sum()/len(s_ref_k.index))
        factors.loc[0, vvar] = fac.values

    return factors


def calc_nat_var(opts, st_data, std_gr):
    """
    calculate natural variability (Eq. 32)
    Args:
        opts: CLI parameter
        st_data: station amplification data
        std_gr: stddevs of GR data in ref period

    Returns:
        std: dataframe with std values
    """

    scaling = calc_factors(opts=opts, st_data=st_data, s_ref_gr=std_gr)

    std = pd.DataFrame(index=st_data.keys(), columns=['lower', 'upper'])

    for vvar in st_data.keys():
        data = st_data[vvar]
        data = data.loc[data.index <= PARAMS['REF']['end_cy']]

        cupp = (data >= 1).astype(int)
        supp = np.sqrt((1/cupp.sum()) * (cupp*(data - 1)**2).sum())
        slow = np.sqrt((1/(1 - cupp).sum()) * ((1 - cupp)*(data - 1)**2).sum())

        supp_nv = np.sqrt((supp ** 2).mean())
        slow_nv = np.sqrt((slow ** 2).mean())

        supp_nv = scaling.loc[0, vvar] * supp_nv
        slow_nv = scaling.loc[0, vvar] * slow_nv

        std.loc[vvar, 'lower'] = slow_nv
        std.loc[vvar, 'upper'] = supp_nv

    return std, scaling


def calc_combined_indicators_natvar(opts, gr_ampl, natvar):
    """
    calc natvar of combined indicators (EQ. 33)
    Args:
        opts: CLI parameter
        gr_ampl: GR amplification
        natvar: NV of basis indicators

    Returns:

    """

    if not opts.precip:
        em_var = 'EMavg_GR_AF'
    else:
        em_var = 'EMavg_Md_GR_AF'

    try:
        gr_ampl_ref = gr_ampl.sel(ctp=slice(PARAMS['REF']['start_cy'], PARAMS['REF']['end_cy']))
    except KeyError:
        gr_ampl_ref = gr_ampl.sel(time=slice(PARAMS['REF']['start_cy'], PARAMS['REF']['end_cy']))

    # calc DM
    gr_ampl_ref['DM_AF'] = gr_ampl_ref['EDavg_GR_AF'] * gr_ampl_ref[em_var]

    # Eq. 33_1 and 33_2
    try:
        s_a_ref = np.sqrt((1/len(gr_ampl_ref.ctp)) * ((gr_ampl_ref['EAavg_GR_AF'] - 1)**2).sum(
            dim='ctp'))
        s_dm_ref = np.sqrt((1/len(gr_ampl_ref.ctp)) * ((gr_ampl_ref['DM_AF'] - 1)**2).sum(dim='ctp'))
    except AttributeError:
        s_a_ref = np.sqrt((1/len(gr_ampl_ref.time)) * ((gr_ampl_ref['EAavg_GR_AF'] - 1)**2).sum(
            dim='time'))
        s_dm_ref = np.sqrt((1/len(gr_ampl_ref.time)) * ((gr_ampl_ref['DM_AF'] - 1)**2).sum(
            dim='time'))

    # Eq. 33_3 - 33_8
    natvar.loc['EA', 'lower'] = ((s_a_ref/s_dm_ref) * natvar.loc['DM', 'lower']).values
    natvar.loc['EA', 'upper'] = ((s_a_ref/s_dm_ref) * natvar.loc['DM', 'upper']).values

    natvar.loc['ES', 'lower'] = np.sqrt((natvar.loc[['DM', 'EA'], 'lower']**2).sum())
    natvar.loc['ES', 'upper'] = np.sqrt((natvar.loc[['DM', 'EA'], 'upper']**2).sum())

    natvar.loc['TEX', 'lower'] = np.sqrt((natvar.loc[['EF', 'ES'], 'lower']**2).sum())
    natvar.loc['TEX', 'upper'] = np.sqrt((natvar.loc[['EF', 'ES'], 'upper']**2).sum())

    # add scaling factor to df
    natvar.loc['SFAC', 'lower'] = (s_a_ref/s_dm_ref).values
    natvar.loc['SFAC', 'upper'] = (s_a_ref/s_dm_ref).values

    return natvar


def run():
    # load CFG parameter
    opts = load_opts(fname=__file__)
    # TODO: rerun when Precip bug in TEA is fixed (Md EM is not stored at the moment)
    data, st_ampl = load_data(opts=opts)
    gr_ref, gr_cc, gr_ampl, gr_cc_ampl, std_ref_gr = get_gr_vals(opts=opts)
    nv, facs = calc_nat_var(opts=opts, st_data=data, std_gr=std_ref_gr)
    nv = calc_combined_indicators_natvar(opts=opts, gr_ampl=gr_ampl, natvar=nv)

    path = Path(f'{opts.outpath}natural_variability/')
    path.mkdir(parents=True, exist_ok=True)
    nv.to_csv(f'{opts.outpath}natural_variability/NV_AF_{opts.param_str}_{opts.region}.csv')
    facs.to_csv(f'{opts.outpath}natural_variability/SFACS_{opts.param_str}_{opts.region}.csv')


if __name__ == '__main__':
    run()
