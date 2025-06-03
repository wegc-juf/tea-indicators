#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: hst

"""
import argparse
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

from scripts.general_stuff.general_functions import ref_cc_params

PARAMS = ref_cc_params()


def getopts():
    """
    get arguments
    :return: command line parameters
    """

    parser = argparse.ArgumentParser()

    def dir_path(path):
        if os.path.isdir(path):
            return path
        else:
            raise argparse.ArgumentTypeError(f'{path} is not a valid path')

    parser.add_argument('--parameter',
                        type=str,
                        default='T',
                        choices=['T', 'P'],
                        help='Temperature (T, default) or Precipitation (P).')

    parser.add_argument('--region',
                        type=str,
                        default='AUT',
                        choices=['AUT', 'SEA'],
                        help='Austria (AUT, default) or SE-Austria (SEA).')

    parser.add_argument('--inpath',
                        default='/data/users/hst/TEA-clean/TEA/paper_data/station/'
                                'dec_indicator_variables/',
                        type=dir_path,
                        help='Path of folder where output data should be saved.')

    parser.add_argument('--outpath',
                        default='/data/users/hst/TEA-clean/TEA/misc_data/NatVarCheck/',
                        type=dir_path,
                        help='Path of folder where output data should be saved.')

    myopts = parser.parse_args()

    return myopts


def load_data(opts):
    """
    load ACTEM station data and apply decadal moving average
    :return: data (station data)
    :return: ampl (amplification factors)
    """

    if opts.parameter == 'T':
        perc = 'Tx99.0p'
        split_idx = 2
    else:
        perc = 'P24h_7to7_95.0p'
        split_idx = 4

    files = sorted(glob.glob(f'{opts.inpath}*{perc}*.nc'))
    af_files = sorted(glob.glob(f'{opts.inpath}amplification/*{perc}*.nc'))

    if opts.region == 'AUT':
        stations = ['GRAZ', 'INNS', 'KREM', 'SALZ', 'WIEN']
    else:
        stations = ['GRAZ', 'BADG', 'DEUT']

    vkeep = ['EF', 'ED_avg', 'EM_avg', 'tEX']
    vkeep_af = [f'{vvar}_AF' for vvar in vkeep]
    vkeep_af_cc = [f'{vvar}_AF_CC' for vvar in vkeep]
    ef = pd.DataFrame()
    ed = pd.DataFrame()
    em = pd.DataFrame()
    tex = pd.DataFrame()
    ampl = pd.DataFrame(index=stations, columns=['EF', 'ED_avg', 'EM_avg', 'DM', 'tEX'])

    for ifile, file in enumerate(files):
        basename = os.path.basename(file)
        station_abbr = basename.split('_')[split_idx][:4].upper()
        if station_abbr not in stations:
            continue
        # load data
        data = xr.open_dataset(file)
        vdrop = [vvar for vvar in data.data_vars if vvar not in vkeep]
        data = data.drop(vdrop)

        # load amplification factor data and stor CC variables in separate ds
        ampl_facs = xr.open_dataset(af_files[ifile])
        vdrop_af = [vvar for vvar in ampl_facs.data_vars if vvar not in vkeep_af]
        ampl_cc_facs = ampl_facs[vkeep_af_cc]
        ampl_facs = ampl_facs.drop(vdrop_af)

        for vvar in ampl.columns:
            if vvar == 'DM':
                continue
            ampl.loc[station_abbr, vvar] = ampl_cc_facs[f'{vvar}_AF_CC'].values
        ampl.loc[station_abbr, 'DM'] = (ampl.loc[station_abbr, 'ED_avg']
                                        * ampl.loc[station_abbr, 'EM_avg'])

        tef_stat = pd.DataFrame(index=data.time.values,
                                data=ampl_facs.EF_AF.values,
                                columns=[station_abbr])
        ed_stat = pd.DataFrame(index=data.time.values,
                               data=ampl_facs.ED_avg_AF.values,
                               columns=[station_abbr])
        em_stat = pd.DataFrame(index=data.time.values,
                               data=ampl_facs.EM_avg_AF.values,
                               columns=[station_abbr])
        tex_stat = pd.DataFrame(index=data.time.values,
                                data=ampl_facs.tEX_AF.values,
                                columns=[station_abbr])

        ef = pd.concat([ef, tef_stat], axis=1)
        ed = pd.concat([ed, ed_stat], axis=1)
        em = pd.concat([em, em_stat], axis=1)
        tex = pd.concat([tex, tex_stat], axis=1)

    # calc compound indicators
    dm = ed * em

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

        tex['year'] = tex.index.year
        tex.loc[tex['year'] < 1920, 'KREM'] = np.nan
        tex = tex.drop(['year'], axis=1)

    # combine in dict
    data = {'EF': ef, 'ED': ed, 'EM': em, 'DM': dm, 'tEX': tex}

    return data, ampl


def get_gr_vals(opts):
    """
    load SPARTACUS DEC TEA indicators for GR
    Args:
        opts: CLI parameter

    Returns:

    """

    if opts.parameter == 'T':
        pstr = 'Tx99.0p'
        em_var = 'EM_avg_GR'
    else:
        pstr = 'P24h_7to7_95.0p'
        em_var = 'EM_avg_Md_GR'

    ref_data = xr.open_dataset(f'/data/users/hst/TEA-clean/TEA/paper_data/dec_indicator_variables/'
                               f'DEC_{pstr}_{opts.region}_annual_SPARTACUS_1961to2024.nc')
    ref_data_af = xr.open_dataset(f'/data/users/hst/TEA-clean/TEA/paper_data/'
                                  f'dec_indicator_variables/amplification/'
                                  f'AF_{pstr}_{opts.region}_annual_SPARTACUS_1961to2024.nc')

    vkeep = ['EF_GR', 'ED_avg_GR', em_var, 'EA_avg_GR', 'tEX_GR']
    vdrop = [vvar for vvar in ref_data.data_vars if vvar not in vkeep]
    ref_data = ref_data.drop_vars(vdrop)

    vkeep_af = [f'{vvar}_AF' for vvar in vkeep]
    vkeep_af_cc = [f'{vvar}_CC' for vvar in vkeep_af]
    vdrop_af = [vvar for vvar in ref_data_af.data_vars if vvar not in vkeep_af]
    cc_ampl = ref_data_af[vkeep_af_cc]
    ampl = ref_data_af.drop_vars(vdrop_af)

    ref_vals, cc_vals = calc_ref_cc_mean(data=ref_data)

    # add combined indicator variables
    ref_vals['DM'] = ref_vals['ED_avg_GR'] * ref_vals[em_var]
    cc_vals['DM'] = cc_vals['ED_avg_GR'] * cc_vals[em_var]
    cc_ampl['DM'] = cc_ampl['ED_avg_GR_AF_CC'] * cc_ampl[f'{em_var}_AF_CC']
    ampl['DM_GR_AF'] = ampl['ED_avg_GR_AF'] * ampl[f'{em_var}_AF']


    # calc std of ref period
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

    if opts.parameter == 'T':
        em_var = 'EM_avg'
    else:
        em_var = 'EM_avg_Md'

    vnames = {'EF': 'EF_GR_AF', 'ED': 'ED_avg_GR_AF', 'EM': f'{em_var}_GR_AF',
              'DM': 'DM_GR_AF', 'tEX': 'tEX_GR_AF'}

    factors = pd.DataFrame(columns=['EF', 'ED_avg', em_var, 'DM', 'tEX'])

    for vvar in st_data.keys():
        vdata = st_data[vvar].loc[PARAMS['REF']['start_cy']:PARAMS['REF']['end_cy']]
        s_ref_k = vdata.std()
        fac = s_ref_gr[vnames[vvar]] / np.sqrt((s_ref_k ** 2).sum() / len(s_ref_k.index))
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
        supp = np.sqrt((1 / cupp.sum()) * (cupp * (data - 1) ** 2).sum())
        slow = np.sqrt((1 / (1 - cupp).sum()) * ((1 - cupp) * (data - 1) ** 2).sum())

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

    if opts.parameter == 'T':
        em_var = 'EM_avg_GR_AF'
    else:
        em_var = 'EM_avg_Md_GR_AF'

    gr_ampl_ref = gr_ampl.sel(time=slice(PARAMS['REF']['start_cy'], PARAMS['REF']['end_cy']))

    # calc DM
    gr_ampl_ref['DM_AF'] = gr_ampl_ref['ED_avg_GR_AF'] * gr_ampl_ref[em_var]

    # Eq. 33_1 and 33_2
    s_a_ref = np.sqrt((1 / len(gr_ampl_ref.time)) * ((gr_ampl_ref['EA_avg_GR_AF'] - 1) ** 2).sum(
        dim='time'))
    s_dm_ref = np.sqrt(
        (1 / len(gr_ampl_ref.time)) * ((gr_ampl_ref['DM_AF'] - 1) ** 2).sum(dim='time'))

    # Eq. 33_3 - 33_8
    natvar.loc['EA', 'lower'] = ((s_a_ref / s_dm_ref) * natvar.loc['DM', 'lower']).values
    natvar.loc['EA', 'upper'] = ((s_a_ref / s_dm_ref) * natvar.loc['DM', 'upper']).values

    natvar.loc['ES', 'lower'] = np.sqrt((natvar.loc[['DM', 'EA'], 'lower'] ** 2).sum())
    natvar.loc['ES', 'upper'] = np.sqrt((natvar.loc[['DM', 'EA'], 'upper'] ** 2).sum())

    natvar.loc['TEX', 'lower'] = np.sqrt((natvar.loc[['EF', 'ES'], 'lower'] ** 2).sum())
    natvar.loc['TEX', 'upper'] = np.sqrt((natvar.loc[['EF', 'ES'], 'upper'] ** 2).sum())

    # add scaling factor to df
    natvar.loc['SFAC', 'lower'] = (s_a_ref / s_dm_ref).values
    natvar.loc['SFAC', 'upper'] = (s_a_ref / s_dm_ref).values

    return natvar


def run():
    opts = getopts()

    data, st_ampl = load_data(opts=opts)
    gr_ref, gr_cc, gr_ampl, gr_cc_ampl, std_ref_gr = get_gr_vals(opts=opts)
    nv, facs = calc_nat_var(opts=opts, st_data=data, std_gr=std_ref_gr)
    nv = calc_combined_indicators_natvar(opts=opts, gr_ampl=gr_ampl, natvar=nv)

    if opts.parameter == 'T':
        pstr = 'T99.0p'
    else:
        pstr = 'P24h_7to7_95.0p'

    path = Path(f'{opts.outpath}natural_variability/')
    path.mkdir(parents=True, exist_ok=True)
    nv.to_csv(f'{opts.outpath}natural_variability/NV_AF_{pstr}_{opts.region}.csv')
    facs.to_csv(f'{opts.outpath}natural_variability/SFACS_{pstr}_{opts.region}.csv')


if __name__ == '__main__':
    run()
