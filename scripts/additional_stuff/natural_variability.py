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
                                                             calc_basis_amplification_factors)
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
                        default='/data/users/hst/TEA-clean/TEA/',
                        type=dir_path,
                        help='Path of folder where output data should be saved.')

    parser.add_argument('--outpath',
                        default='/data/users/hst/TEA-clean/TEA/',
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
        perc = 'T99.0p'
        split_idx = 2
    else:
        perc = 'P24h_7to7_95.0p'
        split_idx = 4

    files = sorted(glob.glob(f'{opts.inpath}station_indices/*{perc}*.nc'))

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
        station_abbr = basename.split('_')[split_idx][:4].upper()
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


def get_gr_vals(opts):
    """
    load SPARTACUS DEC TEA indicators for GR
    Args:
        opts: CLI parameter

    Returns:

    """

    if opts.parameter == 'T':
        pstr = 'T99.0p'
        em_var = 'EMavg_GR'
    else:
        pstr = 'P24h_7to7_95.0p'
        em_var = 'EMavg_Md_GR'

    ref_data = xr.open_dataset(f'{opts.inpath}dec_indicator_variables/'
                               f'DEC_{pstr}_{opts.region}_WAS_SPARTACUS_1961to2022.nc')

    vkeep = ['EF_GR', 'EDavg_GR', em_var, 'EAavg_GR']
    vdrop = [vvar for vvar in ref_data.data_vars if vvar not in vkeep]
    ref_data = ref_data.drop_vars(vdrop)

    ref_vals, cc_vals = calc_ref_cc_mean(data=ref_data)
    ampl, cc_ampl = calc_basis_amplification_factors(data=ref_data, ref=ref_vals, cc=cc_vals)

    # add combined indicator variables
    ref_vals['DM'] = ref_vals['EDavg_GR'] * ref_vals[em_var]
    ref_vals['tEX'] = ref_vals['EF_GR'] * ref_vals['EDavg_GR'] * ref_vals[em_var]
    cc_vals['DM'] = cc_vals['EDavg_GR'] * cc_vals[em_var]
    cc_vals['tEX'] = cc_vals['EF_GR'] * cc_vals['EDavg_GR'] * cc_vals[em_var]
    cc_ampl['DM'] = cc_ampl['EDavg_GR_AF_CC'] * cc_ampl[f'{em_var}_AF_CC']
    cc_ampl['tEX'] = cc_ampl['EF_GR_AF_CC'] * cc_ampl['EDavg_GR_AF_CC'] * cc_ampl[f'{em_var}_AF_CC']

    return ref_vals, cc_vals, ampl, cc_ampl


def calc_factors(opts, st_am, gr_am):
    """
    calculate factor by which station amplification is larger than GR amplification
    Args:
        opts: CLI parameter
        st_am: station amplification data
        gr_am: GR amplification data

    Returns:
        factors: ds with factors
    """

    if opts.parameter == 'T':
        em_var = 'EMavg'
    else:
        em_var = 'EMavg_Md'

    # calc mean station amplification
    st_am = ((st_am ** 2).mean()) ** (1/2)

    # equation 32_4 and 32_5 left part
    factors = pd.DataFrame(columns=['EF', 'EDavg', em_var])
    factors.loc[0, 'EF'] = (gr_am['EF_GR_AF_CC'].values / st_am['EF'])
    factors.loc[0, 'ED'] = (gr_am['EDavg_GR_AF_CC'].values / st_am['EDavg'])
    factors.loc[0, 'EM'] = (gr_am[f'{em_var}_GR_AF_CC'].values / st_am[em_var])

    # calc DM factor
    gr_dm = 10 ** (np.log10(gr_am['EDavg_GR_AF_CC'].values)
                   + np.log10(gr_am[f'{em_var}_GR_AF_CC'].values))
    st_dm = 10 ** (np.log10(st_am['EDavg']) + np.log10(st_am[em_var]))
    factors.loc[0, 'DM'] = (gr_dm / st_dm)

    # calc tEX factor
    gr_tEX = 10 ** (np.log10(gr_am['EF_GR_AF_CC'].values)
                    + np.log10(gr_am['EDavg_GR_AF_CC'].values)
                    + np.log10(gr_am[f'{em_var}_GR_AF_CC'].values))
    st_tEX = 10 ** (np.log10(st_am['EF']) + np.log10(st_am['EDavg']) + np.log10(st_am['EMavg']))
    factors.loc[0, 'tEX'] = (gr_tEX / st_tEX)

    return factors


def calc_nat_var(opts, st_data, st_acc, gr_acc, facs):
    """
    calculate natural variability (Eq. 32)
    Args:
        opts: CLI parameter
        st_data: station amplification data
        st_acc: station CC amplification data
        gr_acc: GR CC amplification data
        facs: set if scaling factors should be applied

    Returns:
        std: dataframe with std values
    """

    if facs:
        scaling = calc_factors(opts=opts, st_am=st_acc, gr_am=gr_acc)

    std = pd.DataFrame(index=st_data.keys(), columns=['lower', 'upper'])

    for vvar in st_data.keys():
        data = st_data[vvar]
        data = data.loc[data.index <= PARAMS['REF']['end_cy']]

        cupp = (data >= 1).astype(int)
        supp = np.sqrt((1/cupp.sum()) * (cupp*(data - 1)**2).sum())
        slow = np.sqrt((1/(1 - cupp).sum()) * ((1 - cupp)*(data - 1)**2).sum())

        supp_nv = np.sqrt((supp ** 2).mean())
        slow_nv = np.sqrt((slow ** 2).mean())

        if facs:
            supp_nv = scaling.loc[0, vvar] * supp_nv
            slow_nv = scaling.loc[0, vvar] * slow_nv

        std.loc[vvar, 'lower'] = slow_nv
        std.loc[vvar, 'upper'] = supp_nv

    return std


def calc_combined_indicators_natvar(opts, gr_ampl, natvar):

    if opts.parameter == 'T':
        em_var = 'EMavg_GR_AF'
    else:
        em_var = 'EMavg_Md_GR_AF'

    gr_ampl_ref = gr_ampl.sel(ctp=slice(PARAMS['REF']['start_cy'], PARAMS['REF']['end_cy']))

    # calc DM
    gr_ampl_ref['DM_AF'] = gr_ampl_ref['EDavg_GR_AF'] * gr_ampl_ref[em_var]

    # Eq. 33_1 and 33_2
    s_a_ref = np.sqrt((1/len(gr_ampl_ref.ctp)) * ((gr_ampl_ref['EAavg_GR_AF'] - 1)**2).sum(
        dim='ctp'))
    s_de_ref = np.sqrt((1/len(gr_ampl_ref.ctp)) * ((gr_ampl_ref['DM_AF'] - 1)**2).sum(dim='ctp'))

    # Eq. 33_3 - 33_8
    natvar.loc['EA', 'lower'] = ((s_a_ref/s_de_ref) * natvar.loc['DM', 'lower']).values
    natvar.loc['EA', 'upper'] = ((s_a_ref/s_de_ref) * natvar.loc['DM', 'upper']).values

    natvar.loc['ES', 'lower'] = np.sqrt((natvar.loc[['ED', 'EM', 'EA'], 'lower']**2).sum())
    natvar.loc['ES', 'upper'] = np.sqrt((natvar.loc[['ED', 'EM', 'EA'], 'upper']**2).sum())

    natvar.loc['TEX', 'lower'] = np.sqrt((natvar.loc[['EF', 'ES'], 'lower']**2).sum())
    natvar.loc['TEX', 'upper'] = np.sqrt((natvar.loc[['EF', 'ES'], 'upper']**2).sum())

    return natvar


def run():
    opts = getopts()

    # apply factors A_cc_GR/A_cc_station for all parameters
    apply_facs = True
    if opts.parameter == 'P':
        apply_facs = False

    data, st_ampl = load_data(opts=opts)
    gr_ref, gr_cc, gr_ampl, gr_cc_ampl = get_gr_vals(opts=opts)
    nv = calc_nat_var(opts=opts, st_data=data, st_acc=st_ampl, gr_acc=gr_cc_ampl, facs=apply_facs)
    nv = calc_combined_indicators_natvar(opts=opts, gr_ampl=gr_ampl, natvar=nv)

    if opts.parameter == 'T':
        pstr = 'T99.0p'
    else:
        pstr = 'P24h_7to7_95.0p'

    path = Path(f'{opts.outpath}natural_variability/')
    path.mkdir(parents=True, exist_ok=True)
    nv.to_csv(f'{opts.outpath}natural_variability/NV_AF_{pstr}_{opts.region}.csv')


if __name__ == '__main__':
    run()
