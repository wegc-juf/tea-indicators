#!/opt/virtualenv3.11/bin/python3
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

sys.path.append('/home/hst/tea-indicators/scripts/calc_indices/')
from calc_amplification_factors import calc_ref_cc_mean, calc_basis_amplification_factors
from calc_decadal_indicators import rolling_decadal_mean

sys.path.append('/home/hst/tea-indicators/scripts/misc/')
from general_functions import create_history, ref_cc_params

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
        perc = 'T99p'
    else:
        perc = 'P24h_7to7_95p'

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
        if basename.split('_')[2][:4].upper() not in stations:
            continue
        # load data
        data = xr.open_dataset(file)

        # calculate decadal means
        data = rolling_decadal_mean(data=data)

        # calculate reference and cc values
        ref, cc = calc_ref_cc_mean(data=data)

        # calc amplification factors
        ampl_facs, ampl_cc_facs = calc_basis_amplification_factors(data=data, ref=ref, cc=cc)

        for vvar in ampl.columns:
            if vvar in ['tEX', 'DM']:
                continue
            basename = os.path.basename(file)
            ampl.loc[basename.split('_')[2][:4].upper(), vvar] = ampl_cc_facs[
                f'{vvar}_AF_CC'].values
        ampl.loc[basename.split('_')[2][:4].upper(), 'DM'] = ampl.loc[basename.split('_')[2][
                                                                      :4].upper(), 'EDavg'] * \
                                                             ampl.loc[basename.split('_')[2][
                                                                      :4].upper(), 'EMavg']
        ampl.loc[basename.split('_')[2][:4].upper(), 'tEX'] = ampl.loc[basename.split('_')[2][
                                                                       :4].upper(), 'EF'] * \
                                                              ampl.loc[basename.split('_')[2][
                                                                       :4].upper(), 'EDavg'] * \
                                                              ampl.loc[basename.split('_')[2][
                                                                       :4].upper(), 'EMavg']

        tef_stat = pd.DataFrame(index=data.ctp.values,
                                data=ampl_facs.EF_AF.values,
                                columns=[basename.split('_')[2][:4].upper()])
        ed_stat = pd.DataFrame(index=data.ctp.values,
                               data=ampl_facs.EDavg_AF.values,
                               columns=[basename.split('_')[2][:4].upper()])
        em_stat = pd.DataFrame(index=data.ctp.values,
                               data=ampl_facs.EMavg_AF.values,
                               columns=[basename.split('_')[2][:4].upper()])

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

    # equation 32_4 and 32_5 left part
    # ampl = ampl.mean(axis=0)

    return data, ampl


def get_gr_vals(opts):
    """
    load SPARTACUS DEC TEA indicators for GR
    Args:
        opts: CLI parameter

    Returns:

    """

    if opts.parameter == 'T':
        pstr = 'T99p'
    else:
        pstr = 'Precip24Hsum_7to7_95percentile'

    ref_data = xr.open_dataset(f'{opts.inpath}dec_indicator_variables/'
                               f'DEC_{pstr}_{opts.region}_WAS_SPARTACUS_1961to2022.nc')

    vkeep = ['EF_GR', 'EDavg_GR', 'EMavg_GR', 'EAavg_GR']
    vdrop = [vvar for vvar in ref_data.data_vars if vvar not in vkeep]
    ref_data = ref_data.drop_vars(vdrop)

    ref_vals, cc_vals = calc_ref_cc_mean(data=ref_data)
    ampl, cc_ampl = calc_basis_amplification_factors(data=ref_data, ref=ref_vals, cc=cc_vals)

    # add combined indicator variables
    ref_vals['DM'] = ref_vals['EDavg_GR'] * ref_vals['EMavg_GR']
    ref_vals['tEX'] = ref_vals['EF_GR'] * ref_vals['EDavg_GR'] * ref_vals['EMavg_GR']
    cc_vals['DM'] = cc_vals['EDavg_GR'] * cc_vals['EMavg_GR']
    cc_vals['tEX'] = cc_vals['EF_GR'] * cc_vals['EDavg_GR'] * cc_vals['EMavg_GR']
    cc_ampl['DM'] = cc_ampl['EDavg_GR_AF_CC'] * cc_ampl['EMavg_GR_AF_CC']
    cc_ampl['tEX'] = cc_ampl['EF_GR_AF_CC'] * cc_ampl['EDavg_GR_AF_CC'] * cc_ampl['EMavg_GR_AF_CC']

    return ref_vals, cc_vals, ampl, cc_ampl


def calc_factors(st_am, gr_am):
    """
    calculate factor by which station amplification is larger than GR amplification
    Args:
        st_am: station amplification data
        gr_am: GR amplification data

    Returns:
        factors: ds with factors
    """

    # calc mean station amplification
    st_am = ((st_am ** 2).mean()) ** (1/2)

    # equation 32_4 and 32_5 left part
    factors = pd.DataFrame(columns=['EF', 'EDavg', 'EMavg'])
    factors.loc[0, 'EF'] = (gr_am['EF_GR_AF_CC'].values / st_am['EF'])
    factors.loc[0, 'ED'] = (gr_am['EDavg_GR_AF_CC'].values / st_am['EDavg'])
    factors.loc[0, 'EM'] = (gr_am['EMavg_GR_AF_CC'].values / st_am['EMavg'])

    # calc DM factor
    gr_dm = 10 ** (np.log10(gr_am['EDavg_GR_AF_CC'].values)
                   + np.log10(gr_am['EMavg_GR_AF_CC'].values))
    st_dm = 10 ** (np.log10(st_am['EDavg']) + np.log10(st_am['EMavg']))
    factors.loc[0, 'DM'] = (gr_dm / st_dm)

    # calc tEX factor
    gr_tEX = 10 ** (np.log10(gr_am['EF_GR_AF_CC'].values)
                    + np.log10(gr_am['EDavg_GR_AF_CC'].values)
                    + np.log10(gr_am['EMavg_GR_AF_CC'].values))
    st_tEX = 10 ** (np.log10(st_am['EF']) + np.log10(st_am['EDavg']) + np.log10(st_am['EMavg']))
    factors.loc[0, 'tEX'] = (gr_tEX / st_tEX)

    return factors


def calc_nat_var(st_data, st_acc, gr_acc, facs):
    """
    calculate natural variability (Eq. 32)
    Args:
        st_data: station amplification data
        st_acc: station CC amplification data
        gr_acc: GR CC amplification data
        facs: set if scaling factors should be applied

    Returns:
        std: dataframe with std values
    """

    if facs:
        scaling = calc_factors(st_am=st_acc, gr_am=gr_acc)

    std = pd.DataFrame(index=st_data.keys(), columns=['lower', 'upper'])

    for vvar in st_data.keys():
        data = st_data[vvar]

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


def calc_combined_indicators_natvar(gr_ampl, natvar):

    gr_ampl_ref = gr_ampl.sel(ctp=slice(PARAMS['REF']['start_cy'], PARAMS['REF']['end_cy']))
    # calc DM
    gr_ampl_ref['DM_AF'] = gr_ampl_ref['EDavg_GR_AF'] * gr_ampl_ref['EMavg_GR_AF']

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
    nv = calc_nat_var(st_data=data, st_acc=st_ampl, gr_acc=gr_cc_ampl, facs=apply_facs)
    nv = calc_combined_indicators_natvar(gr_ampl=gr_ampl, natvar=nv)

    path = Path(f'{opts.outpath}natural_variability/')
    path.mkdir(parents=True, exist_ok=True)
    nv.to_csv(f'{opts.outpath}natural_variability/'
              f'NV_{opts.parameter}_{opts.region}.nc')


if __name__ == '__main__':
    run()
