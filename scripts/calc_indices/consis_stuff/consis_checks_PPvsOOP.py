import argparse
import cftime as cft
import glob
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
import os
import xarray as xr

def get_opts():
    """
    get CLI parameter
    Returns:
        opts: CLI parameter
    """

    def dir_path(path):
        if os.path.isdir(path):
            return path
        else:
            raise argparse.ArgumentTypeError(f'{path} is not a valid path.')

    parser = argparse.ArgumentParser()

    parser.add_argument('--inpath',
                        default='/data/users/hst/TEA-clean/TEA/code_review/',
                        type=dir_path,
                        help='Path of folder where data is located.')

    parser.add_argument('--parameter',
                        default='Tx',
                        choices=['Tx', 'P24h_7to7'],
                        type=str,
                        help='Parameter for which the TEA indices should be calculated'
                             '[default: Tx].')

    parser.add_argument('--threshold',
                        default=99,
                        type=float,
                        help='Threshold in degrees Celsius, mm, or as percentile [default: 99].')

    parser.add_argument('--period',
                        dest='period',
                        default='annual',
                        type=str,
                        choices=['monthly', 'seasonal', 'annual', 'WAS', 'ESS', 'JJA', 'EWS'],
                        help='Climatic time period (CTP) of interest. '
                             'Options: monthly, seasonal, WAS, ESS, EWS, JJA, and  annual [default].')

    parser.add_argument('--start',
                        default=1961,
                        type=int,
                        help='Start of the interval to be processed [default: 1961].')

    parser.add_argument('--end',
                        default=1970,
                        type=int,
                        help='End of the interval to be processed [default: 1970].')

    parser.add_argument('--region',
                        default='AUT',
                        type=str,
                        help='Geo region [options: AUT (default), Austrian state name, '
                             'or ISO2 code of european country].')

    parser.add_argument('--dataset',
                        dest='dataset',
                        default='SPARTACUS',
                        choices=['SPARTACUS', 'ERA5', 'ERA5Land'],
                        type=str,
                        help='Input dataset [default: SPARTACUS].')

    parser.add_argument('--level',
                        dest='level',
                        default='DBV',
                        choices=['DBV', 'CTP', 'DEC'],
                        type=str,
                        help='Level of data to compare.')

    parser.add_argument('--plot',
                        dest='plot',
                        action='store_true',
                        help='Set if plots should be shown.')

    myopts = parser.parse_args()

    return myopts


def load_data(opts):

    sdir = 'daily_basis_variables'
    if opts.level == 'CTP':
        sdir = 'ctp_indicator_variables'
    elif opts.level == 'DEC':
        sdir = 'dec_indicator_variables'

    pstr = f'{opts.parameter}{opts.threshold:.1f}p'
    if opts.parameter != 'Tx':
        pstr = f'{opts.parameter}_{opts.threshold:.1f}p'

    ods = xr.open_dataset(f'{opts.inpath}{sdir}/{opts.level}_{pstr}_{opts.region}'
                          f'_{opts.period}_{opts.dataset}_{opts.start}to{opts.end}.nc')

    nds = xr.open_dataset(f'{opts.inpath}{sdir}/{opts.level}_{pstr}_{opts.region}'
                          f'_{opts.period}_{opts.dataset}_{opts.start}to{opts.end}.nc')

    return ods, nds


def plot_gr_vars(opts, ods, nds):

    gr_vars_all = [vvar for vvar in ods.data_vars if 'GR' in vvar]
    if opts.level == 'DBV':
        gr_vars_all.append('DTEA_frac')
        gr_vars_plt = gr_vars_all
    else:
        gr_vars_plt = ['EF_GR', 'ED_avg_GR', 'EM_avg_GR', 'EA_avg_GR', 'ES_avg_GR', 'TEX_GR']

    if opts.plot:
        fig, axs = plt.subplots(2, 3, figsize=(15, 7))
        axs = axs.reshape(-1)

    print('-- GR VARIABLES --')
    iplt, ldiff = 0, 0
    for ivar, vvar in enumerate(gr_vars_all):
        if opts.plot:
            if vvar in gr_vars_plt:
                axs[iplt].plot(ods[vvar], 'o-', color='tab:grey')
                axs[iplt].plot(nds[vvar], 'o-', color='tab:green')
                axs[iplt].set_title(vvar)
                iplt += 1
        diff = ods[vvar] - nds[vvar]
        if diff.max().values != 0:
            print(vvar, diff.max().values)
            ldiff += 1

    if ldiff == 0:
        print('All GR vars OK.')

    if opts.plot:
        plt.show()


def check_grid_data(opts, ods, nds):

    gvars = ['DTEC', 'DTEM', 'DTEEC']
    if opts.level != 'DBV':
        gvars = [vvar for vvar in ods.data_vars if 'GR' not in vvar]

    print('-- GRID VARIABLES --')
    ldiff = 0
    for gvar in gvars:
        diff = ods[gvar] - nds[gvar]
        if diff.max().values != 0:
            print(gvar, diff.max().values)
            ldiff += 1

    if ldiff == 0:
        print('All GRID vars OK.')

def run():
    opts = get_opts()

    if opts.end - opts.start < 9 and opts.level == 'DEC':
        raise AttributeError('Please pass more than 10 years to compare decadal results.')

    old, new = load_data(opts=opts)

    plot_gr_vars(opts=opts, ods=old, nds=new)
    check_grid_data(opts=opts, ods=old, nds=new)


if __name__ == '__main__':
    run()
