"""
plot CTP TEA indicators
@author: hst
"""

import argparse
import glob
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter, FixedLocator
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib import colormaps
import numpy as np
import os
import pandas as pd
import re
import xarray as xr

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
                        type=float_1pcd,
                        help='Threshold in degrees Celsius, mm, or as percentile [default: 99].')

    parser.add_argument('--threshold_type',
                        type=str,
                        choices=['perc', 'abs'],
                        default='perc',
                        help='Pass "perc" (default) if percentiles should be used as thresholds or '
                             '"abs" for absolute thresholds.')

    parser.add_argument('--inpath',
                        type=dir_path,
                        default='/data/users/hst/TEA-clean/TEA/ctp_indicator_variables/',
                        help='Path of input data folder.')

    parser.add_argument('--outpath',
                        type=dir_path,
                        default='/nas/home/hst/work/TEAclean/plots/',
                        help='Path of folder where plots should be saved.')

    parser.add_argument('--dataset',
                        dest='dataset',
                        default='SPARTACUS',
                        type=str,
                        choices=['SPARTACUS', 'ERA5', 'ERA5Land'],
                        help='Input dataset. Options: SPARTACUS (default), ERA5, ERA5Land.')

    myopts = parser.parse_args()

    return myopts

def preprocess(ds_in):
    """
    drop map data before loading the dataset
    Args:
        ds_in: input ds

    Returns:
        ds_out: output ds

    """

    ds = ds_in.copy()
    vdrop = [vvar for vvar in ds.data_vars if 'GR' not in vvar]

    ds_out = ds.drop_vars(vdrop)

    return ds_out


def load_data(opts):
    """
    load CTP indicator data (output of calc_TEA.py)
    Args:
        opts: CLI parameter

    Returns:
        ds: dataset of CTP variables

    """

    pstr = opts.parameter
    if opts.parameter == 'P':
        pstr = f'{opts.precip_var}_'

    unit, unit_str = '°C', 'degC'
    if opts.parameter == 'P':
        unit, unit_str = 'mm', 'mm'

    param_str = f'{pstr}{opts.threshold:.1f}p'
    if opts.threshold_type == 'abs':
        param_str = f'{pstr}{opts.threshold:.1f}{unit_str}'

    # TODO: only load select files in given timespan
    files = sorted(glob.glob(
        f'{opts.inpath}CTP_{param_str}_{opts.region}_{opts.period}_{opts.dataset}_*.nc'
    ))

    ds = xr.open_mfdataset(files, data_vars='minimal', preprocess=preprocess)

    return ds

def plot_subplot(opts, ax, data, vvar):
    cols = {'EF': 'tab:blue', 'ED': 'tab:purple', 'EM': 'tab:orange', 'EA': 'tab:red',
            'TEX': 'tab:red'}
    params = {'EF_GR': {'title': 'Event Frequency', 'ylbl': f'EF [{data.units}]'},
              'ED_GR': {'title': 'Cumulative Event Duration', 'ylbl': f'ED [{data.units}]'},
              'EDavg_GR': {'title': 'Average Event Duration', 'ylbl': f'EDavg [{data.units}]'},
              'EM_GR': {'title': 'Cumulative Event Magnitude', 'ylbl': f'EM [{data.units}]'},
              'EMavg_GR': {'title': 'Average Event Magnitude', 'ylbl': f'EMavg [{data.units}]'},
              'EAavg_GR': {'title': 'Average Event Area', 'ylbl': f'EAavg [{data.units}]'},
              'TEX_GR': {'title': 'Total Events Extremity', 'ylbl': f'TEX [{data.units}]'}}

    xticks = np.arange(opts.start, opts.end + 1)
    ax.plot(xticks, data, color=cols[vvar])

    ax.grid(linestyle=':', color='lightgrey')
    ax.minorticks_on()
    ax.set_xlim(1960, 2023)
    ax.xaxis.set_minor_locator(FixedLocator(np.arange(opts.start-1, opts.end+1)))

    vmax = data.max().values
    if 'avg' in data.name and vvar not in ['EA', 'TEX']:
        yx = np.ceil(vmax)
    elif vvar in ['EA', 'TEX']:
        yx = np.ceil(vmax / 50) * 50
    else:
        yx = np.ceil(vmax / 5) * 5
    ax.set_ylim(0, yx)

    ax.set_title(params[data.name]['title'])
    ax.set_ylabel(params[data.name]['ylbl'])


def run():
    opts = getopts()

    data = load_data(opts=opts)

    fig, axs = plt.subplots(2, 3, figsize=(15, 7))

    pvars = ['EF', 'ED', 'EM', 'EA']

    for ivar, vvar in enumerate(pvars):
        if f'{vvar}_GR' in data.data_vars:
            plot_subplot(opts=opts, ax=axs[0, ivar], data=data[f'{vvar}_GR'], vvar=vvar)
        if f'{vvar}avg_GR' in data.data_vars and vvar != 'EA':
            plot_subplot(opts=opts, ax=axs[1, ivar], data=data[f'{vvar}avg_GR'], vvar=vvar)
        elif f'{vvar}avg_GR' in data.data_vars and vvar == 'EA':
            plot_subplot(opts=opts, ax=axs[1, 0], data=data[f'{vvar}avg_GR'], vvar=vvar)

    fig.subplots_adjust(left=0.1, right=0.9, bottom=0.1, top=0.9, wspace=0.2, hspace=0.2)
    plt.savefig(f'{opts.outpath}CTP_{opts.region}_{opts.dataset}.png',
                bbox_inches='tight', dpi=300)

    fig2, axs2 = plt.subplots(1, 1, figsize=(7, 5))
    plot_subplot(opts=opts, ax=axs2, data=data[f'TEX_GR'], vvar='TEX')
    plt.savefig(f'{opts.outpath}CTP_TEX_{opts.region}_{opts.dataset}.png',
                bbox_inches='tight', dpi=300)

    # TODO: adjust for precip data
    # TODO: check with ERA5 data


if __name__ == '__main__':
    run()
