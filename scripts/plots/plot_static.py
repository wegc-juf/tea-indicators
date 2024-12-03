"""
plot static input of TEA
@author: hst
"""

import argparse
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
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

    parser.add_argument('--precip-var',
                        dest='precip_var',
                        default='Px1h_7to7',
                        type=str,
                        choices=['Px1h', 'P24h', 'Px1h_7to7', 'P24h_7to7'],
                        help='Precipitation variable used.'
                             '[Px1h, P24h, Px1h_7to7 (default), P24h_7to7]')

    parser.add_argument('--threshold',
                        default=99,
                        type=float_1pcd,
                        help='Threshold in degrees Celsius, mm, or as percentile [default: 99].')

    parser.add_argument('--threshold-type',
                        dest='threshold_type',
                        type=str,
                        choices=['perc', 'abs'],
                        default='perc',
                        help='Pass "perc" (default) if percentiles should be used as thresholds or '
                             '"abs" for absolute thresholds.')

    parser.add_argument('--statpath',
                        type=dir_path,
                        default='/data/arsclisys/normal/clim-hydro/TEA-Indicators/static/',
                        help='Path of folder where static file is located.')

    parser.add_argument('--maskpath',
                        type=dir_path,
                        default='/data/arsclisys/normal/clim-hydro/TEA-Indicators/masks/',
                        help='Path of folder where mask file is located.')

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


def load_data(opts):
    """
    load static input data
    Args:
        opts: CLI parameter

    Returns:
        stat: static input data
        masks: mask data
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

    statname = f'{opts.statpath}static_{param_str}_{opts.region}_{opts.dataset}.nc'
    stat = xr.open_dataset(statname)

    maskname = f'{opts.maskpath}{opts.region}_masks_{opts.dataset}.nc'
    masks = xr.open_dataset(maskname)

    return stat, masks


def load_params(opts, mask):
    """
    load plot parameter
    Args:
        opts: CLI parameter
        mask: mask da

    Returns:
        param_dict: dictionary of all necessary plot parameter
    """
    if 'ERA5' not in opts.dataset:
        x, y = 'x', 'y'
        orig = 'lower'
        xn, xx = 0, len(mask['x'])
        yn, yx = 0, len(mask['y'])
    elif opts.region == 'AUT':
        x, y = 'lon', 'lat'
        orig = 'upper'
        xn, xx = 156, 190
        yx, yn = 91, 105
    else:
        x, y = 'lon', 'lat'
        orig = 'upper'
        masky = mask.nw_mask
        xn_val = masky.where(masky.notnull(), drop=True).lon[0].values
        xx_val = masky.where(masky.notnull(), drop=True).lon[-1].values
        yn_val = masky.where(masky.notnull(), drop=True).lat[-1].values
        yx_val = masky.where(masky.notnull(), drop=True).lat[0].values
        xn = np.where(masky.lon == xn_val)[0][0]
        xx = np.where(masky.lon == xx_val)[0][0]
        yn = np.where(masky.lat == yn_val)[0][0]
        yx = np.where(masky.lat == yx_val)[0][0]

    if opts.parameter == 'T':
        cmap = 'YlOrRd'
        unit = '°C'
    else:
        cmap = 'YlGnBu'
        unit = 'mm'

    params_dict = {'x': x, 'y': y, 'cmap': cmap, 'orig': orig, 'xlims': [xn, xx], 'ylims': [yn, yx],
                   'unit': unit}

    return params_dict


def plot_nw_masks(ax, masks, params):
    ax.imshow(masks.nw_mask, origin=params['orig'], vmin=0, vmax=1.5, cmap='Greys')
    ax.imshow(masks.lt1500_mask + 1, origin=params['orig'], vmin=0, vmax=2.5)

    ax.set_title('Region mask / valid grid points')

    ax.scatter(-100, -100, s=40, marker='s', color='#686868', label='GR mask')
    ax.scatter(-100, -100, s=40, marker='s', color='#79d151', label='valid cells')

    ax.legend(loc='upper left', fontsize=7)


def plot_mask(fig, ax, masks, params):
    cmap = colormaps['Greys']
    new_cols = cmap(np.linspace(0, 1, 12))
    cmap = ListedColormap(new_cols)
    boundaries = np.linspace(0, 100, 11)
    norm = BoundaryNorm(boundaries, cmap.N, clip=True)

    map_vals = ax.imshow(masks.mask * 100, origin=params['orig'], cmap=cmap, norm=norm)
    ax.set_title('Weighted region mask')

    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    cb = fig.colorbar(map_vals, cax=cax, orientation='vertical', label='GR fraction [%]')


def plot_area(opts, fig, ax, static, params):
    cmap = colormaps['Greys']
    new_cols = cmap(np.linspace(0, 1, 12))
    cmap = ListedColormap(new_cols)
    boundaries = np.linspace(static.area_grid.min(), static.area_grid.max(), 11)
    norm = BoundaryNorm(boundaries, cmap.N, clip=True)

    map_vals = ax.imshow(static.area_grid, origin=params['orig'], cmap=cmap, norm=norm)
    ax.set_title(f'Area grid {opts.region} ({static.GR_size:.2f} areals)')

    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    cb = fig.colorbar(map_vals, cax=cax, orientation='vertical', label='cell area [areals]')
    cb.formatter = FormatStrFormatter('%.2f')
    cb.update_ticks()


def plot_thresh(fig, ax, static, params):
    boundaries = np.arange(np.floor(static.threshold.min()), np.ceil(static.threshold.max()) + 1)
    cmap = colormaps[params['cmap']]
    new_cols = cmap(np.linspace(0, 1, len(boundaries)))
    cmap = ListedColormap(new_cols)
    norm = BoundaryNorm(boundaries, cmap.N, clip=True)

    map_vals = ax.imshow(static.threshold, origin=params['orig'], cmap=cmap, norm=norm)

    ax.set_title(f'Threshold grid')

    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    cb = fig.colorbar(map_vals, cax=cax, orientation='vertical', label=f'threshold {params["unit"]}')


def run():
    opts = getopts()

    # load data
    stat_ds, mask_ds = load_data(opts=opts)

    # load params
    params = load_params(opts=opts, mask=mask_ds)

    # setup maps
    fig, axs = plt.subplots(2, 2, figsize=(12, 8))

    plot_nw_masks(ax=axs[0, 0], masks=mask_ds, params=params)
    plot_mask(fig=fig, ax=axs[0, 1], masks=mask_ds, params=params)
    plot_area(opts=opts, fig=fig, ax=axs[1, 0], static=stat_ds, params=params)
    plot_thresh(fig, ax=axs[1, 1], static=stat_ds, params=params)

    for irow in range(2):
        for icol in range(2):
            axs[irow, icol].set_xlim(params['xlims'][0], params['xlims'][1])
            axs[irow, icol].set_ylim(params['ylims'][0], params['ylims'][1])
            axs[irow, icol].axis('off')

    fig.subplots_adjust(left=0.1, right=0.9, bottom=0.1, top=0.9, wspace=0.2, hspace=0.2)

    plt.savefig(f'{opts.outpath}static_{opts.region}_{opts.dataset}.png',
                bbox_inches='tight', dpi=300)


if __name__ == '__main__':
    run()
