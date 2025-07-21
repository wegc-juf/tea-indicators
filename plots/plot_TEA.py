"""
plot decadal-mean TEA indicators (amplification)
@author: hst
"""

import argparse
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.ticker import FormatStrFormatter, FixedLocator
import numpy as np
import os
import xarray as xr

from common.config import load_opts
from teametrics.common.general_functions import ref_cc_params

PARAMS = ref_cc_params()


def _getopts():
    """
    get command line arguments

    Returns:
        opts: command line parameters
    """

    parser = argparse.ArgumentParser()

    parser.add_argument('--config-file', '-cf',
                        dest='config_file',
                        type=str,
                        default='../TEA_CFG.yaml',
                        help='TEA configuration file (default: TEA_CFG.yaml)')

    myopts = parser.parse_args()

    return myopts


def load_data(opts):
    """
    load decadal TEA indicators data
    Args:
        opts: CLI parameter

    Returns:

    """
    agr_str = ''
    if opts.agr:
        agr_str = 'AGR-'

    ds = xr.open_dataset(f'{opts.outpath}/dec_indicator_variables/'
                         f'DEC_{opts.param_str}_{agr_str}{opts.region}_{opts.period}_{opts.dataset}'
                         f'_{opts.start}to{opts.end}.nc')

    # only keep relevant variables
    vvars_dec = ['EF_AGR', 'ED_avg_AGR', 'EM_avg_AGR', 'EA_avg_AGR', 'TEX_AGR',
                 'ES_avg_AGR', 'EF', 'ED_avg', 'EM_avg']
    ds = ds[vvars_dec]

    # prep dec_cc data
    cc_data = ds.sel(time=slice(f'{opts.cc_period[0]}-01-01', f'{opts.cc_period[1]}-12-31'))
    cc_data = cc_data.mean(dim='time')
    for var in cc_data.data_vars:
        ds[var + '_CC'] = cc_data[var]
    ds = ds.drop(['EF', 'ED_avg', 'EM_avg'])

    return ds


def gr_plot_params(vname):
    """
    define props for each GR variable
    Args:
        vname: variable name

    Returns:
        params: dict with properties for plotting

    """
    params = {'EF_AGR': {'col': 'tab:blue',
                            'ylbl': r'EF $(F)$',
                            'title': 'Event Frequency (Annual)',
                            'unit': 'ev/yr'},
              'ED_avg_AGR': {'col': 'tab:purple',
                                'ylbl': r'ED $(D)$',
                                'title': 'Average Event Duration (events-mean)',
                                'unit': 'days'},
              'EM_avg_AGR': {'col': 'tab:orange',
                                'ylbl': r'EM $(M)$',
                                'title': 'Average Exceedance Magnitude (daily-mean)',
                                'unit': '°C'},
              'EA_avg_AGR': {'col': 'tab:red',
                                'ylbl': r'EA $(A)$',
                                'title': 'Average Exceedance Area (daily-mean)',
                                'unit': 'areals'}
              }

    return params[vname]


def get_lims(data, rval):
    """
    get min and max values and round them for plotting
    Args:
        data: input data to get lims for
        rval: rounding value

    Returns:
        vmin: rounded minimum value
        vmax: rounded maximum value
    """

    v_min, v_max = None, None
    try:
        for vvar in data.data_vars:
            da = data[vvar]

            vn = da.min().values
            vx = da.max().values

            # round to next round val (rval)
            vmin = np.floor(vn / rval) * rval
            vmax = np.ceil(vx / rval) * rval

            if v_min is None or v_min > vmin:
                v_min = vmin

            if v_max is None or v_max < vmax:
                v_max = vmax

    except AttributeError:
        vn = data.min().values
        vx = data.max().values

        # round to next round val (rval)
        vmin = np.floor(vn / rval) * rval
        vmax = np.ceil(vx / rval) * rval

        if v_min is None or v_min > vmin:
            v_min = vmin

        if v_max is None or v_max < vmax:
            v_max = vmax


    lims = [v_min, v_max, rval]


    return lims


def map_plot_params(vname):
    """
    define props for each map variable
    Args:
        vname: variable name

    Returns:
        params: dict with properties for plotting

    """
    params = {'EF_CC': {'cmap': 'Blues',
                           'lbl': r'$F_\mathrm{CC}$',
                           'title': f'Event Frequency (EF) (CC2010-2024)'},
              'ED_avg_CC': {'cmap': 'Purples',
                               'lbl': r'$D_\mathrm{CC}$',
                               'title': f'Event Duration (ED) (CC2010-2024)'},
              'EM_avg_CC': {'cmap': 'Oranges',
                               'lbl': r'$M_\mathrm{CC}$',
                               'title': f'Exceedance Magnitude (EM) (CC2010-2024)'}
              }

    return params[vname]


def plot_gr_data(opts, ax, data):
    """
    plot GR data
    Args:
        opts: CLI parameter
        ax: axis to plot on
        data: data to plot

    Returns:

    """
    props = gr_plot_params(vname=data.name)

    xticks = np.arange(opts.start, opts.end + 1)

    ax.plot(xticks, data, 'o-', color=props['col'], markersize=3, linewidth=2)
    ax.set_ylabel(props['ylbl'], fontsize=12)
    ax.tick_params(axis='both', which='major', labelsize=10)
    ax.minorticks_on()
    ax.grid(color='gray', which='major', linestyle=':')
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

    xn, xx = np.floor(opts.start / 5) * 5, np.ceil(opts.end / 5) * 5
    ax.set_xlim(xn, xx)
    ax.xaxis.set_minor_locator(FixedLocator(np.arange(xn, xx)))

    if data.name == 'EA_avg_AGR':
        rval, fac = 5, 2
    elif data.name == 'EF_AGR':
        rval, fac = 2, 1
    else:
        rval, fac = 0.5, 0.25

    lims = get_lims(data=data, rval=rval)
    ax.set_yticks(np.arange(lims[0], lims[1] + lims[2] * fac, lims[2] * fac))
    ax.set_ylim(lims[0], lims[1])

    ax.set_title(props['title'], fontsize=14)

    if data.name == 'EA_avg_AGR_AF':
        ax.set_xlabel('Time (core year of decadal-mean value)', fontsize=10)


def plot_map(opts, fig, ax, data):
    """
    plot map data
    Args:
        opts: CLI parameter
        fig: figure to plot on
        ax: axis to plot on
        data: data to plot

    Returns:

    """
    props = map_plot_params(vname=data.name)

    if data.name == 'EF_CC':
        rval, fac = 5, 1
    elif data.name == 'EM_avg_CC':
        rval, fac = 0.1, 0.05
    else:
        rval, fac = 0.5, 0.25
    lims = get_lims(data=data, rval=rval)
    lvls = np.arange(lims[0], lims[1] + lims[2], lims[2])
    map_vals = ax.contourf(data, cmap=props['cmap'], levels=lvls)

    ax.axis('off')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    cb = fig.colorbar(map_vals, cax=cax, orientation='vertical')
    cb.set_label(label=f'{opts.param_str}-{props["lbl"]}', fontsize=12)
    cb.ax.tick_params(labelsize=10)
    cb.ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    ax.set_title(props["title"], fontsize=14)


def plot_tex_es(opts, ax, data):
    """
    plot ES and TEX data
    Args:
        opts: CLI parameter
        ax: axis to plot on
        data: data to plot

    Returns:

    """
    xticks = np.arange(opts.start, opts.end + 1)

    ax.plot(xticks, data[f'ES_avg_AGR'], 'o-', color='tab:grey', markersize=3, linewidth=2)
    ax.plot(xticks, data[f'TEX_AGR'], 'o-', color='tab:red', markersize=3, linewidth=2)

    ax.set_ylabel(f'ES|TEX ' + r'$(S, T)$',
                  fontsize=12)
    ax.tick_params(axis='both', which='major', labelsize=10)
    ax.minorticks_on()
    ax.grid(color='gray', which='major', linestyle=':')
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))

    xn, xx = np.floor(opts.start / 5) * 5, np.ceil(opts.end / 5) * 5
    ax.set_xlim(xn, xx)
    ax.xaxis.set_minor_locator(FixedLocator(np.arange(xn, xx)))

    lims = get_lims(data[f'TEX_AGR'], rval=100)
    ax.set_yticks(np.arange(lims[0], lims[1] + lims[2] * 2, lims[2] * 2))
    ax.set_ylim(lims[0], lims[1])

    ax.set_title('Avg. Event Severity and Total Events Extremity', fontsize=14)
    ax.set_xlabel('Time (core year of decadal-mean value)', fontsize=10)


def plot_data(opts, data):
    """
    plot TEA indicators data
    Args:
        opts: CLI parameter
        data: data to plot

    Returns:

    """
    fig, axs = plt.subplots(4, 2, figsize=(14, 16))

    # plot GR data
    gr_vars = ['EF_AGR', 'ED_avg_AGR', 'EM_avg_AGR', 'EA_avg_AGR']
    for irow, gr_var in enumerate(gr_vars):
        plot_gr_data(opts=opts, ax=axs[irow, 0], data=data[gr_var])

    # plot maps
    map_vars = ['EF_CC', 'ED_avg_CC', 'EM_avg_CC']
    for irow, map_var in enumerate(map_vars):
        plot_map(opts=opts, fig=fig, ax=axs[irow, 1], data=data[map_var])

    # plot ES and TEX
    plot_tex_es(opts=opts, ax=axs[-1, -1], data=data)

    fig.subplots_adjust(wspace=0.2, hspace=0.33)

    # check and create output path
    outpath = f'{opts.outpath}/plots'
    if not os.path.exists(outpath):
        os.makedirs(outpath)

    plt.savefig(f'{outpath}/'
                f'TEA-indicator_data_{opts.region}_{opts.param_str}_{opts.dataset}.png',
                dpi=300, bbox_inches='tight')


def run():
    # get command line parameters
    cmd_opts = _getopts()

    # load CFG parameters
    opts = load_opts(fname=__file__, config_file=cmd_opts.config_file)

    # load dec and af data
    dec = load_data(opts=opts)

    # plot DECs
    plot_data(opts=opts, data=dec)


if __name__ == '__main__':
    run()
