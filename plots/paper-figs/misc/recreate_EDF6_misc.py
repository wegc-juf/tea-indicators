#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
gki et al. 2024 (TEA) ExtDataFig. 6 plot
@author: hst
"""

import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, MultipleLocator
import numpy as np
import pandas as pd
import xarray as xr

from teametrics.common.general_functions import ref_cc_params

PARAMS = ref_cc_params()


def load_aep_data(ds, thresh, noe=False):
    if noe:
        aep = pd.DataFrame(columns=['L-AUT'])
        aep_af = pd.DataFrame(columns=['L-AUT'])
        data = xr.open_dataset(
            f'/data/users/hst/TEA/TEA/misc_data/dec_indicator_variables/'
            f'DEC_Tx{thresh}.0degC_Niederösterreich_annual_{ds}_1961to2024.nc')
        data_af = xr.open_dataset(
            f'/data/users/hst/TEA/TEA/misc_data/dec_indicator_variables/amplification/'
            f'AF_Tx{thresh}.0degC_Niederösterreich_annual_{ds}_1961to2024.nc')

        aep['L-AUT'] = data['AEP_GR']
        aep_af['L-AUT'] = data_af['AEP_GR_AF']
    else:
        regs = ['AUT', 'SEA', 'FBR']
        aep = pd.DataFrame(columns=regs)
        aep_af = pd.DataFrame(columns=regs)
        for reg in regs:
            data = xr.open_dataset(
                f'/data/users/hst/TEA/TEA/misc_data/dec_indicator_variables/'
                f'DEC_Tx{thresh}.0degC_{reg}_annual_{ds}_1961to2024.nc')
            data_af = xr.open_dataset(
                f'/data/users/hst/TEA/TEA/misc_data/dec_indicator_variables/amplification/'
                f'AF_Tx{thresh}.0degC_{reg}_annual_{ds}_1961to2024.nc')

            aep[reg] = data['AEP_GR']
            aep_af[reg] = data_af['AEP_GR_AF']

    return aep, aep_af


def load_det_data(ds, thresh, noe=False):
    """
    load ERA5 energy content data and calc decadal mean
    :param ds: ERA5 or ERA5Land
    :param thresh: threshold
    :param noe: set if L-AUT data is used
    :return: det (df of daily exposure time)
    """

    if not noe:
        regs = ['AUT', 'SEA', 'FBR']
        det = pd.DataFrame(columns=regs)
        det_af = pd.DataFrame(columns=regs)
        for reg in regs:
            data = xr.open_dataset(
                f'/data/users/hst/TEA/TEA/misc_data/dec_indicator_variables/'
                f'DEC_Tx{thresh}.0degC_{reg}_annual_{ds}_1961to2024.nc')
            data_af = xr.open_dataset(
                f'/data/users/hst/TEA/TEA/misc_data/dec_indicator_variables/amplification/'
                f'AF_Tx{thresh}.0degC_{reg}_annual_{ds}_1961to2024.nc')

            det[reg] = data['h_avg_GR']
            det_af[reg] = data_af['h_avg_GR_AF']
    else:
        det = pd.DataFrame(columns=['L-AUT'])
        det_af = pd.DataFrame(columns=['L-AUT'])
        data = xr.open_dataset(
            f'/data/users/hst/TEA/TEA/misc_data/dec_indicator_variables/'
            f'DEC_Tx{thresh}.0degC_Niederösterreich_annual_{ds}_1961to2024.nc')
        data_af = xr.open_dataset(
            f'/data/users/hst/TEA/TEA/misc_data/dec_indicator_variables/amplification/'
            f'AF_Tx{thresh}.0degC_Niederösterreich_annual_{ds}_1961to2024.nc')

        det['L-AUT'] = data['h_avg_GR']
        det_af['L-AUT'] = data_af['h_avg_GR_AF']

    return det, det_af


def ylims_det_aep(e5, thresh):
    if e5 == 'ERA5Land' and thresh == 25:
        props = {0: {'ymax': 8, 'ymin': 4},
                 2: {'ymax': 6, 'ymin': 2},
                 4: {'ymax': 7, 'ymin': 3}}
    elif e5 == 'ERA5Land' and thresh == 30:
        props = {0: {'ymax': 6, 'ymin': 1},
                 2: {'ymax': 3, 'ymin': 0},
                 4: {'ymax': 5, 'ymin': 0}}
    elif e5 == 'ERA5' and thresh == 25:
        props = {0: {'ymax': 8, 'ymin': 4},
                 2: {'ymax': 6, 'ymin': 3},
                 4: {'ymax': 7, 'ymin': 3}}
    elif e5 == 'ERA5' and thresh == 30:
        props = {0: {'ymax': 6, 'ymin': 1},
                 2: {'ymax': 4, 'ymin': 0},
                 4: {'ymax': 5, 'ymin': 0}}
    else:
        props = {0: {'ymax': 7, 'ymin': 4},
                 2: {'ymax': 6, 'ymin': 3},
                 4: {'ymax': 7, 'ymin': 3}}

    return props


def plot_det_aep(fig, axs, data, nax, e5, thresh, noe=False):
    """
    plot non normalized data
    :param fig: figure
    :param axs: axis
    :param data: data
    :param nax: current axis index
    :param e5: ERA5 or ERA5Land
    :param thresh: threshold
    :param noe: set if Lower-AUT data is plotted
    :return:
    """

    props = {0: {'title': f'Avg. Daily Exposure Time (DET) {e5} | Heat',
                 'ylbl': r'DET ($\overline{h}_s$) (h/day)', 'var': 'DET',
                 'unit': 'h/day',
                 'refn': r'$\overline{h}_\mathrm{Ref}$', 'ccn': r'$\overline{h}_\mathrm{CC}$'},
             2: {'title': f'Annual Exposure Period (AEP) {e5} | Heat',
                 'ylbl': r'AEP ($\Delta Y_s$) (months)', 'var': 'AEP',
                 'unit': 'months',
                 'refn': r'$\Delta Y_\mathrm{Ref}$', 'ccn': r'$\Delta Y_\mathrm{CC}$'},
             4: {'title': 'Annual Exposure Period (AEP) SPCUS | Heat',
                 'ylbl': r'AEP ($\Delta Y_s$) (months)', 'var': 'AEP', 'unit': 'months',
                 'refn': r'$\Delta Y_\mathrm{Ref}$', 'ccn': r'$\Delta Y_\mathrm{CC}$'}}
    ylims = ylims_det_aep(e5=e5, thresh=thresh)

    colors = ['#C5283D', '#E9724C', '#FFC857']
    xticks = np.arange(1961, 2025)
    cc_vals = {}
    regions = ['AUT', 'SEA', 'FBR']
    if noe:
        regions = ['L-AUT']
        colors = ['tab:purple']
    for ireg, reg in enumerate(regions):
        cc_vals[reg] = {}
        axs.plot(xticks, data[reg], 'o-', color=colors[ireg], markersize=2)
        cc_vals[reg]['ref'] = data[reg][5:26].mean()
        cc_vals[reg]['cc'] = data[reg][-10:-4].mean()
        axs.plot(xticks[:30], np.ones(30) * cc_vals[reg]['ref'], color=colors[ireg],
                 linewidth=2)
        axs.plot(xticks[-15:], np.ones(len(xticks[-15:])) * cc_vals[reg]['cc'], color=colors[ireg],
                 linewidth=2)

    axs.set_ylim(ylims[nax]['ymin'], ylims[nax]['ymax'])
    mjticks, mnticks = 1, 0.25
    if ylims[nax]['ymax'] - ylims[nax]['ymin'] >= 10:
        mjticks, mnticks = 2, 0.5
    axs.yaxis.set_major_locator(MultipleLocator(mjticks))
    axs.yaxis.set_minor_locator(MultipleLocator(mnticks))
    axs.set_ylabel(props[nax]['ylbl'], fontsize=12)

    axs.set_title(props[nax]['title'], fontsize=14)

    if not noe:
        largest_reg = 'AUT'
        axs.text(0.02, 0.8, f'TMax-p99ANN-{props[nax]["var"]}' + r'$_\mathrm{Ref | CC}$' + '\n'
                 + f'AUT: {cc_vals["AUT"]["ref"]:.2f}' + r'$\,$|$\,$'
                 + f'{cc_vals["AUT"]["cc"]:.2f} {props[nax]["unit"]} \n'
                 + f'SEA: {cc_vals["SEA"]["ref"]:.2f}' + r'$\,$|$\,$'
                 + f'{cc_vals["SEA"]["cc"]:.2f} {props[nax]["unit"]} \n'
                 + f'FBR: {cc_vals["FBR"]["ref"]:.2f}' + r'$\,$|$\,$'
                 + f'{cc_vals["FBR"]["cc"]:.2f} {props[nax]["unit"]}',
                 horizontalalignment='left',
                 verticalalignment='center', transform=axs.transAxes, backgroundcolor='whitesmoke',
                 fontsize=9)
    else:
        largest_reg = 'L-AUT'
        axs.text(0.02, 0.87, f'TMax-p99ANN-{props[nax]["var"]}' + r'$_\mathrm{Ref | CC}$' + '\n'
                 + f'L-AUT: {cc_vals["L-AUT"]["ref"]:.2f}' + r'$\,$|$\,$'
                 + f'{cc_vals["L-AUT"]["cc"]:.2f} {props[nax]["unit"]}',
                 horizontalalignment='left',
                 verticalalignment='center', transform=axs.transAxes, backgroundcolor='whitesmoke',
                 fontsize=9)

    ref_off, cc_off = 0.05, 0.05
    if nax == 0:
        cc_off = 0.23

    if noe:
        cc_off = 0.05

    axs.text(0.02,
             ((cc_vals[largest_reg]['ref'] - ylims[nax]['ymin']) / (
                     ylims[nax]['ymax'] - ylims[nax]['ymin'])) + ref_off, props[nax]["refn"],
             horizontalalignment='left',
             verticalalignment='center', transform=axs.transAxes,
             fontsize=10)

    axs.text(0.92,
             ((cc_vals[largest_reg]['cc'] - ylims[nax]['ymin']) / (
                     ylims[nax]['ymax'] - ylims[nax]['ymin'])) + cc_off, props[nax]["ccn"],
             horizontalalignment='left',
             verticalalignment='center', transform=axs.transAxes,
             fontsize=10)


def ylims_af(e5, thresh):
    if e5 == 'ERA5Land' and thresh == 25:
        lims = {1: {'yn': 0.75, 'yx': 1.5, 'dmaj': 0.25, 'dmin': 0.125},
                3: {'yn': 0.75, 'yx': 1.5, 'dmaj': 0.25, 'dmin': 0.125},
                5: {'yn': 0.75, 'yx': 1.5, 'dmaj': 0.25, 'dmin': 0.125}}
    elif e5 == 'ERA5Land' and thresh == 30:
        lims = {1: {'yn': 0.5, 'yx': 2.5, 'dmaj': 0.5, 'dmin': 0.25},
                3: {'yn': 0, 'yx': 9, 'dmaj': 1, 'dmin': 0.5},
                5: {'yn': 0, 'yx': 4, 'dmaj': 1, 'dmin': 0.25}}
    elif e5 == 'ERA5' and thresh == 25:
        lims = {1: {'yn': 0.75, 'yx': 1.5, 'dmaj': 0.25, 'dmin': 0.125},
                3: {'yn': 0.75, 'yx': 1.5, 'dmaj': 0.25, 'dmin': 0.125},
                5: {'yn': 0.75, 'yx': 1.5, 'dmaj': 0.25, 'dmin': 0.125}}
    elif e5 == 'ERA5' and thresh == 30:
        lims = {1: {'yn': 0.5, 'yx': 1.75, 'dmaj': 0.25, 'dmin': 0.125},
                3: {'yn': 0.5, 'yx': 3.5, 'dmaj': 0.5, 'dmin': 0.25},
                5: {'yn': 0, 'yx': 3.5, 'dmaj': 0.5, 'dmin': 0.25}}
    else:  # L-AUT 25°
        lims = {1: {'yn': 0.75, 'yx': 1.5, 'dmaj': 0.25, 'dmin': 0.125},
                3: {'yn': 0.75, 'yx': 1.5, 'dmaj': 0.25, 'dmin': 0.125},
                5: {'yn': 0.75, 'yx': 1.5, 'dmaj': 0.25, 'dmin': 0.125}}

    return lims


def plot_af(fig, axs, data, nax, e5, thresh, noe=False):
    """
    plot amplification factors
    :param fig: figure
    :param axs: axis
    :param data: data
    :param nax: number of axis
    :param e5: ERA5 or ERA5Land
    :param thresh: threshold
    :param noe: set if L-AUT data is plotted
    :return:
    """

    props = {1: {'title': f'DET amplification {e5} | Heat',
                 'ylbl': 'DET amplification ($\mathcal{A}^\mathrm{h}$)', 'var': 'h',
                 'unit': 'h', 'boxname': r'$\mathcal{A}^\mathrm{h}_\mathrm{CC}$',
                 'refn': r'$\mathcal{A}_\mathrm{Ref}$'},
             3: {'title': f'AEP amplification {e5} | Heat',
                 'ylbl': r'AEP amplification ($\mathcal{A}^\mathrm{\Delta Y}$)',
                 'var': r'$\Delta$Y',
                 'unit': 'month', 'boxname': r'$\mathcal{A}^\mathrm{\Delta Y}_\mathrm{CC}$',
                 'refn': r'$\mathcal{A}_\mathrm{Ref}$'},
             5: {'title': 'AEP amplification SPCUS | Heat',
                 'ylbl': r'AEP amplification ($\mathcal{A}^\mathrm{\Delta Y}$)',
                 'var': r'$\Delta$Y', 'unit': 'month',
                 'boxname': r'$\mathcal{A}^\mathrm{\Delta Y}_\mathrm{CC}$',
                 'refn': r'$\mathcal{A}_\mathrm{Ref}$'}}

    colors = ['#C5283D', '#E9724C', '#FFC857']
    xticks = np.arange(1961, 2025)

    cc_vals = {}
    lims = ylims_af(e5=e5, thresh=thresh)
    yn, yx = lims[nax]['yn'], lims[nax]['yx']
    dmaj, dmin = lims[nax]['dmaj'], lims[nax]['dmin']
    regions = ['AUT', 'SEA', 'FBR']
    if noe:
        regions = ['L-AUT']
        colors = ['tab:purple']
    for ireg, reg in enumerate(regions):
        # calc af
        ref = data[reg][5:26].mean()
        cc = data[reg][-10:-4].mean()
        cc_vals[reg] = cc / ref

        axs.plot(xticks, data[reg], 'o-', color=colors[ireg], markersize=2)
        if reg == 'AUT':
            axs.plot(xticks[:30], np.ones(30), color='k', linewidth=2)
        if reg == 'L-AUT':
            axs.plot(xticks[:30], np.ones(30), color=colors[ireg], linewidth=2)
        axs.plot(xticks[-15:], np.ones(len(xticks[-15:])) * cc_vals[reg], color=colors[ireg],
                 linewidth=2)

        axs.set_ylim(yn, yx)
        axs.yaxis.set_major_locator(MultipleLocator(dmaj))
        axs.yaxis.set_minor_locator(MultipleLocator(dmin))

    axs.set_ylabel(props[nax]['ylbl'], fontsize=12)

    axs.set_title(props[nax]['title'], fontsize=14)

    if not noe:
        axs.text(0.02, 0.8, f'TMax-p99ANN-{props[nax]["boxname"]}\n'
                 + f'AUT: {cc_vals["AUT"]:.2f} \n'
                 + f'SEA: {cc_vals["SEA"]:.2f} \n'
                 + f'FBR: {cc_vals["FBR"]:.2f}',
                 horizontalalignment='left',
                 verticalalignment='center', transform=axs.transAxes, backgroundcolor='whitesmoke',
                 fontsize=9)
        smallest_region = 'FBR'
    else:
        axs.text(0.02, 0.87, f'TMax-p99ANN-{props[nax]["boxname"]}\n'
                 + f'L-AUT: {cc_vals["L-AUT"]:.2f}',
                 horizontalalignment='left',
                 verticalalignment='center', transform=axs.transAxes, backgroundcolor='whitesmoke',
                 fontsize=9)
        smallest_region = 'L-AUT'

    ref_off, cc_off = 0.07, 0.06
    if nax == 3:
        ref_off, cc_off = 0.05, 0.09

    axs.text(0.02, ((1 - yn) / (yx - yn)) + ref_off, props[nax]["refn"],
             horizontalalignment='left',
             verticalalignment='center', transform=axs.transAxes,
             fontsize=10)

    axs.text(0.92, ((cc_vals[smallest_region] - yn) / (yx - yn)) + cc_off, props[nax]["boxname"],
             horizontalalignment='left',
             verticalalignment='center', transform=axs.transAxes,
             fontsize=10)


def run():
    e5_ds = 'ERA5Land'
    threshold = 30

    # load SPARTACUS data
    aep_spcus, aep_af_spcus = load_aep_data(ds='SPARTACUS', thresh=threshold)

    # load ERA5(Land) data
    aep_era5, aep_af_era5 = load_aep_data(ds=e5_ds, thresh=threshold)
    e5_det, e5_det_af = load_det_data(ds=e5_ds, thresh=threshold)

    data = {0: e5_det, 1: e5_det_af, 2: aep_era5, 3: aep_af_era5, 4: aep_spcus, 5: aep_af_spcus}

    fig, axs = plt.subplots(3, 2, figsize=(12, 10))
    axs = axs.reshape(-1)

    for iax, ax in enumerate(axs):
        if iax % 2 == 0:
            plot_det_aep(fig=fig, axs=ax, data=data[iax], nax=iax, e5=e5_ds, thresh=threshold)
        else:
            plot_af(fig=fig, axs=ax, data=data[iax], nax=iax, e5=e5_ds, thresh=threshold)

        if iax in [4, 5]:
            ax.set_xlabel('Time (core year of decadal-mean value)')

        # general plot props
        ax.set_xlim(1960, 2025)
        ax.grid(color='lightgray', which='major', linestyle=':')
        ax.xaxis.set_minor_locator(FixedLocator(np.arange(1960, 2025)))

    colors = ['#C5283D', '#E9724C', '#FFC857']
    aut, = axs[0].plot([-9, -9], 'o-', color=colors[0], markersize=2)
    sea, = axs[0].plot([-9, -9], 'o-', color=colors[1], markersize=2)
    fbr, = axs[0].plot([-9, -9], 'o-', color=colors[2], markersize=2)

    fig.legend((aut, sea, fbr), ('AUT', 'SEA', 'FBR'), ncols=3, loc=(0.37, 0.01))
    fig.subplots_adjust(hspace=0.4, wspace=0.25)

    # iterate over each subplot and add a text label
    labels = ['a', 'b', 'c', 'd', 'e', 'f']
    for i, ax in enumerate(axs.flat):
        ax.text(-0.1, 1.2, labels[i], transform=ax.transAxes, fontsize=14, fontweight='bold',
                va='top', ha='left')

    fig.subplots_adjust(left=0.1, right=0.9, bottom=0.1, top=0.9, wspace=0.2, hspace=0.35)

    outpath = f'./EDF6_Tx{threshold}_{e5_ds}.png'
    plt.savefig(outpath, bbox_inches='tight', dpi=300)

    print(f'plot saved to: {outpath}')


def run_noe():
    e5_ds = 'ERA5Land'

    # load SPARTACUS data
    aep_spcus, aep_af_spcus = load_aep_data(ds='SPARTACUS', thresh=25, noe=True)

    # load ERA5(Land) data
    aep_era5, aep_af_era5 = load_aep_data(ds=e5_ds, thresh=25, noe=True)
    e5_det, e5_det_af = load_det_data(ds=e5_ds, thresh=25, noe=True)

    data = {0: e5_det, 1: e5_det_af, 2: aep_era5, 3: aep_af_era5, 4: aep_spcus, 5: aep_af_spcus}

    fig, axs = plt.subplots(3, 2, figsize=(12, 10))
    axs = axs.reshape(-1)

    for iax, ax in enumerate(axs):
        if iax % 2 == 0:
            plot_det_aep(fig=fig, axs=ax, data=data[iax], nax=iax, e5=e5_ds, thresh=25, noe=True)
        else:
            plot_af(fig=fig, axs=ax, data=data[iax], nax=iax, e5=e5_ds, thresh=25, noe=True)

        if iax in [4, 5]:
            ax.set_xlabel('Time (core year of decadal-mean value)')

        # general plot props
        ax.set_xlim(1960, 2025)
        ax.grid(color='lightgray', which='major', linestyle=':')
        ax.xaxis.set_minor_locator(FixedLocator(np.arange(1960, 2025)))

    fig.subplots_adjust(hspace=0.4, wspace=0.25)

    # iterate over each subplot and add a text label
    labels = ['a', 'b', 'c', 'd', 'e', 'f']
    for i, ax in enumerate(axs.flat):
        ax.text(-0.1, 1.2, labels[i], transform=ax.transAxes, fontsize=14, fontweight='bold',
                va='top', ha='left')

    fig.subplots_adjust(left=0.1, right=0.9, bottom=0.1, top=0.9, wspace=0.2, hspace=0.35)

    outpath = f'./EDF6_Tx25_{e5_ds}_L-AUT_TESTY.png'
    plt.savefig(outpath, bbox_inches='tight', dpi=300)


if __name__ == '__main__':
    run()
    run_noe()
