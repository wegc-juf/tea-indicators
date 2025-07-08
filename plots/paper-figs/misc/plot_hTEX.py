"""
gki et al. 2024 (TEA)
@author: hst
"""

import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, MultipleLocator
import numpy as np
import os
import pandas as pd
import sys
import xarray as xr

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from teametrics.common.general_functions import ref_cc_params

PARAMS = ref_cc_params()


def load_data(ds):
    htex = pd.DataFrame(columns=['htEX', 'hTEX', 'htEX_AF', 'hTEX_AF'])
    cc_vals = {}

    data = xr.open_dataset(
        f'/data/users/hst/TEA-clean/TEA/misc_data/dec_indicator_variables/'
        f'DEC_Tx25.0degC_Niederösterreich_annual_{ds}_1961to2024.nc')
    data_af = xr.open_dataset(
        f'/data/users/hst/TEA-clean/TEA/misc_data/dec_indicator_variables/amplification/'
        f'AF_Tx25.0degC_Niederösterreich_annual_{ds}_1961to2024.nc')

    htex['htEX'] = data['htEX_GR']
    htex['hTEX'] = data['hTEX_GR']
    htex['htEX_AF'] = data_af['htEX_GR_AF']
    htex['hTEX_AF'] = data_af['hTEX_GR_AF']

    cc_vals['htEX_AF'] = data_af['htEX_GR_AF_CC'].values
    cc_vals['hTEX_AF'] = data_af['hTEX_GR_AF_CC'].values

    htex = htex.set_index(data.time.values)

    return htex, cc_vals


def plot_data(fig, axs, data, e5, thresh):
    """
    plot non normalized data
    :param fig: figure
    :param axs: axis
    :param data: data
    :param vvar: current axis index
    :param e5: ERA5 or ERA5Land
    :param thresh: threshold
    :return:
    """

    props = {'htEX': {'title': f'Hourly Temporal Events Extremity {e5}',
                      'ylbl': r'htEX (°C hours/yr)', 'var': 'htEX',
                      'unit': '°C hours / yr',
                      'refn': r'$\overline{htEX}_\mathrm{Ref}$',
                      'ccn': r'$\overline{htEX}_\mathrm{CC}$',
                      'ylims': [400, 1600],
                      'color': 'tab:orange',
                      'mjticks': 200,
                      'mnticks': 50},
             'hTEX': {'title': f'Hourly Total Events Extremity {e5}',
                      'ylbl': r'hTEX (areal °C hours/yr)', 'var': 'hTEX',
                      'unit': 'areal °C hours/yr',
                      'refn': r'$\overline{hTEX}_\mathrm{Ref}$',
                      'ccn': r'$\overline{hTEX}_\mathrm{CC}$',
                      'ylims': [40000, 240000],
                      'color': 'tab:red',
                      'mjticks': 40000,
                      'mnticks': 10000},
             }

    xticks = np.arange(1961, 2025)
    cc_vals = {'htEX': {}, 'hTEX': {}}

    for ivar, vvar in enumerate(['htEX', 'hTEX']):
        axs[ivar, 0].plot(xticks, data[vvar], 'o-', color=props[vvar]['color'], markersize=2)

        cc_vals[vvar]['ref'] = data[vvar][5:26].mean()
        cc_vals[vvar]['cc'] = data[vvar][-10:-4].mean()
        axs[ivar, 0].plot(xticks[:30], np.ones(30) * cc_vals[vvar]['ref'],
                          color=props[vvar]['color'],
                          linewidth=2)
        axs[ivar, 0].plot(xticks[-15:], np.ones(len(xticks[-15:])) * cc_vals[vvar]['cc'],
                          color=props[vvar]['color'],
                          linewidth=2)

        axs[ivar, 0].set_ylim(props[vvar]['ylims'][0], props[vvar]['ylims'][1])
        axs[ivar, 0].yaxis.set_major_locator(MultipleLocator(props[vvar]['mjticks']))
        axs[ivar, 0].yaxis.set_minor_locator(MultipleLocator(props[vvar]['mnticks']))
        axs[ivar, 0].set_ylabel(props[vvar]['ylbl'], fontsize=12)
        axs[ivar, 0].set_title(props[vvar]['title'], fontsize=14)

        axs[ivar, 0].set_xlim(1960, 2025)
        axs[ivar, 0].grid(color='lightgray', which='major', linestyle=':')
        axs[ivar, 0].xaxis.set_minor_locator(FixedLocator(np.arange(1960, 2025)))

        axs[ivar, 0].text(0.02, 0.86,
                          f'TMax-p99ANN-{props[vvar]["var"]}' + r'$_\mathrm{Ref | CC}$' + '\n'
                          + f'{cc_vals[vvar]["ref"]:.2f}' + r'$\,$|$\,$'
                          + f'{cc_vals[vvar]["cc"]:.2f} {props[vvar]["unit"]}',
                          horizontalalignment='left',
                          verticalalignment='center', transform=axs[ivar, 0].transAxes,
                          backgroundcolor='whitesmoke',
                          fontsize=9)

        ref_off, cc_off = 0.05, 0.05

        axs[ivar, 0].text(0.02,
                          ((cc_vals[vvar]['ref'] - props[vvar]['ylims'][0]) / (
                                  props[vvar]['ylims'][1] - props[vvar]['ylims'][0])) + ref_off,
                          props[vvar]["refn"],
                          horizontalalignment='left',
                          verticalalignment='center', transform=axs[ivar, 0].transAxes,
                          fontsize=10)

        axs[ivar, 0].text(0.77,
                          ((cc_vals[vvar]['cc'] - props[vvar]['ylims'][0]) / (
                                  props[vvar]['ylims'][1] - props[vvar]['ylims'][0])) + cc_off,
                          props[vvar]["ccn"],
                          horizontalalignment='left',
                          verticalalignment='center', transform=axs[ivar, 0].transAxes,
                          fontsize=10)


def plot_af(fig, axs, data, e5, cc_af):
    """
    plot amplification factors
    :param fig: figure
    :param axs: axis
    :param data: data
    :param e5: ERA5 or ERA5Land
    :return:
    """

    props = {'htEX_AF': {'title': f'htEX amplification',
                         'ylbl': 'htEX amplification ($\mathcal{A}^\mathrm{htEX}$)', 'var': 'htEX',
                         'unit': '1', 'boxname': r'$\mathcal{A}^\mathrm{htEX}_\mathrm{CC}$',
                         'refn': r'$\mathcal{A}_\mathrm{Ref}$',
                         'color': 'tab:orange',
                         'ylims': [0.5, 2.5],
                         'mjticks': 0.5,
                         'mnticks': 0.1
                         },
             'hTEX_AF': {'title': f'hTEX amplification',
                         'ylbl': r'hTEX amplification ($\mathcal{A}^\mathrm{hTEX}$)',
                         'var': 'hTEX',
                         'unit': '1', 'boxname': r'$\mathcal{A}^\mathrm{hTEX}_\mathrm{CC}$',
                         'refn': r'$\mathcal{A}_\mathrm{Ref}$',
                         'color': 'tab:red',
                         'ylims': [0.5, 3],
                         'mjticks': 0.5,
                         'mnticks': 0.1
                         }}

    xticks = np.arange(1961, 2025)

    for ivar, vvar in enumerate(['htEX_AF', 'hTEX_AF']):
        axs[ivar, 1].plot(xticks, data[vvar], 'o-', color=props[vvar]['color'], markersize=2)
        axs[ivar, 1].plot(xticks[:30], np.ones(30), color=props[vvar]['color'], linewidth=2)
        axs[ivar, 1].plot(xticks[-15:], np.ones(len(xticks[-15:])) * cc_af[vvar],
                          color=props[vvar]['color'],
                          linewidth=2)

        axs[ivar, 1].set_ylim(props[vvar]['ylims'][0], props[vvar]['ylims'][1])
        axs[ivar, 1].yaxis.set_major_locator(MultipleLocator(props[vvar]['mjticks']))
        axs[ivar, 1].yaxis.set_minor_locator(MultipleLocator(props[vvar]['mnticks']))

        axs[ivar, 1].set_ylabel(props[vvar]['ylbl'], fontsize=12)
        axs[ivar, 1].set_title(props[vvar]['title'], fontsize=14)

        axs[ivar, 1].set_xlim(1960, 2025)
        axs[ivar, 1].grid(color='lightgray', which='major', linestyle=':')
        axs[ivar, 1].xaxis.set_minor_locator(FixedLocator(np.arange(1960, 2025)))

        axs[ivar, 1].text(0.02, 0.87, f'TMax-p99ANN-{props[vvar]["boxname"]}\n'
                          + f'{cc_af[vvar]:.2f}',
                          horizontalalignment='left',
                          verticalalignment='center', transform=axs[ivar, 1].transAxes,
                          backgroundcolor='whitesmoke',
                          fontsize=9)

        ref_off, cc_off = 0.05, 0.06

        axs[ivar, 1].text(0.02, ((1 - props[vvar]['ylims'][0]) / (
                    props[vvar]['ylims'][1] - props[vvar]['ylims'][0])) + ref_off,
                          props[vvar]["refn"],
                          horizontalalignment='left',
                          verticalalignment='center', transform=axs[ivar, 1].transAxes,
                          fontsize=10)

        axs[ivar, 1].text(0.79, ((cc_af[vvar] - props[vvar]['ylims'][0]) / (
                    props[vvar]['ylims'][1] - props[vvar]['ylims'][0])) + cc_off,
                          props[vvar]["boxname"],
                          horizontalalignment='left',
                          verticalalignment='center', transform=axs[ivar, 1].transAxes,
                          fontsize=10)


def run():
    ds = 'ERA5'

    data, af_cc = load_data(ds=ds)

    fig, axs = plt.subplots(2, 2, figsize=(12, 6))

    plot_data(fig=fig, axs=axs, data=data, e5=ds, thresh=25.0)
    plot_af(fig=fig, axs=axs, data=data, e5=ds, cc_af=af_cc)

    # iterate over each subplot and add a text label
    labels = ['a', 'b', 'c', 'd']
    for i, ax in enumerate(axs.flat):
        ax.text(-0.1, 1.2, labels[i], transform=ax.transAxes, fontsize=14, fontweight='bold',
                va='top', ha='left')

    fig.subplots_adjust(left=0.1, right=0.9, bottom=0.1, top=0.9, wspace=0.2, hspace=0.35)

    outpath = f'/nas/home/hst/work/TEAclean/plots/misc/hTEX_Tx25_{ds}_L-AUT.png'
    plt.savefig(outpath, bbox_inches='tight', dpi=300)


if __name__ == '__main__':
    run()
