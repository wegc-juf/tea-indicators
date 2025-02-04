import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, ScalarFormatter, FixedFormatter
import numpy as np
import pandas as pd
import xarray as xr


def get_data(reg, ds, thresh):

    pstr = f'Tx{thresh}.0degC'
    reg_str, gr_str = reg, 'GR'

    data = xr.open_dataset(f'/data/users/hst/TEA-clean/TEA_no-largeGR/amplification/'
                           f'AF_{pstr}_{reg_str}_WAS_{ds}_1961to2024.nc')

    if 'ERA5' in ds:
        tEX_gr = 10 ** (np.log10(data[f'EF_{gr_str}_AF'])
                        + np.log10(data[f'EDavg_{gr_str}_AF'])
                        + np.log10(data[f'EMavg_{gr_str}_AF']))
        tEX_gr_cc = 10 ** (np.log10(data[f'EF_{gr_str}_AF_CC'])
                           + np.log10(data[f'EDavg_{gr_str}_AF_CC'])
                           + np.log10(data[f'EMavg_{gr_str}_AF_CC']))
        data[f'tEX_{gr_str}_AF'] = tEX_gr
        data[f'tEX_{gr_str}_AF_CC'] = tEX_gr_cc

    tEX = 10 ** (np.log10(data[f'EF_AF'])
                 + np.log10(data[f'EDavg_AF'])
                 + np.log10(data['EMavg_AF']))
    tEX_cc = 10 ** (np.log10(data[f'EF_AF_CC'])
                    + np.log10(data[f'EDavg_AF_CC'])
                    + np.log10(data['EMavg_AF_CC']))
    data[f'tEX_AF'] = tEX
    data[f'tEX_AF_CC'] = tEX_cc

    fd_gr = 10 ** (np.log10(data[f'EF_{gr_str}_AF'])
                   + np.log10(data[f'EDavg_{gr_str}_AF']))
    fd_gr_cc = 10 ** (np.log10(data[f'EF_{gr_str}_AF_CC'])
                      + np.log10(data[f'EDavg_{gr_str}_AF_CC']))
    data[f'FD_{gr_str}_AF'] = fd_gr
    data[f'FD_{gr_str}_AF_CC'] = fd_gr_cc

    return data


def plot_subplot(ax, spcus, era5, var, reg, land, thresh):

    cols = {'EF': 'tab:blue', 'FD': 'tab:purple', 'tEX': 'tab:orange', 'TEX': 'tab:red'}
    plot_vars = ['EF', 'FD', 'tEX', 'TEX']

    if thresh == 30:
        ymin, ymax = 2 * 10 ** -1, 30
    else:
        ymin, ymax = 5 * 10 ** -1, 5

    gr_str = 'GR'

    xticks = np.arange(1961, 2025)

    pstr = f'Tx{thresh}.0degC'
    nv_var = 'TEX'

    rstr = 'AUT'
    if reg != 'AUT':
        rstr = 'SEA'

    if reg != 'L-AUT':
        nv = pd.read_csv(f'/data/users/hst/TEA-clean/TEA/natural_variability/'
                         f'NV_AF_{pstr}_{rstr}.csv',
                         index_col=0)

        nat_var_low = np.ones(len(xticks)) * (1 - nv.loc[nv_var, 'lower'] * 1.645)
        nat_var_upp = np.ones(len(xticks)) * (1 + nv.loc[nv_var, 'upper'] * 1.645)
        ax.fill_between(x=xticks, y1=nat_var_low, y2=nat_var_upp, color=cols[nv_var], alpha=0.2)

    acc = 100
    for ivar, pvar in enumerate(plot_vars):
        ax.plot(xticks, era5[f'{pvar}_{gr_str}_AF'], '--', color=cols[pvar], linewidth=1.5,
                alpha=0.5)
        ax.plot(xticks, spcus[f'{pvar}_GR_AF'], color=cols[pvar], linewidth=2, markersize=3)
        ax.plot(xticks[49:], np.ones(len(xticks[49:])) * spcus[f'{pvar}_GR_AF_CC'].values,
                color=cols[pvar], linewidth=2)
        ax.plot(xticks[49:], np.ones(len(xticks[49:])) * era5[f'{pvar}_{gr_str}_AF_CC'].values,
                '--',
                alpha=0.5, color=cols[pvar], linewidth=2)
        if spcus[f'{pvar}_GR_AF_CC'] < acc:
            acc = spcus[f'{pvar}_GR_AF_CC'].values

    ax.plot(xticks[0:30], np.ones(len(xticks[0:30])), alpha=0.5, color='tab:red', linewidth=2)

    ax.set_yscale('log')
    ax.minorticks_on()
    ax.grid(color='gray', which='major', linestyle=':')
    if thresh == 30:
        ax.yaxis.set_minor_formatter(FixedFormatter(['0.2', '', '', '0.5', '', '', '', '',
                                                     '2.0', '', '', '5.0', '', '', '', '', '',
                                                     '20.0', '30.0']))
        ax.yaxis.set_minor_locator(FixedLocator([0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,
                                                 2, 3, 4, 5, 6, 7, 8, 9, 15, 20, 30]))
    else:
        ax.yaxis.set_minor_formatter(FixedFormatter(['0.5', '', '', '', '',
                                                     '2.0', '', '', '5.0']))
        ax.yaxis.set_minor_locator(FixedLocator([0.5, 0.6, 0.7, 0.8, 0.9,
                                                 2, 3, 4, 5]))
    e5 = 'E5'
    if land:
        e5 = 'E5L'

    ax.tick_params(axis='both', which='major', labelsize=10)
    ax.set_title(f'Extremity amplification {reg} | {var}', fontsize=14)
    ax.set_ylim(ymin, ymax)
    ax.set_xlim(1960, 2025)
    ax.xaxis.set_minor_locator(FixedLocator(np.arange(1960, 2025)))
    ax.yaxis.set_major_formatter(ScalarFormatter())

    ax.set_ylabel(
        'F' + r'$\,$|$\,$' + 'FD' + r'$\,$|$\,$' + 'tEX' + r'$\,$|$\,$' + 'TEX amplification',
        fontsize=10)
    xpos, ypos = 0.02, 0.26
    if reg != 'FBR':
        off = 0.31
    else:
        off = 0.34
    xpos_cc, ypos_cc = 0.83, ((acc - ymin) / (ymax - ymin)) + off,
    cc_name = r'$\mathcal{A}_\mathrm{CC}^\mathrm{F, FD, t, T}$'

    if thresh == 25:
        ypos_cc = ypos_cc - 0.23

    box_txt = ((('SPCUS-TMax-p99ANN-' + r'$\mathcal{A}_\mathrm{CC}^\mathrm{T}$ = '
                 + f'{np.round(spcus["TEX_GR_AF_CC"], 2):.2f}\n')
                + f'{e5}-TMax-p99ANN-' + r'$\mathcal{A}_\mathrm{CC}^\mathrm{T}$ = ')
               + f'{np.round(era5[f"TEX_{gr_str}_AF_CC"], 2):.2f}')

    ax.text(xpos, ypos, r'$\mathcal{A}_\mathrm{Ref}$',
            horizontalalignment='left',
            verticalalignment='center', transform=ax.transAxes,
            fontsize=11)
    ax.text(xpos_cc, ypos_cc, cc_name,
            horizontalalignment='left',
            verticalalignment='center', transform=ax.transAxes,
            fontsize=11)

    ypos_box = 0.86
    if reg == 'L-AUT':
        ypos_box = 0.9

    ax.text(0.02, ypos_box, box_txt,
            horizontalalignment='left', verticalalignment='center', transform=ax.transAxes,
            backgroundcolor='whitesmoke', fontsize=9)


def create_legend(fig, ax):
    f, = ax.plot([-9, -9], 'tab:blue', linewidth=2)
    fd, = ax.plot([-9, -9], 'tab:purple', linewidth=2)
    fdm, = ax.plot([-9, -9], 'tab:orange', linewidth=2)
    fdma, = ax.plot([-9, -9], color='tab:red', linewidth=2)
    spar, = ax.plot([-9, -9], color='tab:gray', linewidth=2)
    era5, = ax.plot([-9, -9], color='tab:gray', linestyle='--', alpha=0.5, linewidth=2)


    fig.legend((spar, era5, f, fd, fdm, fdma),
               ('SPCUS', 'ERA5(L)', r'$\mathcal{A}^\mathrm{F}$', r'$\mathcal{A}^\mathrm{FD}$',
                r'$\mathcal{A}^\mathrm{t}$', r'$\mathcal{A}^\mathrm{T}$'),
               ncol=6, loc=(0.27, 0.01))


def run():
    threshold = 25
    era5_land = [True, False]
    regions = ['AUT', 'SEA', 'FBR']

    fig, axs = plt.subplots(3, 2, figsize=(12, 10))

    for icol, dsets in enumerate(era5_land):
        e5_ds = 'ERA5'
        if dsets:
            e5_ds = f'{e5_ds}Land'
        for irow, reg in enumerate(regions):
            e5_data = get_data(reg=reg, ds=e5_ds, thresh=threshold)
            sp_data = get_data(reg=reg, ds='SPARTACUS', thresh=threshold)
            plot_subplot(ax=axs[irow, icol], spcus=sp_data, era5=e5_data, var=e5_ds, reg=reg,
                         land=dsets, thresh=threshold)

    axs[2, 0].set_xlabel('Time (core year of decadal-mean value)', fontsize=12)
    axs[2, 1].set_xlabel('Time (core year of decadal-mean value)', fontsize=12)

    fig.subplots_adjust(left=0.1, right=0.9, bottom=0.1, top=0.9, wspace=0.2, hspace=0.35)

    create_legend(fig=fig, ax=axs[0, 0])

    plt.savefig(f'/nas/home/hst/work/TEAclean/plots/misc/Fig3-EDF5/'
                f'Fig3-EDF5_{threshold}degC.png', dpi=300, bbox_inches='tight')


def run_noe():
    era5_land = [True, False]

    fig, axs = plt.subplots(1, 2, figsize=(12, 4))

    for icol, dsets in enumerate(era5_land):
        e5_ds = 'ERA5'
        if dsets:
            e5_ds = f'{e5_ds}Land'

        e5_data = get_data(reg='Niederösterreich', ds=e5_ds, thresh=25)
        sp_data = get_data(reg='Niederösterreich', ds='SPARTACUS', thresh=25)
        plot_subplot(ax=axs[icol], spcus=sp_data, era5=e5_data, var=e5_ds, reg='L-AUT',
                     land=dsets, thresh=25)

        axs[icol].set_xlabel('Time (core year of decadal-mean value)', fontsize=12)

    fig.subplots_adjust(left=0.1, right=0.9, bottom=0.2, top=0.9, wspace=0.2, hspace=0.35)

    create_legend(fig=fig, ax=axs[0])

    plt.savefig(f'/nas/home/hst/work/TEAclean/plots/misc/Fig3-EDF5/'
                f'Fig3-EDF5_Niederösterreich_25degC.png', dpi=300, bbox_inches='tight')

if __name__ == '__main__':
    # run()
    run_noe()
