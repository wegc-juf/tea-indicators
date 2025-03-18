import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as pat
from matplotlib.ticker import FixedLocator, ScalarFormatter, FixedFormatter
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
import pandas as pd
import pyproj
import xarray as xr


def get_data(reg, var, ds):
    if var == 'Temperature':
        pstr = 'Tx99.0p'
    else:
        pstr = 'P24h_7to7_95.0p'

    reg_str, gr_str = reg, 'GR'
    if 'ERA5' in ds and reg != 'FBR':
        reg_str = f'AGR-{reg}'
        gr_str = 'AGR'

    data = xr.open_dataset(f'/data/users/hst/TEA-clean/TEA/paper_data/dec_indicator_variables/'
                           f'amplification/AF_{pstr}_{reg_str}_WAS_{ds}_1961to2024.nc')

    fd_gr = 10 ** (np.log10(data[f'EF_{gr_str}_AF'])
                   + np.log10(data[f'ED_avg_{gr_str}_AF']))
    fd_gr_cc = 10 ** (np.log10(data[f'EF_{gr_str}_AF_CC'])
                      + np.log10(data[f'ED_avg_{gr_str}_AF_CC']))
    data[f'FD_{gr_str}_AF'] = fd_gr
    data[f'FD_{gr_str}_AF_CC'] = fd_gr_cc

    return data


def regrid_era5(ds, grid):
    """
    regrid ERA5 data to SPARTACUS grid
    :param ds: ERA5 data
    :param grid: destination grid
    :return: ds_reg
    """

    x_spar = grid.x.values
    y_spar = grid.y.values

    projection = pyproj.Proj('EPSG:32633')
    lon, lat = projection(*np.meshgrid(x_spar, y_spar), inverse=True)

    x = xr.DataArray(lon, dims=["y", "x"], coords={"x": x_spar, "y": y_spar})
    y = xr.DataArray(lat, dims=["y", "x"], coords={"x": x_spar, "y": y_spar})

    # interpolation
    ds_reg = ds.interp(lon=x, lat=y, method='linear')

    return ds_reg


def plot_maps(fig, spcus, era5, land):
    if land:
        e5 = 'ERA5L'
    else:
        e5 = 'ERA5'

    levels = np.arange(0.25, 2.25, 0.25)

    era5 = regrid_era5(ds=era5['tEX_AF_CC'], grid=spcus['tEX_AF_CC'])

    params = {
        'SPARTACUS': {'data': spcus['tEX_AF_CC'], 'ax': fig.add_axes([0.53, 0.69, 0.17, 0.18]),
                      'title': 'SPCUS-P24H-p95WAS-'
                               + r'$\mathcal{A}_\mathrm{CC}^\mathrm{t}$'},
        'ERA5': {'data': era5, 'ax': fig.add_axes([0.71, 0.69, 0.19, 0.18]),
                 'title': f'{e5}-P24H-p95WAS-'
                          + r'$\mathcal{A}_\mathrm{CC}^\mathrm{t}$'}}

    # create mask
    mask = spcus['tEX_AF_CC']
    mask = mask.where(mask.isnull(), 1)

    for ids in ['SPARTACUS', 'ERA5']:
        acc = params[ids]['data']
        if ids == 'ERA5':
            acc = acc * mask

        iax = params[ids]['ax']
        map_vals = iax.imshow(acc,
                              cmap=matplotlib.cm.get_cmap('Oranges', len(levels)),
                              origin='lower', vmin=levels[0], vmax=levels[-1], aspect='auto')

        iax.set_xlim(409, 538)
        iax.set_ylim(27, 120)

        iax.axis('off')
        iax.set_title(params[ids]['title'], fontsize=11)

        iax.add_patch(pat.Rectangle(xy=(473, 56), height=20, width=25, edgecolor='black',
                                    fill=False, linewidth=1))
        iax.add_patch(pat.Rectangle(xy=(410, 27), height=93, width=128, edgecolor='black',
                                    fill=False, linewidth=1))

        if ids == 'ERA5':
            divider = make_axes_locatable(iax)
            cax = divider.append_axes('right', size='5%', pad=0.15)
            cb = fig.colorbar(map_vals, cax=cax, orientation='vertical', boundaries=levels,
                              extend='max')
            cb.set_label(label='tEX amplification', fontsize=9)

        iax.text(0.85, 0.05, 'SEA',
                 horizontalalignment='left',
                 verticalalignment='center', transform=iax.transAxes,
                 fontsize=9)
        iax.text(0.7, 0.3, 'FBR',
                 horizontalalignment='left',
                 verticalalignment='center', transform=iax.transAxes,
                 fontsize=9)


def plot_subplot(ax, spcus, era5, var, reg, land):

    cols = {'EF': 'tab:blue', 'FD': 'tab:purple', 'tEX': 'tab:orange', 'TEX': 'tab:red'}

    if var == 'Precip24Hsum_7to7':
        plot_vars = ['EF', 'FD', 'tEX']
        ymin, ymax = 0.6, 1.8
    else:
        plot_vars = ['EF', 'FD', 'tEX', 'TEX']
        ymin, ymax = 2 * 10 ** -1, 30

    gr_str, lgr_str = 'GR', 'AGR'
    if reg == 'FBR':
        lgr_str = 'GR'

    xticks = np.arange(1961, 2025)

    pstr = 'Tx99.0p'
    nv_var = 'TEX'
    if var != 'Temperature':
        pstr = 'P24h_7to7_95.0p'
        nv_var = 'tEX'

    rstr = 'AUT'
    if reg != 'AUT':
        rstr = 'SEA'

    nv = pd.read_csv(f'/data/users/hst/TEA-clean/TEA/paper_data/natural_variability/'
                     f'NV_AF_{pstr}_{rstr}.csv',
                     index_col=0)

    nat_var_low = np.ones(len(xticks)) * (1 - nv.loc[nv_var, 'lower'] * 1.645)
    nat_var_upp = np.ones(len(xticks)) * (1 + nv.loc[nv_var, 'upper'] * 1.645)
    ax.fill_between(x=xticks, y1=nat_var_low, y2=nat_var_upp, color=cols[nv_var], alpha=0.2)

    acc = 100
    for ivar, pvar in enumerate(plot_vars):
        try:
            ax.plot(xticks, era5[f'{pvar}_{lgr_str}_AF'], '--', color=cols[pvar], linewidth=1.5,
                    alpha=0.5)
            ax.plot(xticks[49:], np.ones(len(xticks[49:])) * era5[f'{pvar}_{lgr_str}_AF_CC'].values,
                    '--',
                    alpha=0.5, color=cols[pvar], linewidth=2)
        except KeyError:
            pass
        ax.plot(xticks, spcus[f'{pvar}_{gr_str}_AF'], color=cols[pvar], linewidth=2, markersize=3)
        ax.plot(xticks[49:], np.ones(len(xticks[49:])) * spcus[f'{pvar}_GR_AF_CC'].values,
                color=cols[pvar], linewidth=2)
        if spcus[f'{pvar}_{gr_str}_AF_CC'] < acc:
            acc = spcus[f'{pvar}_{gr_str}_AF_CC'].values

    ref_col = 'tab:red'
    if var == 'Precip24Hsum_7to7':
        ref_col = 'tab:orange'
    ax.plot(xticks[0:30], np.ones(len(xticks[0:30])), alpha=0.5, color=ref_col, linewidth=2)

    if var != 'Precip24Hsum_7to7':
        ax.set_yscale('log')
        ax.minorticks_on()
        ax.grid(color='gray', which='major', linestyle=':')
        ax.yaxis.set_minor_formatter(FixedFormatter(['0.2', '', '', '0.5', '', '', '', '',
                                                     '2.0', '', '', '5.0', '', '', '', '', '',
                                                     '20.0', '30.0']))
        ax.yaxis.set_minor_locator(FixedLocator([0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,
                                                 2, 3, 4, 5, 6, 7, 8, 9, 15, 20, 30]))
    else:
        ax.minorticks_on()
        ax.grid(color='gray', which='major', linestyle=':')

    tvar = var
    if var != 'Temperature':
        tvar = 'Precipitation'
    ax.tick_params(axis='both', which='major', labelsize=10)
    ax.set_title(f'Extremity amplification {reg} | {tvar}', fontsize=14)
    ax.set_ylim(ymin, ymax)
    ax.set_xlim(1960, 2025)
    ax.xaxis.set_minor_locator(FixedLocator(np.arange(1960, 2025)))
    ax.yaxis.set_major_formatter(ScalarFormatter())

    e5 = 'E5'
    if land:
        e5 = 'E5L'
    if var == 'Precip24Hsum_7to7':
        ax.yaxis.set_major_locator(FixedLocator(np.arange(0.6, 2.2, 0.4)))
        ax.set_ylabel('F' + r'$\,$|$\,$' + 'FD' + r'$\,$|$\,$' + 'tEX amplification', fontsize=10)
        xpos, ypos = 0.02, 0.28
        if land:
            off = 0.36
        else:
            off = 0.33
        xpos_cc, ypos_cc = 0.87, ((acc - ymin) / (ymax - ymin)) + off,
        cc_name = r'$\mathcal{A}_\mathrm{CC}^\mathrm{F, FD, t}$'
        e5_var = f'tEX_{lgr_str}_AF_CC'
        box_txt = ((('SPCUS-P24H-p95WAS-' + r'$\mathcal{A}_\mathrm{CC}^\mathrm{t}$ = '
                     + f'{np.round(spcus["tEX_GR_AF_CC"], 2):.2f}\n')
                    + f'{e5}-P24H-p95WAS-' + r'$\mathcal{A}_\mathrm{CC}^\mathrm{t}$ = ')
                   + f'{np.round(era5[e5_var], 2):.2f}')
    else:
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
        e5_var = f'TEX_{lgr_str}_AF_CC'
        box_txt = ((('SPCUS-TMax-p99ANN-' + r'$\mathcal{A}_\mathrm{CC}^\mathrm{T}$ = '
                     + f'{np.round(spcus["TEX_GR_AF_CC"], 2):.2f}\n')
                    + f'{e5}-TMax-p99ANN-' + r'$\mathcal{A}_\mathrm{CC}^\mathrm{T}$ = ')
                   + f'{np.round(era5[e5_var], 2):.2f}')

    ax.text(xpos, ypos, r'$\mathcal{A}_\mathrm{Ref}$',
            horizontalalignment='left',
            verticalalignment='center', transform=ax.transAxes,
            fontsize=11)
    ax.text(xpos_cc, ypos_cc, cc_name,
            horizontalalignment='left',
            verticalalignment='center', transform=ax.transAxes,
            fontsize=11)

    ax.text(0.02, 0.86, box_txt,
            horizontalalignment='left', verticalalignment='center', transform=ax.transAxes,
            backgroundcolor='whitesmoke', fontsize=9)


def create_legend(fig, ax, land):
    f, = ax.plot([-9, -9], 'tab:blue', linewidth=2)
    fd, = ax.plot([-9, -9], 'tab:purple', linewidth=2)
    fdm, = ax.plot([-9, -9], 'tab:orange', linewidth=2)
    fdma, = ax.plot([-9, -9], color='tab:red', linewidth=2)
    spar, = ax.plot([-9, -9], color='tab:gray', linewidth=2)
    era5, = ax.plot([-9, -9], color='tab:gray', linestyle='--', alpha=0.5, linewidth=2)

    if land:
        e5 = 'ERA5L'
    else:
        e5 = 'ERA5'

    fig.legend((spar, era5, f, fd, fdm, fdma),
               ('SPCUS', e5, r'$\mathcal{A}^\mathrm{F}$', r'$\mathcal{A}^\mathrm{FD}$',
                r'$\mathcal{A}^\mathrm{t}$', r'$\mathcal{A}^\mathrm{T}$'),
               ncol=6, loc=(0.27, 0.01))


def run():
    land = False

    vvars = ['Temperature', 'Precip24Hsum_7to7']
    regions = ['AUT', 'SEA', 'FBR']

    e5_ds = 'ERA5'
    if land:
        e5_ds = f'{e5_ds}Land'

    fig, axs = plt.subplots(3, 2, figsize=(12, 10))

    for icol, vvar in enumerate(vvars):
        for irow, reg in enumerate(regions):
            if vvar == 'Precip24Hsum_7to7' and reg == 'AUT':
                axs[irow, icol].axis('off')
                continue
            e5_data = get_data(reg=reg, var=vvar, ds=e5_ds)
            sp_data = get_data(reg=reg, var=vvar, ds='SPARTACUS')
            if vvar == 'Precip24Hsum_7to7' and reg == 'SEA':
                e5_data = get_data(reg='SAR', var=vvar, ds=e5_ds)
                plot_maps(fig=fig, spcus=sp_data, era5=e5_data, land=land)
            plot_subplot(ax=axs[irow, icol], spcus=sp_data, era5=e5_data, var=vvar, reg=reg,
                         land=land)

    axs[2, 0].set_xlabel('Time (core year of decadal-mean value)', fontsize=12)
    axs[2, 1].set_xlabel('Time (core year of decadal-mean value)', fontsize=12)
    axs[0, 1].set_title('Extremity amplification SEA/FBR | Precipitation', fontsize=14)

    fig.subplots_adjust(left=0.1, right=0.9, bottom=0.1, top=0.9, wspace=0.2, hspace=0.35)

    create_legend(fig=fig, ax=axs[0, 0], land=land)

    if land:
        fstr = 'Figure3'
        sdir = 'figure3/'
    else:
        fstr = 'ExtDataFig5'
        sdir = 'ExtDataFigs/'
    plt.savefig(f'/nas/home/hst/work/cdrDPS/plots/01_paper_figures/{sdir}{fstr}.png',
                dpi=300, bbox_inches='tight')


if __name__ == '__main__':
    run()
