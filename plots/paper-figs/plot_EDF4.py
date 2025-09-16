import matplotlib.pyplot as plt
import matplotlib.patches as pat
from matplotlib.ticker import FormatStrFormatter, FixedLocator
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
from scipy.stats import gmean
import xarray as xr

from teametrics.common.general_functions import ref_cc_params


def get_data():
    af = xr.open_dataset('/data/users/hst/TEA-clean/TEA/paper_data/dec_indicator_variables/'
                         'amplification/'
                         'AF_P24h_7to7_95.0p_SEA_WAS_SPARTACUS_1961to2024.nc')

    nv = xr.open_dataset('/data/users/hst/TEA-clean/TEA/paper_data/natural_variability/'
                         'NV_AF_P24h_7to7_95.0p_SEA.nc')

    dec = xr.open_dataset('/data/users/hst/TEA-clean/TEA/paper_data/dec_indicator_variables/'
                          'DEC_P24h_7to7_95.0p_SEA_WAS_SPARTACUS_1961to2024.nc')

    return af, nv, dec


def gr_plot_params(vname):
    params = {'EF_GR_AF': {'col': 'tab:blue',
                           'ylbl': r'EF amplification $(\mathcal{A}^\mathrm{F})$',
                           'title': 'Event Frequency (Annual)',
                           'acc': r'$\mathcal{A}_\mathrm{CC}^\mathrm{F}$',
                           'nv_name': 's_EF_AF_NV', 'unit': 'ev/yr', 'legend_name': 'EF'},
              'ED_avg_GR_AF': {'col': 'tab:purple',
                               'ylbl': r'ED amplification $(\mathcal{A}^\mathrm{D})$',
                               'title': 'Average Event Duration (events-mean)',
                               'acc': r'$\mathcal{A}_\mathrm{CC}^\mathrm{D}$',
                               'nv_name': 's_ED_avg_AF_NV', 'unit': 'days', 'legend_name': 'ED'},
              'EM_avg_GR_Md_AF': {'col': 'tab:orange',
                                  'ylbl': r'EM amplification $(\mathcal{A}^\mathrm{M})$',
                                  'title': 'Average Exceedance Magnitude (daily-median)',
                                  'acc': r'$\mathcal{A}_\mathrm{CC}^\mathrm{M}$',
                                  'nv_name': 's_EM_avg_Md_AF_NV', 'unit': 'mm', 'legend_name': 'EM'},
              'tEX_GR_AF': {'col': 'tab:orange',
                            'ylbl': r'tEX amplification $(\mathcal{A}^\mathrm{t})$',
                            'title': 'Temporal Events Extremity (Annual)',
                            'acc': r'$\mathcal{A}_\mathrm{CC}^\mathrm{tEX}$',
                            'nv_name': 's_tEX_AF_NV',
                            'unit': 'mm days/yr', 'legend_name': 'tEX'}
              }

    return params[vname]


def map_plot_params(vname):
    params = {'EF_AF_CC': {'cmap': 'Blues',
                           'lbl': r'$\mathcal{A}^\mathrm{F}_\mathrm{CC}$',
                           'title': 'Event Frequency (EF) amplification (CC2008-2024)',
                           'lvls': np.arange(0.4, 1.8, 0.2), 'vn': 0.4, 'vx': 1.6},
              'ED_avg_AF_CC': {'cmap': 'Purples',
                               'lbl': r'$\mathcal{A}^\mathrm{D}_\mathrm{CC}$',
                               'title': 'Event Duration (ED) amplification (CC2008-2024)',
                               'lvls': np.arange(0.4, 1.8, 0.2), 'vn': 0.4, 'vx': 1.6},
              'EM_avg_Md_AF_CC': {'cmap': 'Oranges',
                                  'lbl': r'$\mathcal{A}^\mathrm{M}_\mathrm{CC}$',
                                  'title': 'Exceedance Magnitude (EM) amplification (CC2008-2024)',
                                  'lvls': np.arange(0.4, 1.8, 0.2), 'vn': 0.4, 'vx': 1.6},
              'tEX_AF_CC': {'cmap': 'Oranges',
                            'lbl': r'$\mathcal{A}^\mathrm{tEX}_\mathrm{CC}$',
                            'title': 'Temporal Events Extremity (tEX) Ampl. (CC2008-2024)',
                            'lvls': np.arange(0.25, 2.25, 0.25), 'vn': 0.25, 'vx': 2}
              }

    return params[vname]


def plot_gr_data(ax, data, af_cc, nv, ddata):
    props = gr_plot_params(vname=data.name)

    xvals = data.time
    xticks = np.arange(1961, 2025)

    nat_var_low = np.ones(len(xvals)) * (1 - nv[f'{props["nv_name"]}low'].values * 1.645)
    nat_var_upp = np.ones(len(xvals)) * (1 + nv[f'{props["nv_name"]}upp'].values * 1.645)
    ax.fill_between(x=xticks, y1=nat_var_low, y2=nat_var_upp, color=props['col'], alpha=0.2)

    ax.plot(xticks, data, 'o-', color=props['col'], markersize=3, linewidth=2)

    ax.plot(xticks[0:30], np.ones(len(xvals[:30])), alpha=0.5, color=props['col'], linewidth=2)
    ax.plot(xticks[49:], np.ones(len(xvals[49:])) * af_cc.values, alpha=0.5,
            color=props['col'], linewidth=2)

    ax.set_ylabel(props['ylbl'], fontsize=12)
    ax.tick_params(axis='both', which='major', labelsize=10)
    ax.minorticks_on()
    ax.grid(color='gray', which='major', linestyle=':')
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    ax.set_xlim(1960, 2025)
    ax.xaxis.set_minor_locator(FixedLocator(np.arange(1960, 2025)))

    ymin, ymax = 0.6, 1.6
    maj_ticks = np.arange(ymin, ymax + 0.1, 0.1)
    ax.set_yticks(maj_ticks)
    ax.set_ylim(ymin, ymax)
    ax.yaxis.set_minor_locator(FixedLocator(np.arange(ymin, ymax + 0.02, 0.02)))
    labels = [f'{tick:.1f}' if i % 2 == 0 else '' for i, tick in enumerate(maj_ticks)]
    ax.set_yticklabels(labels)

    ax.set_title(props['title'], fontsize=14)

    ypos_ref = 0.55
    ypos_cc = ((af_cc.values - ymin) / (ymax - ymin)) + 0.05
    ax.text(0.02, ypos_ref, r'$\mathcal{A}_\mathrm{Ref}$',
            horizontalalignment='left',
            verticalalignment='center', transform=ax.transAxes,
            fontsize=11)
    ax.text(0.93, ypos_cc, props['acc'],
            horizontalalignment='left',
            verticalalignment='center', transform=ax.transAxes,
            fontsize=11)

    params = ref_cc_params()
    ref = gmean(ddata.sel(time=slice(params['REF']['start_cy'], params['REF']['end_cy'])).values)
    cc = gmean(ddata.sel(time=slice(params['CC']['start_cy'], params['CC']['end_cy'])).values)

    ax.text(0.02, 0.9, f'P24H-p95WAS-{props["legend_name"]}' + r'$_\mathrm{Ref | CC}$ = '
            + f'{ref:.2f}' + r'$\,$|$\,$'
            + f'{cc:.2f} {props["unit"]} \n'
            + props['acc'] + ' = ' + f'{af_cc:.2f}',
            horizontalalignment='left',
            verticalalignment='center', transform=ax.transAxes, backgroundcolor='whitesmoke',
            fontsize=9)

    if data.name == 'EA_avg_GR_AF':
        ax.set_xlabel('Time (core year of decadal-mean value)', fontsize=10)


def find_range(data):
    """
    find min and max values for AUT, SEA and FBR
    :param data: input data
    :return: dict with min and max values
    """

    data = data.where(data > 0)

    sea_mask = xr.open_dataset('/data/arsclisys/normal/clim-hydro/TEA-Indicators/masks/'
                               'SEA_masks_SPARTACUS.nc')
    sea_data = data * sea_mask.nw_mask
    sea_min, sea_max = sea_data.min().values, sea_data.max().values

    fbr_mask = xr.open_dataset('/data/arsclisys/normal/clim-hydro/TEA-Indicators/masks/'
                               'FBR_masks_SPARTACUS.nc')
    fbr_data = data * fbr_mask.nw_mask
    fbr_min, fbr_max = fbr_data.min().values, fbr_data.max().values

    ranges = {'SEA': [sea_min, sea_max], 'FBR': [fbr_min, fbr_max]}

    return ranges


def plot_map(fig, ax, data):
    props = map_plot_params(vname=data.name)

    aut = xr.open_dataset('/data/arsclisys/normal/clim-hydro/TEA-Indicators/masks/'
                          'AUT_masks_SPARTACUS.nc')
    ax.contourf(aut.x, aut.y, aut.nw_mask, colors='mistyrose')

    if data.max() > props['lvls'][-1] and data.min() > props['lvls'][0]:
        ext = 'max'
    elif data.max() < props['lvls'][-1] and data.min() > props['lvls'][0]:
        ext = 'neither'
    else:
        ext = 'min'

    range_vals = find_range(data=data)

    xn_idx = np.where(aut.x.values == data.x.values[0])[0][0]
    xx_idx = np.where(aut.x.values == data.x.values[-1])[0][0]
    yn_idx = np.where(aut.y.values == data.y.values[0])[0][0]
    yx_idx = np.where(aut.y.values == data.y.values[-1])[0][0]

    map_vals = ax.contourf(aut.x[xn_idx:xx_idx + 1], aut.y[yn_idx:yx_idx + 1], data,
                           cmap=props['cmap'], extend=ext, levels=props['lvls'],
                           vmin=props['vn'], vmax=props['vx'])

    fbr_x, fbr_y = aut.x[470], aut.y[74]
    sea_x, sea_y = aut.x[402], aut.y[42]
    ax.add_patch(pat.Rectangle(xy=(fbr_x, fbr_y), height=20000, width=25000, edgecolor='black',
                               fill=False, linewidth=1))
    ax.add_patch(pat.Rectangle(xy=(sea_x, sea_y), height=97000, width=135000, edgecolor='black',
                               fill=False, linewidth=1))
    ax.axis('off')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    cb = fig.colorbar(map_vals, cax=cax, orientation='vertical')
    cb.set_label(label=f'P24H-p95WAS-{props["lbl"]}', fontsize=12)
    cb.ax.tick_params(labelsize=10)
    cb.ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    ax.set_title(props["title"], fontsize=14)

    ax.text(0.02, 0.85, props['lbl'] + '(i,j)\n'
            + f'SEA: [{range_vals["SEA"][0]:.2f}, {range_vals["SEA"][1]:.2f}]\n'
              f'FBR: [{range_vals["FBR"][0]:.2f}, {range_vals["FBR"][1]:.2f}]',
            horizontalalignment='left',
            verticalalignment='center', transform=ax.transAxes, backgroundcolor='whitesmoke',
            fontsize=9)


def run():
    data, natv, ddata = get_data()

    fig, axs = plt.subplots(4, 2, figsize=(14, 16))

    gr_vars = ['EF_GR', 'ED_avg_GR', 'EM_avg_GR_Md', 'tEX_GR']
    for irow, gr_var in enumerate(gr_vars):
        plot_gr_data(ax=axs[irow, 0], data=data[f'{gr_var}_AF'], ddata=ddata[gr_var],
                     af_cc=data[f'{gr_var}_AF_CC'], nv=natv)

    map_vars = ['EF_AF_CC', 'ED_avg_AF_CC', 'EM_avg_Md_AF_CC', 'tEX_AF_CC']
    for irow, map_var in enumerate(map_vars):
        plot_map(fig=fig, ax=axs[irow, 1], data=data[map_var])

    axs[3, 1].text(0, 0, 'Alpine data at z > 1500m excluded.',
                   horizontalalignment='left', verticalalignment='center',
                   transform=axs[3, 1].transAxes, backgroundcolor='mistyrose',
                   fontsize=8)

    labels = ['a)', 'e)', 'b)', 'f)', 'c)', 'g)', 'd)', 'h)']
    for i, ax in enumerate(axs.flat):
        ax.text(-0.15, 1.2, labels[i], transform=ax.transAxes, fontsize=14,
                va='top', ha='left')

    fig.subplots_adjust(wspace=0.2, hspace=0.33)
    plt.savefig('/nas/home/hst/work/cdrDPS/plots/01_paper_figures/ExtDataFigs/'
                'ExtDataFig4.png', dpi=300, bbox_inches='tight')


if __name__ == '__main__':
    run()
