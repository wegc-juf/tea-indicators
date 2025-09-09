import matplotlib.pyplot as plt
import matplotlib.patches as pat
from matplotlib.ticker import FormatStrFormatter, FixedLocator
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
from scipy.stats import gmean
import xarray as xr

from teametrics.common.general_functions import ref_cc_params

PARAMS = ref_cc_params()


def get_data():
    af = xr.open_dataset('/data/users/hst/TEA-clean/TEA/paper_data/dec_indicator_variables/'
                         'amplification/'
                         'AF_Tx99.0p_AUT_annual_SPARTACUS_1961to2024.nc')

    dec = xr.open_dataset('/data/users/hst/TEA-clean/TEA/paper_data/dec_indicator_variables/'
                          'DEC_Tx99.0p_AUT_annual_SPARTACUS_1961to2024.nc')

    nv = xr.open_dataset('/data/users/hst/TEA-clean/TEA/paper_data/natural_variability/'
                         'NV_AF_Tx99.0p_AUT.nc')

    return af, dec, nv


def gr_plot_params(vname):
    params = {'EF_GR_AF': {'col': 'tab:blue',
                           'ylbl': r'EF amplification $(\mathcal{A}^\mathrm{F})$',
                           'title': 'Event Frequency (Annual)',
                           'unit': 'ev/yr',
                           'acc': r'$\mathcal{A}_\mathrm{CC}^\mathrm{F}$',
                           'nv_name': 's_EF_AF_NV'},
              'ED_avg_GR_AF': {'col': 'tab:purple',
                              'ylbl': r'ED amplification $(\mathcal{A}^\mathrm{D})$',
                              'title': 'Average Event Duration (events-mean)',
                              'unit': 'days',
                              'acc': r'$\mathcal{A}_\mathrm{CC}^\mathrm{D}$',
                              'nv_name': 's_ED_avg_AF_NV'},
              'EM_avg_GR_AF': {'col': 'tab:orange',
                              'ylbl': r'EM amplification $(\mathcal{A}^\mathrm{M})$',
                              'title': 'Average Exceedance Magnitude (daily-mean)',
                              'unit': '°C',
                              'acc': r'$\mathcal{A}_\mathrm{CC}^\mathrm{M}$',
                              'nv_name': 's_EM_avg_AF_NV'},
              'EA_avg_GR_AF': {'col': 'tab:red',
                              'ylbl': r'EA amplification $(\mathcal{A}^\mathrm{A})$',
                              'title': 'Average Exceedance Area (daily-mean)',
                              'unit': 'areals',
                              'acc': r'$\mathcal{A}_\mathrm{CC}^\mathrm{A}$',
                              'nv_name': 's_EA_avg_AF_NV'}
              }

    return params[vname]


def map_plot_params(vname):
    params = {'EF_AF_CC': {'cmap': 'Blues',
                           'lbl': r'$\mathcal{A}^\mathrm{F}_\mathrm{CC}$',
                           'title': f'Event Frequency (EF) amplification (CC2010-2024)'},
              'ED_avg_AF_CC': {'cmap': 'Purples',
                              'lbl': r'$\mathcal{A}^\mathrm{D}_\mathrm{CC}$',
                              'title': f'Event Duration (ED) amplification (CC2010-2024)'},
              'EM_avg_AF_CC': {'cmap': 'Oranges',
                              'lbl': r'$\mathcal{A}^\mathrm{M}_\mathrm{CC}$',
                              'title': f'Exceedance Magnitude (EM) amplification (CC2010-2024)'}
              }

    return params[vname]


def plot_gr_data(ax, data, af_cc, ddata, nv):
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

    ymin, ymax = 0.5, 2
    ax.set_yticks(np.arange(ymin, ymax + 0.5, 0.5))
    ax.set_ylim(ymin, ymax)

    ax.set_title(props['title'], fontsize=14)

    ypos_ref = 0.4
    ypos_cc = ((af_cc.values - ymin) / (ymax - ymin)) + 0.05
    ax.text(0.02, ypos_ref, r'$\mathcal{A}_\mathrm{Ref}$',
            horizontalalignment='left',
            verticalalignment='center', transform=ax.transAxes,
            fontsize=11)
    ax.text(0.93, ypos_cc, props['acc'],
            horizontalalignment='left',
            verticalalignment='center', transform=ax.transAxes,
            fontsize=11)

    ref_abs = gmean(ddata.sel(time=slice(PARAMS['REF']['start_cy'], PARAMS['REF']['end_cy'])))
    cc_abs = gmean(ddata.sel(time=slice(PARAMS['CC']['start_cy'], PARAMS['CC']['end_cy'])))

    if data.name == 'EA_avg_GR_AF':
        ax.text(0.02, 0.9, f'TMax-p99ANN-{data.name[:3]}' + r'$_\mathrm{Ref | CC}$ = '
                + f'{ref_abs:.1f}' + r'$\,$|$\,$'
                + f'{cc_abs:.1f} {props["unit"]} \n'
                + props['acc'] + ' = ' + f'{af_cc:.1f}',
                horizontalalignment='left',
                verticalalignment='center', transform=ax.transAxes, backgroundcolor='whitesmoke',
                fontsize=9)
        ax.set_xlabel('Time (core year of decadal-mean value)', fontsize=10)
    else:
        ax.text(0.02, 0.9, f'TMax-p99ANN-{data.name[:3]}' + r'$_\mathrm{Ref | CC}$ = '
                + f'{ref_abs:.2f}' + r'$\,$|$\,$'
                + f'{cc_abs:.2f} {props["unit"]} \n'
                + props['acc'] + ' = ' + f'{af_cc:.2f}',
                horizontalalignment='left',
                verticalalignment='center', transform=ax.transAxes, backgroundcolor='whitesmoke',
                fontsize=9)

def plot_tex_es(ax, data, af_cc, ddata,  nv):
    xvals = data.time
    xticks = np.arange(1961, 2025)

    nat_var_low = np.ones(len(xvals)) * (1 - nv['s_TEX_AF_NVlow'].values * 1.645)
    nat_var_upp = np.ones(len(xvals)) * (1 + nv['s_TEX_AF_NVupp'].values * 1.645)
    ax.fill_between(x=xticks, y1=nat_var_low, y2=nat_var_upp, color='tab:red', alpha=0.2)

    ax.plot(xticks, data['ES_avg_GR_AF'], 'o-', color='tab:grey', markersize=3, linewidth=2)
    ax.plot(xticks, data['TEX_GR_AF'], 'o-', color='tab:red', markersize=3, linewidth=2)

    ax.plot(xticks[0:30], np.ones(len(xvals[:30])), alpha=0.5, color='tab:grey', linewidth=2)
    ax.plot(xticks[49:], np.ones(len(xvals[49:])) * af_cc['ES_avg_GR_AF_CC'].values, alpha=0.5,
            color='tab:grey', linewidth=2)
    ax.plot(xticks[49:], np.ones(len(xvals[49:])) * af_cc['TEX_GR_AF_CC'].values, alpha=0.5,
            color='tab:red', linewidth=2)

    ax.set_ylabel(r'ES|TEX amplification $(\mathcal{A}^\mathrm{S}, \mathcal{A}^\mathrm{T})$',
                  fontsize=12)
    ax.tick_params(axis='both', which='major', labelsize=10)
    ax.minorticks_on()
    ax.grid(color='gray', which='major', linestyle=':')
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.0f'))
    ax.set_xlim(1960, 2025)
    ax.xaxis.set_minor_locator(FixedLocator(np.arange(1960, 2025)))

    ymin, ymax = 0, 10
    ax.set_yticks(np.arange(ymin, ymax + 1, 1))
    ax.set_ylim(ymin, ymax)

    ax.set_title('Avg. Event Severity and Total Events Extremity', fontsize=14)
    ax.set_xlabel('Time (core year of decadal-mean value)', fontsize=10)

    ypos_ref = 0.12
    ypos_cc_tex = ((af_cc['TEX_GR_AF_CC'].values - ymin) / (ymax - ymin)) + 0.05
    ypos_cc_es = ((af_cc['ES_avg_GR_AF_CC'].values - ymin) / (ymax - ymin)) + 0.05
    ax.text(0.02, ypos_ref, r'$\mathcal{A}_\mathrm{Ref}$',
            horizontalalignment='left',
            verticalalignment='center', transform=ax.transAxes,
            fontsize=11)
    ax.text(0.93, ypos_cc_es, r'$\mathcal{A}_\mathrm{CC}^\mathrm{S}$',
            horizontalalignment='left',
            verticalalignment='center', transform=ax.transAxes,
            fontsize=11)
    ax.text(0.93, ypos_cc_tex, r'$\mathcal{A}_\mathrm{CC}^\mathrm{T}$',
            horizontalalignment='left',
            verticalalignment='center', transform=ax.transAxes,
            fontsize=11)

    ref_abs_tex = gmean(ddata['TEX_GR'].sel(
        time=slice(PARAMS['REF']['start_cy'], PARAMS['REF']['end_cy'])))
    cc_abs_tex = gmean(ddata['TEX_GR'].sel(
        time=slice(PARAMS['CC']['start_cy'], PARAMS['CC']['end_cy'])))
    ref_abs_es = gmean(ddata['ES_avg_GR'].sel(
        time=slice(PARAMS['REF']['start_cy'], PARAMS['REF']['end_cy'])))
    cc_abs_es = gmean(ddata['ES_avg_GR'].sel(
        time=slice(PARAMS['CC']['start_cy'], PARAMS['CC']['end_cy'])))

    ax.text(0.02, 0.85,
            'TMax-p99ANN-TEX' + r'$_\mathrm{Ref | CC}$'
            + f' = {ref_abs_tex:.0f}' + r'$\,$|$\,$'
            + f'{cc_abs_tex:.0f} ' + r'areal$\,$°C$\,$days$\,$/$\,$yr '
            + f'\nTMax-p99ANN-ES' + r'$_\mathrm{Ref | CC}$'
            + f' = {ref_abs_es:.0f}' + r'$\,$|$\,$'
            + f'{cc_abs_es:.0f} ' + r'areal$\,$°C$\,$days$\,$'
            + '\n'
            + r'$\mathcal{A}_\mathrm{CC}^\mathrm{S} | \mathcal{A}_\mathrm{CC}^\mathrm{T}$ = '
            + f'{af_cc["ES_avg_GR_AF_CC"]:.2f}' + r'$\,$|$\,$'
            + f'{af_cc["TEX_GR_AF_CC"]:.2f}',
            horizontalalignment='left',
            verticalalignment='center', transform=ax.transAxes, backgroundcolor='whitesmoke',
            fontsize=9)


def find_range(data):
    """
    find min and max values for AUT, SEA and FBR
    :param data: input data
    :return: dict with min and max values
    """

    data = data.where(data > 0)

    aut_min, aut_max = data.min().values, data.max().values

    sea_mask = xr.open_dataset('/data/arsclisys/normal/clim-hydro/TEA-Indicators/masks/'
                               'SEA_masks_SPARTACUS.nc')
    sea_data = data * sea_mask.nw_mask
    sea_min, sea_max = sea_data.min().values, sea_data.max().values

    fbr_mask = xr.open_dataset('/data/arsclisys/normal/clim-hydro/TEA-Indicators/masks/'
                               'FBR_masks_SPARTACUS.nc')
    fbr_data = data * fbr_mask.nw_mask
    fbr_min, fbr_max = fbr_data.min().values, fbr_data.max().values

    ranges = {'AUT': [aut_min, aut_max], 'SEA': [sea_min, sea_max], 'FBR': [fbr_min, fbr_max]}

    return ranges


def plot_map(fig, ax, data):
    props = map_plot_params(vname=data.name)

    aut = xr.open_dataset('/data/arsclisys/normal/clim-hydro/TEA-Indicators/masks/'
                          'AUT_masks_SPARTACUS.nc')
    aut = aut.sel(x=data.x, y=data.y)
    ax.contourf(aut.nw_mask, colors='mistyrose')

    lvls = np.arange(1, 4.25, 0.25)
    if data.max() > lvls[-1] and data.min() > lvls[0]:
        ext = 'max'
    elif data.max() < lvls[-1] and data.min() > lvls[0]:
        ext = 'neither'
    else:
        ext = 'min'

    range_vals = find_range(data=data)

    map_vals = ax.contourf(data, cmap=props['cmap'], extend=ext, levels=lvls, vmin=1, vmax=4)

    ax.add_patch(pat.Rectangle(xy=(473, 55), height=20, width=25, edgecolor='black',
                               fill=False, linewidth=1))
    ax.add_patch(pat.Rectangle(xy=(410, 27), height=92, width=125, edgecolor='black',
                               fill=False, linewidth=1))
    ax.axis('off')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    cb = fig.colorbar(map_vals, cax=cax, orientation='vertical')
    cb.set_label(label=f'TMax-p99ANN-{props["lbl"]}', fontsize=12)
    cb.ax.tick_params(labelsize=10)
    cb.ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    ax.set_title(props["title"], fontsize=14)

    ax.text(0.02, 0.82, props['lbl'] + '(i,j)\n'
            + f'AUT: [{range_vals["AUT"][0]:.2f}, {range_vals["AUT"][1]:.2f}]\n'
              f'SEA: [{range_vals["SEA"][0]:.2f}, {range_vals["SEA"][1]:.2f}]\n'
              f'FBR: [{range_vals["FBR"][0]:.2f}, {range_vals["FBR"][1]:.2f}]',
            horizontalalignment='left',
            verticalalignment='center', transform=ax.transAxes, backgroundcolor='whitesmoke',
            fontsize=9)


def run():
    data, dec_data, natv = get_data()

    fig, axs = plt.subplots(4, 2, figsize=(14, 16))

    gr_vars = ['EF_GR_AF', 'ED_avg_GR_AF', 'EM_avg_GR_AF', 'EA_avg_GR_AF']
    for irow, gr_var in enumerate(gr_vars):
        plot_gr_data(ax=axs[irow, 0], data=data[gr_var], af_cc=data[f'{gr_var}_CC'],
                     ddata=dec_data[gr_var.split('_AF')[0]], nv=natv)

    map_vars = ['EF_AF_CC', 'ED_avg_AF_CC', 'EM_avg_AF_CC']
    for irow, map_var in enumerate(map_vars):
        plot_map(fig=fig, ax=axs[irow, 1], data=data[map_var])

    plot_tex_es(ax=axs[3, 1], data=data[['TEX_GR_AF', 'ES_avg_GR_AF']],
                ddata=dec_data[['TEX_GR', 'ES_avg_GR']],
                af_cc=data[[f'TEX_GR_AF_CC', f'ES_avg_GR_AF_CC']], nv=natv)

    # iterate over each subplot and add a text label
    labels = ['a)', 'e)', 'b)', 'f)', 'c)', 'g)', 'd)', 'h)']
    for i, ax in enumerate(axs.flat):
        ax.text(-0.1, 1.2, labels[i], transform=ax.transAxes, fontsize=14,
                va='top', ha='left')

    fig.subplots_adjust(wspace=0.2, hspace=0.33)
    plt.savefig('/nas/home/hst/work/cdrDPS/plots/01_paper_figures/figure2/Figure2.png',
                dpi=300, bbox_inches='tight')

if __name__ == '__main__':
    run()
