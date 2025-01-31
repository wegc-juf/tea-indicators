import argparse
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter, FixedLocator
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.patches as pat
import numpy as np
import pandas as pd
import xarray as xr


def getopts():
    """
    get CLI arguments
    :return: command line parameters
    """

    parser = argparse.ArgumentParser()

    parser.add_argument('--region',
                        type=str,
                        default='AUT',
                        choices=['AUT', 'SEA', 'Niederösterreich'],
                        help='GeoRegion. Default: AUT.')

    parser.add_argument('--threshold',
                        type=int,
                        default=30,
                        choices=[30, 25],
                        help='Absolute threshold value. Default: 30.')

    myopts = parser.parse_args()

    return myopts


def get_data(opts):
    af = xr.open_dataset(f'/data/users/hst/TEA-clean/TEA/amplification/'
                         f'AF_Tx{opts.threshold:.1f}degC_{opts.region}_WAS_SPARTACUS_1961to2024.nc')

    try:
        nv = pd.read_csv(f'/data/users/hst/TEA-clean/TEA/natural_variability/'
                         f'NV_AF_Tx{opts.threshold:.1f}degC_{opts.region}.csv',
                         index_col=0)
    except FileNotFoundError:
        nv = None

    return af, nv


def gr_plot_params(vname):
    params = {'EF_GR_AF': {'col': 'tab:blue',
                           'ylbl': r'EF amplification $(\mathcal{A}^\mathrm{F})$',
                           'title': 'Event Frequency (Annual)',
                           'unit': 'ev/yr',
                           'acc': r'$\mathcal{A}_\mathrm{CC}^\mathrm{F}$',
                           'nv_name': 'EF'},
              'EDavg_GR_AF': {'col': 'tab:purple',
                              'ylbl': r'ED amplification $(\mathcal{A}^\mathrm{D})$',
                              'title': 'Average Event Duration (events-mean)',
                              'unit': 'days',
                              'acc': r'$\mathcal{A}_\mathrm{CC}^\mathrm{D}$',
                              'nv_name': 'ED'},
              'EMavg_GR_AF': {'col': 'tab:orange',
                              'ylbl': r'EM amplification $(\mathcal{A}^\mathrm{M})$',
                              'title': 'Average Exceedance Magnitude (daily-mean)',
                              'unit': '°C',
                              'acc': r'$\mathcal{A}_\mathrm{CC}^\mathrm{M}$',
                              'nv_name': 'EM'},
              'EAavg_GR_AF': {'col': 'tab:red',
                              'ylbl': r'EA amplification $(\mathcal{A}^\mathrm{A})$',
                              'title': 'Average Exceedance Area (daily-mean)',
                              'unit': 'areals',
                              'acc': r'$\mathcal{A}_\mathrm{CC}^\mathrm{A}$', 'nv_name': 'EA'}
              }

    return params[vname]


def map_plot_params(opts, vname):
    params = {'EF_AF_CC': {'cmap': 'Blues',
                           'lbl': r'$\mathcal{A}^\mathrm{F}_\mathrm{CC}$',
                           'title': f'Event Frequency (EF) amplification (CC2010-2024)'},
              'EDavg_AF_CC': {'cmap': 'Purples',
                           'lbl': r'$\mathcal{A}^\mathrm{D}_\mathrm{CC}$',
                           'title': f'Event Duration (ED) amplification (CC2010-2024)'},
              'EMavg_AF_CC': {'cmap': 'Oranges',
                           'lbl': r'$\mathcal{A}^\mathrm{M}_\mathrm{CC}$',
                           'title': f'Exceedance Magnitude (EM) amplification (CC2010-2024)'}
              }

    cb_params = {'AUT': {'vn': 1, 'vx': 6, 'delta': 0.5},
                 'Niederösterreich': {'vn': 0, 'vx': 3, 'delta': 0.25}}

    return params[vname], cb_params[opts.region]


def plot_gr_data(opts, ax, data, af_cc, nv):
    props = gr_plot_params(vname=data.name)

    xvals = data.ctp
    xticks = np.arange(1961, 2025)

    if nv is not None:
        nat_var_low = np.ones(len(xvals)) * (1 - nv.loc[props['nv_name'], 'lower'] * 1.645)
        nat_var_upp = np.ones(len(xvals)) * (1 + nv.loc[props['nv_name'], 'upper'] * 1.645)
        ax.fill_between(x=xticks, y1=nat_var_low, y2=nat_var_upp, color=props['col'], alpha=0.2)

    ax.plot(xticks, data, 'o-', color=props['col'], markersize=3, linewidth=2)

    ax.plot(xticks[0:30], np.ones(len(xvals[:30])), alpha=0.5, color=props['col'], linewidth=2)
    ax.plot(xticks[49:], np.ones(len(xvals[49:])) * af_cc.values, alpha=0.5,
            color=props['col'], linewidth=2)

    ax.set_ylabel(props['ylbl'], fontsize=12)
    ax.tick_params(axis='both', which='major', labelsize=10)
    ax.minorticks_on()
    ax.grid(color='gray', which='major', linestyle=':')
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    ax.set_xlim(1960, 2025)
    ax.xaxis.set_minor_locator(FixedLocator(np.arange(1960, 2025)))

    ymin, ymax, yticks = get_ylims(opts=opts)
    ax.set_yticks(yticks)
    ax.set_ylim(ymin, ymax)

    ax.set_title(props['title'], fontsize=14)

    ypos_ref = ((1 - ymin) / (ymax - ymin)) + 0.05
    ypos_cc = ((af_cc.values - ymin) / (ymax - ymin)) + 0.05
    ax.text(0.02, ypos_ref, r'$\mathcal{A}_\mathrm{Ref}$',
            horizontalalignment='left',
            verticalalignment='center', transform=ax.transAxes,
            fontsize=11)
    ax.text(0.93, ypos_cc, props['acc'],
            horizontalalignment='left',
            verticalalignment='center', transform=ax.transAxes,
            fontsize=11)

    ax.text(0.02, 0.92, props['acc'] + ' = ' + f'{af_cc:.2f}',
            horizontalalignment='left',
            verticalalignment='center', transform=ax.transAxes, backgroundcolor='whitesmoke',
            fontsize=9)

    if data.name == 'EAavg_GR_AF':
        ax.set_xlabel('Time (core year of decadal-mean value)', fontsize=10)


def get_ylims(opts):

    values = {'AUT': {30: {'yn': 0.5, 'yx': 2.5, 'dy': 0.5}},
              'Niederösterreich': {25: {'yn': 0.5, 'yx': 2, 'dy': 0.25}}}

    props = values[opts.region][opts.threshold]
    yn, yx, dy = props['yn'], props['yx'], props['dy']
    ticks = np.arange(yn, yx + dy, dy)

    return yn, yx, ticks


def get_ylims_tex(opts):

    values = {'AUT': {30: {'yn': 0, 'yx': 9, 'dy': 1}},
              'Niederösterreich': {25: {'yn': 0, 'yx': 4, 'dy': 0.5}}}

    props = values[opts.region][opts.threshold]
    yn, yx, dy = props['yn'], props['yx'], props['dy']
    ticks = np.arange(yn, yx + dy, dy)

    return yn, yx, ticks

def plot_tex_es(opts, ax, data, af_cc, nv):

    xvals = data.ctp
    xticks = np.arange(1961, 2025)

    if nv is not None:
        nat_var_low = np.ones(len(xvals)) * (1 - nv.loc['TEX', 'lower'] * 1.645)
        nat_var_upp = np.ones(len(xvals)) * (1 + nv.loc['TEX', 'upper'] * 1.645)
        ax.fill_between(x=xticks, y1=nat_var_low, y2=nat_var_upp, color='tab:red', alpha=0.2)

    ax.plot(xticks, data['ESavg_GR_AF'], 'o-', color='tab:grey', markersize=3, linewidth=2)
    ax.plot(xticks, data['TEX_GR_AF'], 'o-', color='tab:red', markersize=3, linewidth=2)

    ax.plot(xticks[0:30], np.ones(len(xvals[:30])), alpha=0.5, color='tab:grey', linewidth=2)
    ax.plot(xticks[49:], np.ones(len(xvals[49:])) * af_cc['ESavg_GR_AF_CC'].values, alpha=0.5,
            color='tab:grey', linewidth=2)
    ax.plot(xticks[49:], np.ones(len(xvals[49:])) * af_cc['TEX_GR_AF_CC'].values, alpha=0.5,
            color='tab:red', linewidth=2)

    ax.set_ylabel(r'ES|TEX amplification $(\mathcal{A}^\mathrm{S}, \mathcal{A}^\mathrm{T})$',
                  fontsize=12)
    ax.tick_params(axis='both', which='major', labelsize=10)
    ax.minorticks_on()
    ax.grid(color='gray', which='major', linestyle=':')
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    ax.set_xlim(1960, 2025)
    ax.xaxis.set_minor_locator(FixedLocator(np.arange(1960, 2025)))

    ymin, ymax, yticks = get_ylims_tex(opts=opts)
    ax.set_yticks(yticks)
    ax.set_ylim(ymin, ymax)

    ax.set_title('Avg. Event Severity and Total Events Extremity', fontsize=14)
    ax.set_xlabel('Time (core year of decadal-mean value)', fontsize=10)

    ypos_ref = ((1 - ymin) / (ymax - ymin)) + 0.05
    ypos_cc_tex = ((af_cc['TEX_GR_AF_CC'].values - ymin) / (ymax - ymin)) + 0.05
    ypos_cc_es = ((af_cc['ESavg_GR_AF_CC'].values - ymin) / (ymax - ymin)) + 0.05
    ax.text(0.02, ypos_ref, r'$\mathcal{A}_\mathrm{Ref}$',
            horizontalalignment='left',
            verticalalignment='center', transform=ax.transAxes,
            fontsize=11)
    ax.text(0.93, ypos_cc_es, r'$\mathcal{A}_\mathrm{CC}^\mathrm{S}$',
            horizontalalignment='left',
            verticalalignment='center', transform=ax.transAxes,
            fontsize=11)
    if opts.region != 'Niederösterreich':
        ax.text(0.93, ypos_cc_tex, r'$\mathcal{A}_\mathrm{CC}^\mathrm{T}$',
                horizontalalignment='left',
                verticalalignment='center', transform=ax.transAxes,
                fontsize=11)
    else:
        ax.text(0.93, ypos_cc_tex-0.1, r'$\mathcal{A}_\mathrm{CC}^\mathrm{T}$',
                horizontalalignment='left',
                verticalalignment='center', transform=ax.transAxes,
                fontsize=11)

    ax.text(0.02, 0.9,
            r'$\mathcal{A}_\mathrm{CC}^\mathrm{S} | \mathcal{A}_\mathrm{CC}^\mathrm{T}$ = '
            + f'{af_cc["ESavg_GR_AF_CC"]:.2f}' + r'$\,$|$\,$'
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


def plot_map(opts, fig, ax, data):

    props, cb_props = map_plot_params(opts=opts, vname=data.name)

    aut = xr.open_dataset('/data/arsclisys/normal/clim-hydro/TEA-Indicators/masks/'
                          'AUT_masks_SPARTACUS.nc')
    ax.contourf(aut.nw_mask, colors='mistyrose')


    vn, vx = cb_props['vn'], cb_props['vx']
    lvls = np.arange(vn, vx + cb_props['delta'], cb_props['delta'])
    if data.max() > lvls[-1] and data.min() > lvls[0]:
        ext = 'max'
    elif data.max() < lvls[-1] and data.min() > lvls[0]:
        ext = 'neither'
    else:
        ext = 'min'

    range_vals = find_range(data=data)

    map_vals = ax.contourf(data, cmap=props['cmap'], extend=ext, levels=lvls, vmin=vn, vmax=vx)
    if opts.region in ['AUT', 'SEA']:
        ax.add_patch(pat.Rectangle(xy=(473, 56), height=20, width=25, edgecolor='black',
                                   fill=False, linewidth=1))
        ax.add_patch(pat.Rectangle(xy=(410, 28), height=92, width=125, edgecolor='black',
                                   fill=False, linewidth=1))

    ax.axis('off')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    cb = fig.colorbar(map_vals, cax=cax, orientation='vertical')
    cb.set_label(label=f'TMax-p99ANN-{props["lbl"]}', fontsize=12)
    cb.ax.tick_params(labelsize=10)
    cb.ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    ax.set_title(props["title"], fontsize=14)

    if opts.region == 'AUT':
        ax.text(0.02, 0.82, props['lbl'] + '(i,j)\n'
                + f'AUT: [{range_vals["AUT"][0]:.2f}, {range_vals["AUT"][1]:.2f}]\n'
                  f'SEA: [{range_vals["SEA"][0]:.2f}, {range_vals["SEA"][1]:.2f}]\n'
                  f'FBR: [{range_vals["FBR"][0]:.2f}, {range_vals["FBR"][1]:.2f}]',
                horizontalalignment='left',
                verticalalignment='center', transform=ax.transAxes, backgroundcolor='whitesmoke',
                fontsize=9)
    if opts.region == 'SEA':
        ax.text(0.02, 0.87, props['lbl'] + '(i,j)\n'
                + f'SEA: [{range_vals["SEA"][0]:.2f}, {range_vals["SEA"][1]:.2f}]\n'
                  f'FBR: [{range_vals["FBR"][0]:.2f}, {range_vals["FBR"][1]:.2f}]',
                horizontalalignment='left',
                verticalalignment='center', transform=ax.transAxes, backgroundcolor='whitesmoke',
                fontsize=9)


def run():
    opts = getopts()

    data, natv = get_data(opts=opts)

    fig, axs = plt.subplots(4, 2, figsize=(14, 16))

    gr_vars = ['EF_GR_AF', 'EDavg_GR_AF', 'EMavg_GR_AF', 'EAavg_GR_AF']
    for irow, gr_var in enumerate(gr_vars):
        plot_gr_data(opts=opts, ax=axs[irow, 0], data=data[gr_var],
                     af_cc=data[f'{gr_var}_CC'], nv=natv)

    map_vars = ['EF_AF_CC', 'EDavg_AF_CC', 'EMavg_AF_CC']
    for irow, map_var in enumerate(map_vars):
        plot_map(opts=opts, fig=fig, ax=axs[irow, 1], data=data[map_var])

    plot_tex_es(opts=opts, ax=axs[3, 1], data=data[['TEX_GR_AF', 'ESavg_GR_AF']],
                af_cc=data[[f'TEX_GR_AF_CC', f'ESavg_GR_AF_CC']], nv=natv)

    fig.subplots_adjust(wspace=0.2, hspace=0.33)
    plt.savefig(f'/nas/home/hst/work/TEAclean/plots/misc/'
                f'Fig2_{opts.region}_{opts.threshold}degC.png',
                bbox_inches='tight', dpi=300)


if __name__ == '__main__':
    run()
