import argparse
import glob
import matplotlib.pyplot as plt
import matplotlib.patches as pat
from matplotlib.ticker import FormatStrFormatter, FixedLocator
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
from scipy.stats import gmean
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
    dec = xr.open_dataset(f'/data/users/hst/TEA-clean/TEA/misc_data/dec_indicator_variables/'
                          f'DEC_Tx{opts.threshold:.1f}degC_{opts.region}_annual'
                          f'_SPARTACUS_1961to2024.nc')

    ctp = xr.open_mfdataset(
        sorted(glob.glob(f'/data/users/hst/TEA-clean/TEA/misc_data/ctp_indicator_variables/'
                         f'CTP_Tx{opts.threshold:.1f}degC_{opts.region}_annual_SPARTACUS_*.nc')),
        data_vars='minimal')

    af = xr.open_dataset(f'/data/users/hst/TEA-clean/TEA/misc_data/dec_indicator_variables/'
                         f'amplification/AF_Tx{opts.threshold:.1f}degC_{opts.region}_annual'
                         f'_SPARTACUS_1961to2024.nc')

    return dec, ctp, af


def ylims(opts, vname):
    if opts.region == 'SEA' and opts.threshold == 30:
        params = {'EF_GR': {'yn': 0, 'yx': 18, 'dy': 2},
                  'ED_avg_GR': {'yn': 0, 'yx': 8, 'dy': 2},
                  'EM_avg_GR': {'yn': 0, 'yx': 2, 'dy': 0.5},
                  'EA_avg_GR': {'yn': 0, 'yx': 100, 'dy': 20},
                  'TEX_GR': {'yn': 0, 'yx': 6000, 'dy': 1000}}
    elif opts.region == 'SEA' and opts.threshold == 25:
        params = {'EF_GR': {'yn': 5, 'yx': 25, 'dy': 2.5},
                  'ED_avg_GR': {'yn': 0, 'yx': 12, 'dy': 2},
                  'EM_avg_GR': {'yn': 0.5, 'yx': 3.5, 'dy': 0.5},
                  'EA_avg_GR': {'yn': 50, 'yx': 100, 'dy': 12.5},
                  'TEX_GR': {'yn': 0, 'yx': 30000, 'dy': 5000}}
    elif opts.region == 'AUT' and opts.threshold == 25:
        params = {'EF_GR': {'yn': 5, 'yx': 25, 'dy': 2.5},
                  'ED_avg_GR': {'yn': 2.5, 'yx': 15, 'dy': 2.5},
                  'EM_avg_GR': {'yn': 0.5, 'yx': 3.5, 'dy': 0.5},
                  'EA_avg_GR': {'yn': 200, 'yx': 600, 'dy': 50},
                  'TEX_GR': {'yn': 0, 'yx': 200000, 'dy': 25000}}
    elif opts.region == 'Niederösterreich' and opts.threshold == 25:
        params = {'EF_GR': {'yn': 5, 'yx': 30, 'dy': 5},
                  'ED_avg_GR': {'yn': 2, 'yx': 10, 'dy': 2},
                  'EM_avg_GR': {'yn': 0.5, 'yx': 4, 'dy': 0.5},
                  'EA_avg_GR': {'yn': 100, 'yx': 200, 'dy': 20},
                  'TEX_GR': {'yn': 0, 'yx': 70000, 'dy': 10000}}
    else:  # AUT 30degC
        params = {'EF_GR': {'yn': 0, 'yx': 18, 'dy': 2},
                  'ED_avg_GR': {'yn': 0, 'yx': 8, 'dy': 2},
                  'EM_avg_GR': {'yn': 0, 'yx': 2, 'dy': 0.5},
                  'EA_avg_GR': {'yn': 0, 'yx': 500, 'dy': 100},
                  'TEX_GR': {'yn': 0, 'yx': 45000, 'dy': 5000}}

    return params[vname]


def gr_plot_params(opts, vname):
    params = {'EF_GR': {'col': 'tab:blue',
                        'ylbl': r'EF ($F_s$|$F_p$) (ev/yr)',
                        'title': 'Event Frequency (Annual)',
                        'acc': r'$\mathcal{A}_\mathrm{CC}^\mathrm{F}$',
                        'cc_name': r'F$_\mathrm{CC}$',
                        'ref_name': r'F$_\mathrm{Ref}$',
                        'nv_name': 'EF',
                        'unit': 'ev/yr',
                        'lbl_name': 'F'},
              'ED_avg_GR': {'col': 'tab:purple',
                            'ylbl': r'ED $(\overline{D}_s$|$\overline{D}_p)$ (days)',
                            'title': 'Average Event Duration (events-mean)',
                            'acc': r'$\mathcal{A}_\mathrm{CC}^\mathrm{D}$',
                            'nv_name': 'ED',
                            'cc_name': r'$\overline{D}_\mathrm{CC}$',
                            'ref_name': r'$\overline{D}_\mathrm{Ref}$',
                            'unit': 'days',
                            'lbl_name': 'D'},
              'EM_avg_GR': {'col': 'tab:orange',
                            'ylbl': r'EM $(\overline{M}_s$|$\overline{M}_p)$ (°C)',
                            'title': 'Average Exceedance Magnitude (daily-mean)',
                            'acc': r'$\mathcal{A}_\mathrm{CC}^\mathrm{M}$',
                            'cc_name': r'$\overline{M}_\mathrm{CC}$',
                            'ref_name': r'$\overline{M}_\mathrm{Ref}$',
                            'nv_name': 'EM',
                            'lbl_name': 'M',
                            'unit': '°C'},
              'EA_avg_GR': {'col': 'tab:red',
                            'ylbl': r'EA $(\overline{A}_s$|$\overline{A}_p)$ (areals)',
                            'title': 'Average Exceedance Area (daily-mean)',
                            'cc_name': r'$\overline{A}_\mathrm{CC}$',
                            'ref_name': r'$\overline{A}_\mathrm{Ref}$',
                            'acc': r'$\mathcal{A}_\mathrm{CC}^\mathrm{A}$', 'nv_name': 'EA',
                            'unit': 'areals',
                            'lbl_name': 'A'},
              'TEX_GR': {'col': 'tab:red',
                         'ylbl': r'TEX $(\mathcal{T}_s|\mathcal{T}_p)$ (areal °C days/yr)',
                         'title': 'Total Events Extremity (Annual)',
                         'acc': r'$\mathcal{A}_\mathrm{CC}^\mathrm{T}$', 'nv_name': 'EA',
                         'cc_name': r'$\mathcal{T}_\mathrm{CC}$',
                         'ref_name': r'$\mathcal{T}_\mathrm{Ref}$',
                         'unit': 'areal °C days/yr',
                         'lbl_name': 'T'}
              }

    vparams = params[vname]
    yparams = ylims(opts=opts, vname=vname)
    vparams.update(yparams)

    return vparams


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


def plot_gr_data(opts, ax, adata, ddata, afdata, grvar):
    props = gr_plot_params(opts=opts, vname=grvar)

    sl = ddata[f'{grvar}_slow']
    su = ddata[f'{grvar}_supp']
    adata = adata[grvar]
    ddata = ddata[grvar]

    xticks = np.arange(1961, 2025)
    xvals = ddata.time

    ax.fill_between(x=xticks, y1=ddata - sl, y2=ddata + su, color=props['col'], alpha=0.3)
    ax.plot(xticks, ddata, 'o-', color=props['col'], markersize=3, linewidth=2)
    ax.plot(xticks, adata, 'o-', color=props['col'], markersize=2, linewidth=1, alpha=0.5)

    ax.plot(xticks[:30],
            np.ones(len(xvals[:30])) * gmean(ddata[5:26]),
            alpha=0.8,
            color=props['col'], linewidth=2)
    ax.plot(xticks[49:],
            np.ones(len(xvals[49:])) * gmean(ddata[-10:-4]),
            alpha=0.6,
            color=props['col'], linewidth=2)

    ax.set_ylabel(props['ylbl'], fontsize=12)
    ax.minorticks_on()
    ax.grid(color='gray', which='major', linestyle=':')
    if ddata.name == 'TEX_GR':
        ax.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
    else:
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    ax.set_xlim(1960, 2026)
    ax.xaxis.set_minor_locator(FixedLocator(np.arange(1960, 2026)))
    ax.set_title(props['title'], fontsize=14)
    ax.set_ylim(props['yn'], props['yx'])
    ax.yaxis.set_major_locator(FixedLocator(np.arange(props['yn'],
                                                      props['yx'] + props['dy'],
                                                      props['dy'])))

    ref = gmean(ddata[5:26])
    cc = gmean(ddata[-10:-4])
    af = afdata[f'{grvar}_AF_CC'].values
    ax.text(0.02, 0.89, f'TMax-{opts.threshold}°C-ANN-{props["nv_name"]}'
            + r'$_\mathrm{Ref | CC}$ = ' + f'{ref:.1f}' + r'$\,$|$\,$'
            + f'{cc:.1f} {props["unit"]} \n'
            + r'$\mathcal{A}_\mathrm{CC}^\mathrm{A}$ = '
            + f'{af:.2f}',
            horizontalalignment='left',
            verticalalignment='center', transform=ax.transAxes, backgroundcolor='whitesmoke',
            fontsize=9)

    ymin, ymax = props['yn'], props['yx']
    ypos_ref = ((gmean(ddata[5:26]) - ymin) / (ymax - ymin)) + 0.05
    ypos_cc = ((gmean(ddata[-10:-4]) - ymin) / (ymax - ymin)) + 0.05
    ax.text(0.02, ypos_ref, props['ref_name'],
            horizontalalignment='left',
            verticalalignment='center', transform=ax.transAxes,
            fontsize=9)

    ax.text(0.93, ypos_cc, props['cc_name'],
            horizontalalignment='left',
            verticalalignment='center', transform=ax.transAxes,
            fontsize=9)


def cb_lims(opts, vname):
    if opts.region == 'SEA' and opts.threshold == 25:
        params = {'EF': {'yn': 1, 'yx': 17, 'dy': 2},
                  'ED_avg': {'yn': 1, 'yx': 6, 'dy': 0.5},
                  'EM_avg': {'yn': 1, 'yx': 4, 'dy': 0.25}}
    elif opts.region == 'AUT' and opts.threshold == 25:
        params = {'EF': {'yn': 0, 'yx': 20, 'dy': 2.5},
                  'ED_avg': {'yn': 0, 'yx': 6, 'dy': 0.5},
                  'EM_avg': {'yn': 0, 'yx': 4, 'dy': 0.25}}
    elif opts.region == 'Niederösterreich' and opts.threshold == 25:
        params = {'EF': {'yn': 0, 'yx': 20, 'dy': 2.5},
                  'ED_avg': {'yn': 0, 'yx': 6, 'dy': 0.5},
                  'EM_avg': {'yn': 0, 'yx': 4, 'dy': 0.25}}
    else:  # AUT 30degC, SEA 30degC
        params = {'EF': {'yn': 0, 'yx': 12, 'dy': 1},
                  'ED_avg': {'yn': 0, 'yx': 4, 'dy': 0.5},
                  'EM_avg': {'yn': 0, 'yx': 2, 'dy': 0.25}}

    lvls = np.arange(params[vname]['yn'],
                     params[vname]['yx'] + params[vname]['dy'],
                     params[vname]['dy'])

    return lvls


def map_plot_params(opts, vname):
    params = {'EF': {'cmap': 'Blues',
                     'lbl': r'EF$_\mathrm{CC}$(i,j) (ev/yr)',
                     'title': 'Event Frequency (Annual) (CC2008-2022)'},
              'ED_avg': {'cmap': 'Purples',
                         'lbl': r'ED$_\mathrm{CC}$(i,j) (days)',
                         'title': 'Avarage Event Duration (CC2008-2022)'},
              'EM_avg': {'cmap': 'Oranges',
                         'lbl': r'EM$_\mathrm{CC}$(i,j) (°C)',
                         'title': 'Average Exceedance Magnitude (CC2008-2022)'}
              }

    lvls = cb_lims(opts=opts, vname=vname)
    params[vname]['lvls'] = lvls

    return params[vname]


def plot_map(opts, fig, ax, data):
    props = map_plot_params(opts=opts, vname=data.name)
    data = data.where(data > 0)

    aut = xr.open_dataset('/data/arsclisys/normal/clim-hydro/TEA-Indicators/masks/'
                          'AUT_masks_SPARTACUS.nc')
    ax.contourf(aut.x, aut.y, aut.nw_mask, colors='lightgrey')

    if data.max() > props['lvls'][-1] and data.min() > props['lvls'][0]:
        ext = 'max'
    elif data.max() < props['lvls'][-1] and data.min() > props['lvls'][0]:
        ext = 'neither'
    elif data.max() < props['lvls'][-1] and data.min() < props['lvls'][0]:
        ext = 'min'
    else:
        ext = 'both'

    range_vals = find_range(data=data)

    map = ax.contourf(data.x, data.y, data, cmap=props['cmap'], levels=props['lvls'], extend=ext)
    if opts.region in ['AUT', 'SEA']:
        ax.add_patch(pat.Rectangle(xy=(559000, 5193000), height=20000, width=25000,
                                   edgecolor='black', fill=False, linewidth=1))
        ax.add_patch(pat.Rectangle(xy=(486000, 5161000), height=94000, width=133000,
                                   edgecolor='black', fill=False, linewidth=1))
    ax.axis('off')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    cb = fig.colorbar(map, cax=cax, orientation='vertical', extend=ext)
    cb.set_label(label=props['lbl'], fontsize=12)
    cb.ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    ax.set_title(props['title'], fontsize=14)

    a_sym = props['lbl'].split(' ')[0]
    if opts.region == 'AUT':
        ax.text(0.02, 0.82, a_sym + '\n'
                + f'AUT: [{range_vals["AUT"][0]:.2f}, {range_vals["AUT"][1]:.2f}]\n'
                  f'SEA: [{range_vals["SEA"][0]:.2f}, {range_vals["SEA"][1]:.2f}]\n'
                  f'FBR: [{range_vals["FBR"][0]:.2f}, {range_vals["FBR"][1]:.2f}]',
                horizontalalignment='left',
                verticalalignment='center', transform=ax.transAxes, backgroundcolor='whitesmoke',
                fontsize=9)
    if opts.region == 'SEA':
        ax.text(0.02, 0.87, a_sym + '\n'
                + f'SEA: [{range_vals["SEA"][0]:.2f}, {range_vals["SEA"][1]:.2f}]\n'
                  f'FBR: [{range_vals["FBR"][0]:.2f}, {range_vals["FBR"][1]:.2f}]',
                horizontalalignment='left',
                verticalalignment='center', transform=ax.transAxes, backgroundcolor='whitesmoke',
                fontsize=9)
    else:
        ax.text(0.02, 0.89, a_sym + '\n'
                + f'{opts.region}: [{range_vals["AUT"][0]:.2f}, {range_vals["AUT"][1]:.2f}]',
                horizontalalignment='left',
                verticalalignment='center', transform=ax.transAxes, backgroundcolor='whitesmoke',
                fontsize=9)


def run():
    opts = getopts()

    dec, ann, af = get_data(opts=opts)

    fig, axs = plt.subplots(4, 2, figsize=(14, 16))

    gr_vars = ['EF_GR', 'ED_avg_GR', 'EM_avg_GR', 'EA_avg_GR']
    for irow, gr_var in enumerate(gr_vars):
        plot_gr_data(opts=opts, ax=axs[irow, 0], adata=ann, ddata=dec, afdata=af, grvar=gr_var)

    plot_gr_data(opts=opts, ax=axs[3, 1], adata=ann, ddata=dec, afdata=af, grvar='TEX_GR')

    map_vars = ['EF', 'ED_avg', 'EM_avg']
    for irow, map_var in enumerate(map_vars):
        mdata = gmean(dec[map_var].sel(time=slice('2015-01-01', '2020-12-31')), axis=0)
        mdata = xr.DataArray(data=mdata, coords={'y': (['y'], dec.y.values),
                                                 'x': (['x'], dec.x.values)}, name=map_var)
        plot_map(opts=opts, fig=fig, ax=axs[irow, 1], data=mdata)

    axs[2, 1].text(0, 0, 'Alpine data at z > 1500m excluded.',
                   horizontalalignment='left', verticalalignment='center',
                   transform=axs[2, 1].transAxes, backgroundcolor='lightgrey',
                   fontsize=8)

    fig.subplots_adjust(wspace=0.2, hspace=0.33)
    plt.savefig(f'/nas/home/hst/work/TEAclean/plots/misc/EDF3/'
                f'EDF3_{opts.region}{opts.threshold:.1f}degC.png', dpi=300,
                bbox_inches='tight')


if __name__ == '__main__':
    run()
