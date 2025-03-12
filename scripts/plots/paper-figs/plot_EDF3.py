import glob
import matplotlib.pyplot as plt
import matplotlib.patches as pat
from matplotlib.ticker import FormatStrFormatter, FixedLocator
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
from scipy.stats import gmean
import xarray as xr

from plot_fig2 import find_range

from scripts.general_stuff.general_functions import ref_cc_params

PARAMS = ref_cc_params()

def preprocess(ds_in):
    """
    drops unnecessary variables from ds
    :param ds_in: ds
    :return: new ds
    """
    ds = ds_in.copy()

    ds = ds.drop_vars(['EF', 'EF_GR', 'EDavg', 'EDavg_GR', 'EMavg', 'EMavg_GR',
                       'EAavg_GR', 'TEX_GR', 'ESavg_GR'])

    return ds


def get_data():
    dec = xr.open_dataset('/data/users/hst/TEA-clean/TEA/paper_data/dec_indicator_variables/'
                          'DEC_Tx99.0p_AUT_WAS_SPARTACUS_1961to2024.nc')

    ctp = xr.open_mfdataset(
        sorted(glob.glob('/data/users/hst/TEA-clean/TEA/paper_data/ctp_indicator_variables/'
                         'CTP_Tx99.0p_AUT_WAS_SPARTACUS_*.nc')), data_vars='minimal')

    af = xr.open_dataset('/data/users/hst/TEA-clean/TEA/paper_data/dec_indicator_variables/'
                         'amplification/AF_Tx99.0p_AUT_WAS_SPARTACUS_1961to2024.nc')

    return dec, ctp, af


def gr_plot_params(vname):
    params = {'EF_GR': {'col': 'tab:blue',
                        'ylbl': r'EF ($F_s$|$F_p$) (ev/yr)',
                        'title': 'Event Frequency (Annual)',
                        'acc': r'$\mathcal{A}_\mathrm{CC}^\mathrm{F}$',
                        'nv_name': 'EF',
                        'yx': 16, 'dy': 4,
                        'unit': 'ev/yr',
                        'ref': r'$F_\mathrm{Ref}$', 'cc': r'$F_\mathrm{CC}$'},
              'ED_avg_GR': {'col': 'tab:purple',
                           'ylbl': r'ED $(\overline{D}_s$|$\overline{D}_p)$ (days)',
                           'title': 'Average Event Duration (events-mean)',
                           'acc': r'$\mathcal{A}_\mathrm{CC}^\mathrm{D}$',
                           'nv_name': 'ED',
                           'yx': 10, 'dy': 2,
                            'unit': 'days',
                        'ref': r'$\overline{D}_\mathrm{Ref}$', 'cc': r'$\overline{D}_\mathrm{CC}$'},
              'EM_avg_GR': {'col': 'tab:orange',
                           'ylbl': r'EM $(\overline{M}_s$|$\overline{M}_p)$ (°C)',
                           'title': 'Average Exceedance Magnitude (daily-mean)',
                           'acc': r'$\mathcal{A}_\mathrm{CC}^\mathrm{M}$',
                           'nv_name': 'EM',
                           'yx': 2, 'dy': 0.5, 'unit': '°C',
                            'ref': r'$\overline{M}_\mathrm{Ref}$',
                            'cc': r'$\overline{M}_\mathrm{CC}$'},
              'EA_avg_GR': {'col': 'tab:red',
                           'ylbl': r'EA $(\overline{A}_s$|$\overline{A}_p)$ (areals)',
                           'title': 'Average Exceedance Area (daily-mean)',
                           'acc': r'$\mathcal{A}_\mathrm{CC}^\mathrm{A}$', 'nv_name': 'EA',
                           'yx': 600, 'dy': 100, 'unit': 'areals',
                            'ref': r'$\overline{A}_\mathrm{Ref}$',
                            'cc': r'$\overline{A}_\mathrm{CC}$'},
              'TEX_GR': {'col': 'tab:red',
                         'ylbl': r'TEX $(\mathcal{T}_s|\mathcal{T}_p)$ (areal °C days/yr)',
                         'title': 'Total Events Extremity (Annual)',
                         'acc': r'$\mathcal{A}_\mathrm{CC}^\mathrm{T}$', 'nv_name': 'EA',
                         'yx': 45000, 'dy': 5000, 'unit': 'areal °C days/yr',
                         'ref': r'$\mathcal{T}_\mathrm{Ref}$', 'cc': r'$\mathcal{T}_\mathrm{CC}$'}
              }

    return params[vname]


def plot_gr_data(ax, adata, ddata, afdata, su, sl):
    props = gr_plot_params(vname=ddata.name)

    xticks = np.arange(1961, 2025)

    ref = gmean(ddata.sel(time=slice(PARAMS['REF']['start_cy'], PARAMS['REF']['end_cy'])))
    cc = gmean(ddata.sel(time=slice(PARAMS['CC']['start_cy'], PARAMS['CC']['end_cy'])))

    ax.fill_between(x=xticks, y1=ddata - sl, y2=ddata + su, color=props['col'], alpha=0.2)
    ax.plot(xticks, ddata, 'o-', color=props['col'], markersize=3, linewidth=2)
    ax.plot(xticks, adata, 'o-', color=props['col'], markersize=2, linewidth=1, alpha=0.5)
    ax.plot(xticks[:30], np.ones(len(xticks[:30])) * ref,
            alpha=0.8, color=props['col'], linewidth=2)
    ax.plot(xticks[49:], np.ones(len(xticks[49:])) * cc,
            alpha=0.6, color=props['col'], linewidth=2)

    ypos_ref = (ref / props['yx']) + 0.05
    ypos_cc = (cc / props['yx']) + 0.05
    ax.text(0.02, ypos_ref, props['ref'],
            horizontalalignment='left',
            verticalalignment='center', transform=ax.transAxes,
            fontsize=11)

    ax.text(0.93, ypos_cc, props['cc'],
            horizontalalignment='left',
            verticalalignment='center', transform=ax.transAxes,
            fontsize=11)

    ax.set_ylabel(props['ylbl'], fontsize=12)
    ax.minorticks_on()
    ax.grid(color='gray', which='major', linestyle=':')
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    ax.set_xlim(1960, 2025)
    ax.xaxis.set_minor_locator(FixedLocator(np.arange(1960, 2025)))
    ax.set_title(props['title'], fontsize=14)
    ax.set_ylim(0, props['yx'])
    ax.yaxis.set_major_locator(FixedLocator(np.arange(0, props['yx'] + props['dy'], props['dy'])))

    ax.text(0.02, 0.89, f'TMax-p99ANN-{props["nv_name"]}' + r'$_\mathrm{Ref | CC}$ = '
            + f'{ref:.2f}' + r'$\,$|$\,$'
            + f'{cc:.2f} {props["unit"]} \n'
            + r'$\mathcal{A}_\mathrm{CC}^\mathrm{A}$ = '
            + f'{afdata:.2f}',
            horizontalalignment='left',
            verticalalignment='center', transform=ax.transAxes, backgroundcolor='whitesmoke',
            fontsize=9)


def map_plot_params(vname):
    params = {'EF': {'cmap': 'Blues',
                     'lbl': r'EF$_\mathrm{CC}$(i,j) (ev/yr)',
                     'title': 'Event Frequency (Annual) (CC2008-2022',
                     'lvls': np.arange(1, 11)},
              'ED_avg': {'cmap': 'Purples',
                        'lbl': r'ED$_\mathrm{CC}$(i,j) (days)',
                        'title': 'Avarage Event Duration (CC2008-2022)',
                        'lvls': np.arange(1, 3.75, 0.25)},
              'EM_avg': {'cmap': 'Oranges',
                        'lbl': r'EM$_\mathrm{CC}$(i,j) (°C)',
                        'title': 'Average Exceedance Magnitude (CC2008-2022)',
                        'lvls': np.arange(1, 2.6, 0.2)}
              }

    return params[vname]


def plot_map(fig, ax, data):
    props = map_plot_params(vname=data.name)

    aut = xr.open_dataset('/data/arsclisys/normal/clim-hydro/TEA-Indicators/masks/'
                          'AUT_masks_SPARTACUS.nc')
    ax.contourf(aut.nw_mask, colors='mistyrose')

    data = data.where(data > 0)

    if data.max() > props['lvls'][-1] and data.min() > props['lvls'][0]:
        ext = 'max'
    elif data.max() < props['lvls'][-1] and data.min() > props['lvls'][0]:
        ext = 'neither'
    else:
        ext = 'min'

    range_vals = find_range(data=data)

    map = ax.contourf(data, cmap=props['cmap'], levels=props['lvls'], extend=ext)
    ax.add_patch(pat.Rectangle(xy=(473, 56), height=20, width=25, edgecolor='black',
                               fill=False, linewidth=1))
    ax.add_patch(pat.Rectangle(xy=(410, 28), height=92, width=125, edgecolor='black',
                               fill=False, linewidth=1))
    ax.axis('off')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    cb = fig.colorbar(map, cax=cax, orientation='vertical', extend=ext)
    cb.set_label(label=props['lbl'], fontsize=12)
    cb.ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    ax.set_title(props['title'], fontsize=14)

    a_sym = props['lbl'].split(' ')[0]
    ax.text(0.02, 0.82, a_sym + '\n'
            + f'AUT: [{range_vals["AUT"][0]:.2f}, {range_vals["AUT"][1]:.2f}]\n'
              f'SEA: [{range_vals["SEA"][0]:.2f}, {range_vals["SEA"][1]:.2f}]\n'
              f'FBR: [{range_vals["FBR"][0]:.2f}, {range_vals["FBR"][1]:.2f}]',
            horizontalalignment='left',
            verticalalignment='center', transform=ax.transAxes, backgroundcolor='whitesmoke',
            fontsize=9)


def run():
    dec, ann, af = get_data()

    fig, axs = plt.subplots(4, 2, figsize=(14, 16))

    gr_vars = ['EF_GR', 'ED_avg_GR', 'EM_avg_GR', 'EA_avg_GR']
    for irow, gr_var in enumerate(gr_vars):
        plot_gr_data(ax=axs[irow, 0], adata=ann[gr_var], ddata=dec[gr_var],
                     afdata=af[f'{gr_var}_AF_CC'],
                     su=dec[f'{gr_var}_supp'], sl=dec[f'{gr_var}_slow'])

    plot_gr_data(ax=axs[3, 1], adata=ann['TEX_GR'], ddata=dec['TEX_GR'], afdata=af['TEX_GR_AF_CC'],
                 su=dec['TEX_GR_supp'], sl=dec['TEX_GR_slow'])

    map_vars = ['EF', 'ED_avg', 'EM_avg']
    for irow, map_var in enumerate(map_vars):
        mdata = gmean(dec[map_var].sel(time=slice('2015-01-01', '2020-12-31')), axis=0)
        mdata = xr.DataArray(data=mdata, coords={'y': (['y'], dec.y.values),
                                                 'x': (['x'], dec.x.values)}, name=map_var)
        plot_map(fig=fig, ax=axs[irow, 1], data=mdata)

    # iterate over each subplot and add a text label
    labels = ['a', 'e', 'b', 'f', 'c', 'g', 'd', 'h']
    for i, ax in enumerate(axs.flat):
        ax.text(-0.1, 1.2, labels[i], transform=ax.transAxes, fontsize=14, fontweight='bold',
                va='top', ha='left')

    fig.subplots_adjust(wspace=0.2, hspace=0.33)
    plt.savefig('/nas/home/hst/work/cdrDPS/plots/01_paper_figures/ExtDataFigs/'
                'ExtDataFig3.png', dpi=300, bbox_inches='tight')


if __name__ == '__main__':
    run()
