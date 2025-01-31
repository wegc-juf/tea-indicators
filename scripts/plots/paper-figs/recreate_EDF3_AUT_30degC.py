import glob
import matplotlib.pyplot as plt
import matplotlib.patches as pat
from matplotlib.ticker import FormatStrFormatter, FixedLocator
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
from scipy.stats import gmean
import xarray as xr

from recreate_fig2 import find_range


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
    dec = xr.open_dataset('/data/users/hst/TEA-clean/TEA/dec_indicator_variables/'
                          'DEC_Tx30.0degC_AUT_WAS_SPARTACUS_1961to2024.nc')

    ctp = xr.open_mfdataset(
        sorted(glob.glob('/data/users/hst/TEA-clean/TEA/ctp_indicator_variables/'
                         'CTP_Tx30.0degC_AUT_WAS_SPARTACUS_*.nc')), data_vars='minimal')

    supp = xr.open_dataset('/data/users/hst/TEA-clean/TEA/dec_indicator_variables/'
                           'DEC_sUPP_Tx30.0degC_AUT_WAS_SPARTACUS_1961to2024.nc')

    slow = xr.open_dataset('/data/users/hst/TEA-clean/TEA/dec_indicator_variables/'
                           'DEC_sLOW_Tx30.0degC_AUT_WAS_SPARTACUS_1961to2024.nc')

    return dec, ctp, supp, slow


def gr_plot_params(vname):
    params = {'EF_GR': {'col': 'tab:blue',
                        'ylbl': r'EF ($F_s$|$F_p$) (ev/yr)',
                        'title': 'Event Frequency (Annual)',
                        'acc': r'$\mathcal{A}_\mathrm{CC}^\mathrm{F}$',
                        'cc_name': r'F$_\mathrm{CC}$',
                        'ref_name': r'F$_\mathrm{Ref}$',
                        'nv_name': 'EF',
                        'unit': 'ev/yr',
                        'lbl_name': 'F',
                        'yx': 18, 'dy': 2},
              'EDavg_GR': {'col': 'tab:purple',
                           'ylbl': r'ED $(\overline{D}_s$|$\overline{D}_p)$ (days)',
                           'title': 'Average Event Duration (events-mean)',
                           'acc': r'$\mathcal{A}_\mathrm{CC}^\mathrm{D}$',
                           'nv_name': 'ED',
                           'cc_name': r'$\overline{D}_\mathrm{CC}$',
                           'ref_name': r'$\overline{D}_\mathrm{Ref}$',
                           'unit': 'days',
                           'lbl_name': 'D',
                           'yx': 8, 'dy': 2},
              'EMavg_GR': {'col': 'tab:orange',
                           'ylbl': r'EM $(\overline{M}_s$|$\overline{M}_p)$ (°C)',
                           'title': 'Average Exceedance Magnitude (daily-mean)',
                           'acc': r'$\mathcal{A}_\mathrm{CC}^\mathrm{M}$',
                           'cc_name': r'$\overline{M}_\mathrm{CC}$',
                           'ref_name': r'$\overline{M}_\mathrm{Ref}$',
                           'nv_name': 'EM',
                           'lbl_name': 'M',
                           'unit': '°C',
                           'yx': 2, 'dy': 0.5},
              'EAavg_GR': {'col': 'tab:red',
                           'ylbl': r'EA $(\overline{A}_s$|$\overline{A}_p)$ (areals)',
                           'title': 'Average Exceedance Area (daily-mean)',
                           'cc_name': r'$\overline{A}_\mathrm{CC}$',
                           'ref_name': r'$\overline{A}_\mathrm{Ref}$',
                           'acc': r'$\mathcal{A}_\mathrm{CC}^\mathrm{A}$', 'nv_name': 'EA',
                           'unit': 'areals',
                           'lbl_name': 'A',
                           'yx': 500, 'dy': 100},
              'TEX_GR': {'col': 'tab:red',
                         'ylbl': r'TEX $(\mathcal{T}_s|\mathcal{T}_p)$ (areal °C days/yr)',
                         'title': 'Total Events Extremity (Annual)',
                         'acc': r'$\mathcal{A}_\mathrm{CC}^\mathrm{T}$', 'nv_name': 'EA',
                         'cc_name': r'$\mathcal{T}_\mathrm{CC}$',
                         'ref_name': r'$\mathcal{T}_\mathrm{Ref}$',
                         'unit': 'areal °C days/yr',
                         'lbl_name': 'T',
                         'yx': 45000, 'dy': 5000}
              }

    return params[vname]


def plot_gr_data(ax, adata, ddata, su, sl):
    props = gr_plot_params(vname=ddata.name)

    xticks = np.arange(1961, 2025)
    xvals = ddata.ctp

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
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    ax.set_xlim(1960, 2026)
    ax.xaxis.set_minor_locator(FixedLocator(np.arange(1960, 2026)))
    ax.set_title(props['title'], fontsize=14)
    ax.set_ylim(0, props['yx'])
    ax.yaxis.set_major_locator(FixedLocator(np.arange(0, props['yx'] + props['dy'], props['dy'])))

    ref = gmean(ddata[5:26])
    cc = gmean(ddata[-10:-4])
    af = cc / ref
    ax.text(0.02, 0.89, f'TMax-p99ANN-{props["nv_name"]}' + r'$_\mathrm{Ref | CC}$ = '
            + f'{ref:.1f}' + r'$\,$|$\,$'
            + f'{cc:.1f} {props["unit"]} \n'
            + r'$\mathcal{A}_\mathrm{CC}^\mathrm{A}$ = '
            + f'{af:.2f}',
            horizontalalignment='left',
            verticalalignment='center', transform=ax.transAxes, backgroundcolor='whitesmoke',
            fontsize=9)

    ymin, ymax = 0, props['yx']
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


def map_plot_params(vname):
    params = {'EF': {'cmap': 'Blues',
                     'lbl': r'EF$_\mathrm{CC}$(i,j) (ev/yr)',
                     'title': 'Event Frequency (Annual) (CC2008-2022',
                     'lvls': np.arange(1, 12)},
              'EDavg': {'cmap': 'Purples',
                        'lbl': r'ED$_\mathrm{CC}$(i,j) (days)',
                        'title': 'Avarage Event Duration (CC2008-2022)',
                        'lvls': np.arange(1, 4.25, 0.25)},
              'EMavg': {'cmap': 'Oranges',
                        'lbl': r'EM$_\mathrm{CC}$(i,j) (°C)',
                        'title': 'Average Exceedance Magnitude (CC2008-2022)',
                        'lvls': np.arange(1, 2.8, 0.2)}
              }

    return params[vname]


def plot_map(fig, ax, data):
    props = map_plot_params(vname=data.name)

    aut = xr.open_dataset('/data/arsclisys/normal/clim-hydro/TEA-Indicators/masks/'
                          'AUT_masks_SPARTACUS.nc')
    ax.contourf(aut.nw_mask, colors='mistyrose')

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
    dec, ann, supp, slow = get_data()

    fig, axs = plt.subplots(4, 2, figsize=(14, 16))

    gr_vars = ['EF_GR', 'EDavg_GR', 'EMavg_GR', 'EAavg_GR']
    for irow, gr_var in enumerate(gr_vars):
        plot_gr_data(ax=axs[irow, 0], adata=ann[gr_var], ddata=dec[gr_var],
                     su=supp[f'{gr_var}_supp'], sl=slow[f'{gr_var}_slow'])

    plot_gr_data(ax=axs[3, 1], adata=ann['TEX_GR'], ddata=dec['TEX_GR'],
                 su=supp['TEX_GR_supp'], sl=slow['TEX_GR_slow'])

    map_vars = ['EF', 'EDavg', 'EMavg']
    for irow, map_var in enumerate(map_vars):
        mdata = gmean(dec[map_var].sel(ctp=slice('2015-01-01', '2020-12-31')), axis=0)
        mdata = xr.DataArray(data=mdata, coords={'y': (['y'], dec.y.values),
                                                 'x': (['x'], dec.x.values)}, name=map_var)
        plot_map(fig=fig, ax=axs[irow, 1], data=mdata)

    fig.subplots_adjust(wspace=0.2, hspace=0.33)
    plt.savefig('/nas/home/hst/work/TEAclean/plots/misc/EDF3_AUT30.0degC.png', dpi=300,
                bbox_inches='tight')


if __name__ == '__main__':
    run()
