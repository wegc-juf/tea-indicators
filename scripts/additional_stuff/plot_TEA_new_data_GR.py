import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import numpy as np
from scipy.stats import gmean
import xarray as xr


def get_props(var, pvar):
    if pvar == 'P24h_7to7':
        pstr = 'P24H'
        yx_f, yx_m, yx_fm = 45, 50, 2000
    else:
        pstr = 'Px1H'
        yx_f, yx_m, yx_fm = 45, 35, 1250
    props = {
        'EF_GR': {'title': 'Event Frequency (Annual)', 'ylbl': 'F [1/yr]',
                  'bref': r'$\mathrm{F^{GR}_{Ref}}$', 'bcc': r'$\mathrm{F^{GR}_{CC}}$',
                  'btitle': f'{pstr}-p95WAS-F', 'unit': '1/yr', 'yn': 0, 'yx': yx_f, 'dy': 5,
                  'af': r'$\mathcal{A}_\mathrm{CC}^\mathrm{F}$'},
        'EMavg_GR': {'title': 'Average Exceedance Magnitude (daily-median)', 'ylbl': 'M [mm]',
                     'bref': r'$\mathrm{M^{GR}_{Ref}}$', 'bcc': r'$\mathrm{M^{GR}_{CC}}$',
                     'btitle': f'{pstr}-p95WAS-M', 'unit': 'mm', 'yn': 0, 'yx': yx_m, 'dy': 5,
                     'af': r'$\mathcal{A}_\mathrm{CC}^\mathrm{M}$'},
        'FM_GR': {'title': 'Compound Frequency-Magnitude (Annual)', 'ylbl': 'FM [mm/yr]',
                  'bref': r'$\mathrm{FM^{GR}_{Ref}}$', 'bcc': r'$\mathrm{FM^{GR}_{CC}}$',
                  'btitle': f'{pstr}-p95WAS-FM', 'unit': 'mm/yr', 'yn': 0, 'yx': yx_fm, 'dy': 250,
                  'af': r'$\mathcal{A}_\mathrm{CC}^\mathrm{FM}$'}}

    return props[var]


def plot_timeseries(fig, axs, reg, data, pvar):
    col = '#6baed6'
    if reg == 'AUT':
        col = '#08519c'

    # define xvals
    xvals = np.arange(1941, 2024)
    ref_yrs = np.arange(1961, 1991)
    cc_yrs = np.arange(2010, 2023)

    ref_vals, cc_vals = {}, {}
    for ivar, vvar in enumerate(['EF_GR', 'EMavg_GR', 'FM_GR']):
        # plot data
        axs[ivar].plot(xvals, data[vvar], 'o-', markersize=3, color=col)

        # calc REF and CC vals
        ref = gmean(data[vvar].sel(ctp=slice('1965-01-01', '1985-12-31')))
        cc = gmean(data[vvar].sel(ctp=slice('2015-01-01', '2019-12-31')))
        ref_vals[vvar], cc_vals[vvar] = ref, cc
        axs[ivar].plot(ref_yrs, np.ones(len(ref_yrs)) * ref, color=col, alpha=0.5,
                       linewidth=2)
        axs[ivar].plot(cc_yrs, np.ones(len(cc_yrs)) * cc, color=col, alpha=0.5,
                       linewidth=2)

        props = get_props(var=vvar, pvar=pvar)
        axs[ivar].text(0.25, (
                (ref - props['yn']) / (props['yx'] - props['yn'])) + 0.04,
                       props['bref'],
                       horizontalalignment='left',
                       verticalalignment='center', transform=axs[ivar].transAxes,
                       fontsize=9)

        axs[ivar].text(0.92, ((cc - props['yn']) / (
                props['yx'] - props['yn'])) + 0.04,
                       props['bcc'],
                       horizontalalignment='left',
                       verticalalignment='center', transform=axs[ivar].transAxes,
                       fontsize=9)

    return ref_vals, cc_vals


def set_plot_props(axs, vvars, ref, cc, pvar):
    for ivar, vvar in enumerate(vvars):
        props = get_props(var=vvar, pvar=pvar)

        axs[ivar].tick_params(axis='both', labelsize=10)
        axs[ivar].minorticks_on()
        axs[ivar].grid(color='lightgray', which='major', linestyle=':')
        axs[ivar].set_xlabel('Time (core year of decadal-mean value)', fontsize=12)
        axs[ivar].set_ylabel(props['ylbl'], fontsize=12)
        axs[ivar].set_ylim(0, props['yx'])
        axs[ivar].yaxis.set_major_locator(MultipleLocator(props['dy']))

        axs[ivar].set_xlim(1940, 2026)
        axs[ivar].xaxis.set_minor_locator(MultipleLocator(2))
        axs[ivar].xaxis.set_major_locator(MultipleLocator(10))
        axs[ivar].set_title(props['title'], fontsize=12)

        axs[ivar].text(0.02, 0.87, props['btitle'] + r'$_\mathrm{Ref | CC}$ '
                       + f'({props["af"]})' + '\n'
                       + f'AUT: {ref["AUT"][vvar]:.2f}' + r'$\,$|$\,$'
                       + f'{cc["AUT"][vvar]:.2f} {props["unit"]} '
                         f'({cc["AUT"][vvar] / ref["AUT"][vvar]:.2f})\n'
                       + f'SEA: {ref["SEA"][vvar]:.2f}' + r'$\,$|$\,$'
                       + f'{cc["SEA"][vvar]:.2f} {props["unit"]} '
                         f'({cc["SEA"][vvar] / ref["SEA"][vvar]:.2f})',
                       horizontalalignment='left', verticalalignment='center',
                       transform=axs[ivar].transAxes, backgroundcolor='whitesmoke', fontsize=10)


def run():
    pvar = 'Px1h_7to7'
    fig, axs = plt.subplots(1, 3, figsize=(16, 4.5))

    regions = ['AUT', 'SEA']
    ref_vals, cc_vals = {}, {}
    for reg in regions:
        data = xr.open_dataset(f'/data/users/hst/additional_stuff/heavyrainfall_Haslinger2025/'
                               f'data/TEA/DEC_{reg}_{pvar}.nc')
        ref, cc = plot_timeseries(fig, axs, reg=reg, data=data, pvar=pvar)
        ref_vals[reg], cc_vals[reg] = ref, cc

    set_plot_props(axs=axs, vvars=['EF_GR', 'EMavg_GR', 'FM_GR'], ref=ref_vals, cc=cc_vals,
                   pvar=pvar)

    aleg, = axs[0].plot([-9, -9], 'o-', markersize=3, color='#6baed6')
    sleg, = axs[0].plot([-9, -9], 'o-', markersize=3, color='#08519c')
    fig.legend((aleg, sleg), ('AUT', 'SEA'), ncol=2, loc=(0.45, 0.03))

    fig.subplots_adjust(bottom=0.25, left=0.05, right=0.95, wspace=0.2)

    plt.savefig(f'/nas/home/hst/work/cdrDPS/plots/misc/Annual_FM_{pvar}.png',
                bbox_inches='tight', dpi=300)


if __name__ == '__main__':
    run()
