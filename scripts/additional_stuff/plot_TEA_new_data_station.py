import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import numpy as np
from scipy.stats import gmean
import xarray as xr


def get_props(var, pvar):
    if pvar == 'P24h_7to7':
        pstr = 'P24H'
        yx_f, yx_d, yx_m, yx_tex = 8, 1.6, 30, 140
        dy_f, dy_d, dy_m, dy_tex = 2, 0.2, 5, 20
    else:
        pstr = 'Px1H'
        yx_f, yx_d, yx_m, yx_tex = 10, 1.6, 10, 50
        dy_f, dy_d, dy_m, dy_tex = 2, 0.2, 2, 10
    props = {
        'EF': {'title': 'Event Frequency (Annual)', 'ylbl': 'F [1/yr]',
               'bref': r'$\mathrm{F_{Ref}}$', 'bcc': r'$\mathrm{F_{CC}}$',
               'btitle': f'{pstr}-p95WAS-F', 'unit': '1/yr', 'yn': 0, 'yx': yx_f, 'dy': dy_f,
               'af': r'$\mathcal{A}_\mathrm{CC}^\mathrm{F}$'},
        'EDavg': {'title': 'Average Event Duration (events-mean)', 'ylbl': 'D [days]',
                  'bref': r'$\mathrm{D_{Ref}}$', 'bcc': r'$\mathrm{D_{CC}}$',
                  'btitle': f'{pstr}-p95WAS-D', 'unit': 'days', 'yn': 0.8, 'yx': yx_d, 'dy': dy_d,
                  'af': r'$\mathcal{A}_\mathrm{CC}^\mathrm{D}$'},
        'EMavg': {'title': 'Average Exceedance Magnitude (daily-median)', 'ylbl': 'M [mm]',
                  'bref': r'$\mathrm{M_{Ref}}$', 'bcc': r'$\mathrm{M_{CC}}$',
                  'btitle': f'{pstr}-p95WAS-M', 'unit': 'mm', 'yn': 0, 'yx': yx_m, 'dy': dy_m,
                  'af': r'$\mathcal{A}_\mathrm{CC}^\mathrm{M}$'},
        'tEX': {'title': 'Temporal Events Extremity (Annual)', 'ylbl': 'tEX [mm days/yr]',
                'bref': r'$\mathrm{tEX_{Ref}}$', 'bcc': r'$\mathrm{tEX_{CC}}$',
                'btitle': f'{pstr}-p95WAS-tEX', 'unit': 'mm days/yr', 'yn': 0, 'yx': yx_tex,
                'dy': dy_tex, 'af': r'$\mathcal{A}_\mathrm{CC}^\mathrm{tEX}$'}}

    return props[var]


def plot_timeseries(fig, axs, reg, data, pvar):
    # define xvals
    xvals = np.arange(1941, 2024)
    ref_yrs = np.arange(1961, 1991)
    cc_yrs = np.arange(2010, 2023)

    alpha = 1
    eyr = 2017
    if reg == 'AUT':
        alpha = 0.3
        eyr = 2018

    colors = {'EF': 'tab:blue', 'EDavg': 'tab:purple', 'EMavg': 'tab:orange', 'tEX': 'tab:orange'}

    ref_vals, cc_vals = {'AUT': {}, 'SEA': {}}, {'AUT': {}, 'SEA': {}}
    for ivar, vvar in enumerate(['EF', 'EDavg', 'EMavg', 'tEX']):
        # plot data
        axs[ivar].plot(xvals, data[vvar], color=colors[vvar], alpha=alpha)

        if reg == 'AUT':
            axs[ivar].plot(xvals, data[vvar].sel(stations='gsa_38'), color='tab:grey', alpha=0.9,
                           linewidth=1.5, linestyle='--')
            ref_sea = gmean(
                data[vvar].sel(ctp=slice('1965-01-01', '1985-12-31'), stations='gsa_38'))
            cc_sea = gmean(
                data[vvar].sel(ctp=slice('2015-01-01', f'2017-12-31'), stations='gsa_38'))
            ref_vals['SEA'][vvar], cc_vals['SEA'][vvar] = ref_sea, cc_sea

            axs[ivar].plot(xvals, data[vvar].mean(dim='stations'), color=colors[vvar],
                           linewidth=1.5)
            data[vvar] = data[vvar].mean(dim='stations')

        # calc REF and CC vals
        ref = gmean(data[vvar].sel(ctp=slice('1965-01-01', '1985-12-31')))
        cc = gmean(data[vvar].sel(ctp=slice('2015-01-01', f'{eyr}-12-31')))
        ref_vals[reg][vvar], cc_vals[reg][vvar] = ref, cc
        axs[ivar].plot(ref_yrs, np.ones(len(ref_yrs)) * ref, color=colors[vvar],
                       linewidth=2)
        axs[ivar].plot(cc_yrs, np.ones(len(cc_yrs)) * cc, color=colors[vvar],
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


def set_plot_props(axs, vvars, ref, cc, pvar, reg):
    for ivar, vvar in enumerate(vvars):
        props = get_props(var=vvar, pvar=pvar)

        axs[ivar].tick_params(axis='both', labelsize=10)
        axs[ivar].minorticks_on()
        axs[ivar].grid(color='lightgray', which='major', linestyle=':')
        axs[ivar].set_xlabel('Time (core year of decadal-mean value)', fontsize=12)
        axs[ivar].set_ylabel(props['ylbl'], fontsize=12)
        axs[ivar].set_ylim(props['yn'], props['yx'])
        axs[ivar].yaxis.set_major_locator(MultipleLocator(props['dy']))

        axs[ivar].set_xlim(1940, 2026)
        axs[ivar].xaxis.set_minor_locator(MultipleLocator(2))
        axs[ivar].xaxis.set_major_locator(MultipleLocator(10))
        axs[ivar].set_title(props['title'], fontsize=12)

        if reg == 'SEA':
            try:
                axs[ivar].text(0.02, 0.87, props['btitle'] + r'$_\mathrm{Ref | CC}$ '
                               + f'({props["af"]})' + '\n'
                               + f'{reg}: {ref[reg][vvar]:.2f}' + r'$\,$|$\,$'
                               + f'{cc[reg][vvar]:.2f} {props["unit"]} '
                               + f'({cc[reg][vvar] / ref[reg][vvar]:.2f})',
                               horizontalalignment='left', verticalalignment='center',
                               transform=axs[ivar].transAxes, backgroundcolor='whitesmoke',
                               fontsize=10)
            except TypeError:
                axs[ivar].text(0.02, 0.87, props['btitle'] + r'$_\mathrm{Ref | CC}$ '
                               + f'({props["af"]})' + '\n'
                               + f'{reg}: {ref[reg][vvar][0]:.2f}' + r'$\,$|$\,$'
                               + f'{cc[reg][vvar][0]:.2f} {props["unit"]} '
                               + f'({cc[reg][vvar][0] / ref[reg][vvar][0]:.2f})',
                               horizontalalignment='left', verticalalignment='center',
                               transform=axs[ivar].transAxes, backgroundcolor='whitesmoke',
                               fontsize=10)
        else:
            axs[ivar].text(0.02, 0.85, props['btitle'] + r'$_\mathrm{Ref | CC}$ '
                           + f'({props["af"]})' + '\n'
                           + f'AUT: {ref["AUT"][vvar]:.2f}' + r'$\,$|$\,$'
                           + f'{cc["AUT"][vvar]:.2f} {props["unit"]} '
                           + f'({cc["AUT"][vvar] / ref["AUT"][vvar]:.2f})' + '\n'
                           + f'SEA: {ref["SEA"][vvar]:.2f}' + r'$\,$|$\,$'
                           + f'{cc["SEA"][vvar]:.2f} {props["unit"]} '
                           + f'({cc["SEA"][vvar] / ref["SEA"][vvar]:.2f})',
                           horizontalalignment='left', verticalalignment='center',
                           transform=axs[ivar].transAxes, backgroundcolor='whitesmoke', fontsize=10)


def run():
    pvar = 'P24h_7to7'
    region = 'AUT'
    fig, axs = plt.subplots(2, 2, figsize=(12, 8))
    axs = axs.reshape(-1)

    data = xr.open_dataset(f'/data/users/hst/additional_stuff/heavyrainfall_Haslinger2025/'
                           f'data/TEA/DEC_{region}_{pvar}.nc')
    ref, cc = plot_timeseries(fig, axs, reg=region, data=data, pvar=pvar)

    set_plot_props(axs=axs, vvars=['EF', 'EDavg', 'EMavg', 'tEX'], ref=ref, cc=cc,
                   pvar=pvar, reg=region)

    if region == 'AUT':
        aleg, = axs[0].plot([-9, -9], color='tab:grey', alpha=0.3)
        sleg, = axs[0].plot([-9, -9], color='tab:grey', alpha=0.9, linewidth=1.5, linestyle='--')
        mleg, = axs[0].plot([-9, -9], color='tab:grey', linewidth=1.5)
        fig.legend((mleg, aleg, sleg), ('AUT-mean', 'AUT', 'SEA'), ncol=3, loc=(0.37, 0.02))

    fig.subplots_adjust(bottom=0.15, left=0.05, right=0.95, wspace=0.2, hspace=0.4)

    plt.savefig(f'/nas/home/hst/work/cdrDPS/plots/misc/Annual_stationTEA_{region}_{pvar}.png',
                bbox_inches='tight', dpi=300)


if __name__ == '__main__':
    run()
