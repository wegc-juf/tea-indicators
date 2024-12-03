import cartopy.crs as ccrs
import cartopy.feature as cfea
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import seaborn as sns
import xarray as xr


def create_cmap_tex():
    cmax = 35
    cmap = plt.cm.Reds
    cmaplist = [cmap(i) for i in range(cmap.N)]
    col_idx = np.arange(np.floor(256 / (cmax / 2.5)), 256, np.floor((256 / (cmax / 2.5))))
    cmaplist = [col for icol, col in enumerate(cmaplist) if icol in col_idx]
    cmaplist = [element for element in cmaplist for _ in range(5)]
    cmaplist[0] = (0.619, 0.792, 0.870, 1.0)
    cmaplist[1] = (0.619, 0.792, 0.870, 1.0)
    cmaplist = cmaplist[:-10]
    cmax = 30
    cmap = mpl.colors.LinearSegmentedColormap.from_list('Custom cmap',
                                                        cmaplist, cmax * 2)

    return cmap


def create_cmap():
    cbar_step = 0.25
    cmax = 4

    ncolors = int(np.floor((cmax - 0.5) / cbar_step))
    color_steps = int(np.floor(256 / ncolors))

    cmap = plt.cm.Reds
    cmaplist = [cmap(i) for i in range(cmap.N)]
    col_idx = np.arange(color_steps, 256, color_steps)
    cmaplist = [col for icol, col in enumerate(cmaplist) if icol in col_idx]

    cmap_lt1 = plt.cm.Blues_r
    norm_vals = np.linspace(0, 1, int(1 / cbar_step))
    for iblue in range(int(0.5 / cbar_step)):
        cmaplist[iblue] = cmap_lt1(norm_vals[iblue])

    cmaplist = cmaplist[:-2]
    ncolors = ncolors - 2

    cmap = mpl.colors.LinearSegmentedColormap.from_list('Custom cmap',
                                                        cmaplist, ncolors)

    return cmap


def scale_figsize(figwidth, figheight, figdpi):
    """
    scale figsize to certain dpi
    :param figwidth: width in inch
    :param figheight: height in inch
    :param figdpi: desired dpi
    :return:
    """
    # factor for making fonts larger
    enlarge_fonts = 1.4

    scaling_factor = 2 / enlarge_fonts

    width = figwidth * scaling_factor
    height = figheight * scaling_factor
    dpi = figdpi / scaling_factor

    return width, height, dpi


def fig4a():
    data = xr.open_dataset('/data/users/hst/TEA-clean/TEA/amplification/'
                           'AF_Tx99.0p_AGR-EUR_WAS_ERA5_1961to2022.nc')
    vkeep = ['EF_AF_CC', 'EDavg_AF_CC', 'EMavg_AF_CC', 'EAavg_AF_CC', 'TEX_AF_CC']
    vdrop = [vvar for vvar in data.data_vars if vvar not in vkeep]
    data = data.drop_vars(vdrop)

    cmap_tex = create_cmap_tex()
    cmax_tex = 30

    cmap = create_cmap()
    cmax = 3.5

    cb_lbl = {'TEX_AF_CC': 'ERA5-TMax-p99ANN-TEX' + r'$_\mathrm{CC}$ amplification '
                                                    r'($\mathcal{A}_\mathrm{CC}^\mathrm{T}$)',
              'EF_AF_CC': 'ERA5-TMax-p99ANN-EF' + r'$_\mathrm{CC}$ amplification '
                                                  r'($\mathcal{A}_\mathrm{CC}^\mathrm{F}$)',
              'EDavg_AF_CC': 'ERA5-TMax-p99ANN-ED' + r'$_\mathrm{CC}$ amplification '
                                                     r'($\mathcal{A}_\mathrm{CC}^\mathrm{D}$)',
              'EMavg_AF_CC': 'ERA5-TMax-p99ANN-EM' + r'$_\mathrm{CC}$ amplification '
                                                     r'($\mathcal{A}_\mathrm{CC}^\mathrm{M}$)',
              'EAavg_AF_CC': 'ERA5-TMax-p99ANN-EA' + r'$_\mathrm{CC}$ amplification '
                                                     r'($\mathcal{A}_\mathrm{CC}^\mathrm{A}$)'}

    for vvar in data.data_vars:
        fw, fh, dpi = scale_figsize(figwidth=10, figheight=7, figdpi=300)
        fig = plt.figure(figsize=(fw, fh), dpi=dpi)
        proj = ccrs.LambertConformal(central_longitude=13.5, central_latitude=53.5, cutoff=30)
        axs = plt.axes(projection=proj)
        if vvar == 'TEX_AF_CC':
            im = data[vvar].plot.imshow(ax=axs, transform=ccrs.PlateCarree(), cmap=cmap_tex,
                                        vmin=0, vmax=cmax_tex, add_colorbar=False)
            fig_str = 'Fig4'
            cx = cmax_tex
            dc = 5
        else:
            im = data[vvar].plot.imshow(ax=axs, transform=ccrs.PlateCarree(), cmap=cmap,
                                        vmin=0.5, vmax=cmax, add_colorbar=False)
            cx = cmax
            dc = 0.5
            fig_str = 'EDF7'
        ext = 'neither'
        if data[vvar].max() > cx:
            ext = 'max'
        cb_ticks = list(np.arange(0, cx + dc, dc))
        cb_ticks.insert(1, 1)
        cb = plt.colorbar(im, pad=0.03, ticks=cb_ticks, extend=ext)
        cb.set_label(label=cb_lbl[vvar], fontsize=10)
        cb.ax.tick_params(labelsize=8)

        axs.add_feature(cfea.BORDERS)
        axs.coastlines()
        gl = axs.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, color='black', linestyle=':',
                           x_inline=False, y_inline=False)
        gl.top_labels = False
        gl.right_labels = False
        gl.xlocator = mticker.FixedLocator([-10, 0, 10, 20, 30, 40])
        gl.ylocator = mticker.FixedLocator([35, 45, 55, 65, 75])
        gl.xformatter = LONGITUDE_FORMATTER
        gl.yformatter = LATITUDE_FORMATTER
        axs.set_extent([-10, 40, 30, 75])
        axs.tick_params(axis='both', which='major', labelsize=10)

        plt.title(f'{vvar}', fontsize=14)

        vstr = vvar.split('_')[0]
        plt.savefig(f'/nas/home/hst/work/TEAclean/plots/paper-figs/{fig_str}_{vstr}.png',
                    dpi=300)


def fig4c():
    regs = ['EUR', 'S-EUR', 'C-EUR', 'N-EUR']
    cols = {'EUR': 'tab:grey',
            'S-EUR': sns.color_palette('Spectral', n_colors=10)[2],
            'C-EUR': sns.color_palette('RdYlGn', n_colors=10)[7],
            'N-EUR': sns.color_palette('Spectral', n_colors=10)[8]}

    fig, axs = plt.subplots(1, 1, figsize=(7, 3.5))
    xticks = np.arange(1960, 2023)

    for ireg, reg in enumerate(regs):
        data = xr.open_dataset(f'/data/users/hst/TEA-clean/TEA/amplification/'
                               f'AF_Tx99.0p_AGR-{reg}_WAS_ERA5_1961to2022.nc')
        data = data['TEX_AGR_AF']

        axs.plot(xticks[1:], data, color=cols[reg], linewidth=2, markersize=3,
                 label=reg)

    axs.set_title('Total Events Extremity (TEX) amplification | EUR & Europe regions',
                  fontsize=12)
    axs.set_xlabel('Time (core year of decadal-mean value)', fontsize=10)
    axs.set_ylabel(r'TEX amplification ($\mathcal{A}_\mathrm{CC}^\mathrm{T}$)',
                   fontsize=10)
    axs.minorticks_on()
    axs.grid(color='gray', which='major', linestyle=':')
    axs.set_xlim(1960, 2023)
    axs.set_ylim(0, 25)
    axs.xaxis.set_minor_locator(mticker.FixedLocator(np.arange(1960, 2023)))
    axs.legend()

    plt.show()


if __name__ == '__main__':
    # fig4a()
    fig4c()