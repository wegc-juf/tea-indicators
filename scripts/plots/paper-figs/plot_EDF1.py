"""
Plot script for ExtDataFig1
@Author: hst
@revised by: juf 2024-09-23
"""

import cartopy.crs as ccrs
import cartopy.feature as cfea
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import glob
import matplotlib.patches as pat
import matplotlib.ticker as mticker
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
import seaborn as sns
from shapely import geometry
import xarray as xr


def load_data(perc, temp=False):
    if temp:
        temp = '_T'
    else:
        temp = ''

    path = '/data/arsclisys/backup/clim-hydro/gcci_ewm/actea_input/static/thresholds/'
    file = 'threshold_grid_AUT_{perc}percentile{temp}_SPARTACUSreg.npy'.format(perc=perc,
                                                                               temp=temp)

    data = np.load(path + file)

    return data


def get_lims(_param, _reg):
    """
    set limits of target region
    :param _reg: SAF, SCN, or IBE
    :return: lims for lat and lon and center of target region
    """
    # check ok

    if _param == 'T':
        fac = 1
    else:
        fac = 0.5

    if _reg == 'SAF':
        center = [15.5, 47]
    elif _reg == 'SCN':
        center = [26, 62]
    else:
        center = [-6, 38]

    lat_lim = [center[1] - fac, center[1] + fac]
    lon_lim = [center[0] - (fac / np.cos(np.deg2rad(center[1]))),
               center[0] + (fac / np.cos(np.deg2rad(center[1])))]

    return lat_lim, lon_lim, center


def load_lsm():
    """
    loads land sea mask for ERA5 data and prepares it
    :return: lsm
    """

    data = xr.open_dataset('/data/users/hst/cdrDPS/ERA5/ERA5_altitude.nc')
    data = data.altitude[0, :, :]

    lsm_raw = xr.open_dataset('/data/users/hst/cdrDPS/ERA5/ERA5_LSM.nc')

    lsm_e = lsm_raw.sel(longitude=slice(180.25, 360))
    lsm_w = lsm_raw.sel(longitude=slice(0, 180))
    lsm_values = np.concatenate((lsm_e.lsm.values[0, :, :], lsm_w.lsm.values[0, :, :]),
                                axis=1)

    lsm_lon = np.arange(-180, 180, 0.25)

    lsm = xr.DataArray(data=lsm_values, dims=('lat', 'lon'), coords={
        'lon': (['lon'], lsm_lon), 'lat': (['lat'], lsm_raw.latitude.values)})

    lsm = lsm.sel(lat=data.lat.values, lon=data.lon.values)

    lsm = lsm.where(lsm > 0.5)
    lsm = lsm.where(lsm.isnull(), 1)

    return lsm


def plot_map(_fig, _ax, _data, _levels, _temp=False):
    if _temp:
        colmap = 'Reds'
        label = 'Temperature (°C)'
    else:
        colmap = 'Blues'
        label = 'Precipitation (mm)'

    perc = _ax.contourf(_data, cmap=colmap, levels=_levels)

    divider = make_axes_locatable(_ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    cb = _fig.colorbar(perc, cax=cax, orientation='vertical')
    cb.set_label(label)

    _ax.axis('off')


def plot_eur_thresh():
    """
    ExtDataFig 1a
    :return:
    """
    thr = xr.open_dataset('/data/arsclisys/normal/clim-hydro/TEA-Indicators/static/'
                          'static_Tx99.0p_EUR_ERA5.nc')

    thr = thr.threshold
    thr = thr.sel(lat=slice(72, 35), lon=slice(-10, 40))

    levels = np.arange(10, 42.5, 2.5)
    cmap = sns.color_palette('Reds', len(levels))

    fig = plt.figure(figsize=(10, 7))
    proj = ccrs.LambertConformal(central_longitude=13.5, central_latitude=53.5, cutoff=30)
    axs = plt.axes(projection=proj)

    im = thr.plot.imshow(ax=axs, transform=ccrs.PlateCarree(), colors=cmap, vmin=10, vmax=40,
                         levels=levels, cbar_kwargs={'label': 'Ref-p99ANN Temperature (°C)',
                                                     'ticks': np.arange(10, 45, 5)})
    axs.set_title('')

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

    for reg in ['SAF', 'IBE', 'SCN']:
        lat_lim, lon_lim, center = get_lims(_param='T', _reg=reg)
        geom = geometry.box(minx=lon_lim[0], maxx=lon_lim[1], miny=lat_lim[0], maxy=lat_lim[1])
        axs.add_geometries([geom], crs=ccrs.PlateCarree(),
                           edgecolor='black', facecolor='None', linewidth=1.5)
        axs.scatter(center[0], center[1], marker='x', color='black', s=15,
                    transform=ccrs.PlateCarree())

    axs.text(0.595, 0.69, 'SCN', horizontalalignment='left', verticalalignment='center',
             transform=axs.transAxes, fontsize=8, rotation=10)
    axs.text(0.527, 0.305, 'SAF', horizontalalignment='left', verticalalignment='center',
             transform=axs.transAxes, fontsize=8)
    axs.text(0.11, 0.18, 'IBE', horizontalalignment='left', verticalalignment='center',
             transform=axs.transAxes, fontsize=8, rotation=-10)

    coords = []
    for ilon in np.arange(-11, 40.5, 0.5):
        coords.append((ilon, 35))
    for ilat in np.arange(35, 71.5, 0.5):
        coords.append((40, ilat))
    for ilon in np.arange(-11, 40.5, 0.5)[::-1]:
        coords.append((ilon, 71))
    for ilat in np.arange(35, 71.5, 0.5)[::-1]:
        coords.append((-11, ilat))

    geom_eur = geometry.Polygon(coords)
    axs.add_geometries([geom_eur], crs=ccrs.PlateCarree(),
                       edgecolor='black', facecolor='None', linewidth=1.5)
    axs.text(0.25, 0.82, 'EUR', horizontalalignment='left', verticalalignment='center',
             transform=axs.transAxes, fontsize=12, rotation=-10)

    # add horizontal lines
    for ieur in [45, 55, 70]:
        hl = []
        for ilon in np.arange(-11, 40.5, 0.5):
            hl.append((ilon, ieur))
        geom_hl = geometry.LineString(hl)
        axs.add_geometries([geom_hl], crs=ccrs.PlateCarree(), edgecolor='black',
                           facecolor='None', linewidth=1, linestyle='--')

    axs.text(0.09, 0.28, 'S-EUR', horizontalalignment='left', verticalalignment='center',
             transform=axs.transAxes, fontsize=8, rotation=-10)
    axs.text(0.12, 0.39, 'C-EUR', horizontalalignment='left', verticalalignment='center',
             transform=axs.transAxes, fontsize=8, rotation=-10)
    axs.text(0.21, 0.7, 'N-EUR', horizontalalignment='left', verticalalignment='center',
             transform=axs.transAxes, fontsize=8, rotation=-10)

    axs.text(0.02, 0.97, 'ERA5-TMax-Ref1961-1990', horizontalalignment='left',
             verticalalignment='center', transform=axs.transAxes, backgroundcolor='whitesmoke',
             fontsize=10)

    plt.savefig('/nas/home/hst/work/cdrDPS/plots/01_paper_figures/ExtDataFigs/panels/EDF1/'
                'ExtDataFig1a.png',
                bbox_inches='tight', dpi=300)


def plot_aut_era5land():
    """
    ExtDataFig 1 b
    :return:
    """
    thr = xr.open_dataset('/data/arsclisys/normal/clim-hydro/TEA-Indicators/static/'
                          'static_Tx99.0p_AUT_ERA5Land.nc')

    aut = xr.open_dataset('/data/arsclisys/normal/clim-hydro/TEA-Indicators/masks/'
                          'AUT_masks_ERA5Land.nc')

    thr = thr.where(aut.nw_mask == 1)

    fig = plt.figure(figsize=(5, 3))
    proj = ccrs.LambertConformal(central_longitude=13.5, central_latitude=53.5, cutoff=30)
    axs = plt.axes(projection=proj)
    axs.contourf(aut.lon, aut.lat, aut.nw_mask, colors='gainsboro', transform=ccrs.PlateCarree())
    vals = axs.contourf(thr.lon, thr.lat, thr.threshold, levels=np.arange(19, 34, 1),
                        transform=ccrs.PlateCarree(), cmap='Reds')
    axs.add_feature(cfea.BORDERS)
    axs.coastlines()

    cb = plt.colorbar(vals, pad=0.03, shrink=0.84, label='Ref-p99ANN Temperature (°C)')

    gl = axs.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, color='black', linestyle=':',
                       x_inline=False, y_inline=False)
    gl.top_labels = False
    gl.right_labels = False
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    gl.rotate_labels = False

    # add FBR and SEA boxes
    geom = geometry.box(minx=14.8, maxx=16.5, miny=46.6, maxy=47.4)
    axs.add_geometries([geom], crs=ccrs.PlateCarree(), edgecolor='black', facecolor='None',
                       linewidth=1)
    axs.text(0.75, 0.2, 'FBR', horizontalalignment='left', verticalalignment='center',
             transform=axs.transAxes, fontsize=8)
    geom2 = geometry.box(minx=15.7, maxx=16.1, miny=46.8, maxy=47)
    axs.add_geometries([geom2], crs=ccrs.PlateCarree(), edgecolor='black', facecolor='None',
                       linewidth=1)
    axs.text(0.701, 0.33, 'SEA', horizontalalignment='left', verticalalignment='center',
             transform=axs.transAxes, fontsize=8)

    axs.text(0.02, 0.55, 'AUT', horizontalalignment='left', verticalalignment='center',
             transform=axs.transAxes, fontsize=10)

    axs.set_extent([9.4, 17.1, 46.3, 49.1])

    fig.subplots_adjust(bottom=0.1, top=0.9, left=0.05, right=0.95)

    fig.text(0.06, 0.205, 'Alpine data at z > 1500m excluded.',
             horizontalalignment='left', verticalalignment='center',
             backgroundcolor='gainsboro',
             fontsize=6)

    axs.text(0.02, 0.93, 'ERA5L-TMax-Ref1961-1990', horizontalalignment='left',
             verticalalignment='center', transform=axs.transAxes, backgroundcolor='whitesmoke',
             fontsize=10)

    plt.savefig('/nas/home/hst/work/cdrDPS/plots/01_paper_figures/ExtDataFigs/panels/EDF1/'
                'ExtDataFig1b.png', bbox_inches='tight', dpi=300)


def plot_sea_spartacus():
    """
    ExtDataFig 1 c & d
    :return:
    """
    param = ['Tx', 'P24h_7to7']

    props = {'Tx': {'levels': np.arange(19, 34, 1), 'cb_lbl': 'Ref-p99ANN Temperature (°C)',
                    'cmap': 'Reds', 'ext': 'neither', 'pstr': 'Tx99.0p'},
             'P24h_7to7': {'levels': np.arange(18, 44, 2),
                           'cb_lbl': 'Ref-p95WAS Precipitation (mm)', 'cmap': 'Blues',
                           'ext': 'neither', 'pstr': 'P24h_7to7_95.0p'}}

    for par in param:
        thr = xr.open_dataset(f'/data/arsclisys/normal/clim-hydro/TEA-Indicators/static/'
                              f'static_{props[par]["pstr"]}_SEA_SPARTACUS.nc')
        thr = thr.threshold

        fig, axs = plt.subplots(1, 1, figsize=(4.5, 3))
        perc = axs.contourf(thr, cmap=props[par]['cmap'], levels=props[par]['levels'],
                            extend=props[par]['ext'])

        divider = make_axes_locatable(axs)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        cb = fig.colorbar(perc, cax=cax, orientation='vertical', extend=props[par]['ext'],
                          shrink=0.83, ticks=props[par]['levels'][::2])
        cb.set_label(props[par]['cb_lbl'])

        axs.add_patch(pat.Rectangle(xy=(470, 74), height=20, width=25, edgecolor='black',
                                    fill=False, linewidth=1))
        axs.add_patch(pat.Rectangle(xy=(402, 42), height=97, width=135, edgecolor='black',
                                    fill=False, linewidth=1))

        axs.set_xlim(401, 539)
        axs.set_ylim(40, 142)
        if par == 'Tx':
            title = 'SPCUS-TMax-Ref1961-1990'
        else:
            title = 'SPCUS-P24H-Ref1961-1990'
        axs.set_title(title, fontsize=12)

        axs.axis('off')

        panels = {'Tx': 'c', 'P24h_7to7': 'd'}

        plt.savefig(f'/nas/home/hst/work/cdrDPS/plots/01_paper_figures/ExtDataFigs/panels/EDF1/'
                    f'ExtDataFig1{panels[par]}.png',
                    bbox_inches='tight', dpi=300)
        plt.close()


if __name__ == '__main__':
    plot_eur_thresh()
    # plot_aut_era5land()
    # plot_sea_spartacus()
