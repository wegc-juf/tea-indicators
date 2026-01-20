"""
Creates Figure 1 panels c, d, e.
"""
import cartopy.crs as ccrs
import cartopy.feature as cfea
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import glob
from pathlib import Path
import matplotlib.ticker as mticker
import matplotlib.pyplot as plt
import matplotlib.patches as pat
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
from shapely import geometry
import warnings
import xarray as xr

from plot_EDF1 import get_lims

# Input data paths; modify as needed
ERA5LAND_PATH = Path('/data/arsclisys/normal/ERA5_land/')
STATIC_PATH = Path('/data/arsclisys/normal/clim-hydro/TEA-Indicators/static/')
MASKS_PATH = Path('/data/arsclisys/normal/clim-hydro/TEA-Indicators/masks/')


def calc_percentile(_data):
    percentile = 0.99

    perc = _data.quantile(percentile, dim='time', skipna=True)

    return perc


def plot_map(fig, ax, data, levels, cbar_ticks, ext):
    perc = ax.contourf(data, cmap='Reds', levels=levels, extend=ext)

    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)

    cb = fig.colorbar(perc, cax=cax, orientation='vertical', extend=ext)
    cb.ax.tick_params(labelsize=14)
    cb.set_label('Ref-p99ANN Temperature (°C)', fontsize=16)

    ax.axis('off')


def load_data_1c(file, var):
    """
    loads land sea mask for ERA5 data and prepares it
    :return: lsm
    """

    data_raw = xr.open_dataset(file)

    data_e = data_raw.sel(longitude=slice(180.1, 360))
    data_w = data_raw.sel(longitude=slice(0, 180))
    data_values = np.concatenate((data_e[var].values[0, :, :], data_w[var].values[0, :, :]),
                                 axis=1)

    lsm_lon = np.arange(-180, 180, 0.1)

    lsm = xr.DataArray(data=data_values, dims=('lat', 'lon'), coords={
        'lon': (['lon'], lsm_lon), 'lat': (['lat'], data_raw.latitude.values)})

    return lsm


def plot_fig1c():
    lsm = load_data_1c(file=ERA5LAND_PATH / 'invariants' / 'lsm_1279l4_0.1x0.1.grb_v4_unpack.nc', var='lsm')
    lsm = lsm.where(lsm > 0.5)
    lsm = lsm.where(lsm.isnull(), 1)

    z = load_data_1c(file=ERA5LAND_PATH / 'invariants' / 'geo_1279l4_0.1x0.1.grib2_v4_unpack.nc', var='z')
    altitude = z / 9.80665

    data = altitude * lsm
    data = data.sel(lat=slice(72, 34), lon=slice(-13, 41))

    fig = plt.figure(figsize=(10, 7))
    proj = ccrs.LambertConformal(central_longitude=13.5, central_latitude=53.5, cutoff=30)
    axs = plt.axes(projection=proj)
    gl = axs.gridlines(crs=ccrs.PlateCarree(), draw_labels=False, color='black', linestyle=':')
    gl.xlocator = mticker.FixedLocator([-10, 0, 10, 20, 30, 40])
    gl.ylocator = mticker.FixedLocator([35, 45, 55, 65, 75])
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    axs.contourf(data.lon, data.lat, data, transform=ccrs.PlateCarree(), cmap='Greys')
    axs.add_feature(cfea.BORDERS)
    axs.coastlines()
    axs.set_extent([-12, 40, 30, 75])

    for reg in ['SAF', 'IBE', 'SCN']:
        lat_lim, lon_lim, center = get_lims(param='T', reg=reg)
        geom = geometry.box(minx=lon_lim[0], maxx=lon_lim[1], miny=lat_lim[0], maxy=lat_lim[1])
        axs.add_geometries([geom], crs=ccrs.PlateCarree(),
                           edgecolor='black', facecolor='None', linewidth=1)
        axs.scatter(center[0], center[1], marker='x', color='black', s=15,
                    transform=ccrs.PlateCarree())

    axs.text(0.605, 0.685, 'SCN', horizontalalignment='left', verticalalignment='center',
             transform=axs.transAxes, fontsize=8, rotation=10)
    axs.text(0.545, 0.3, 'SAF', horizontalalignment='left', verticalalignment='center',
             transform=axs.transAxes, fontsize=8)
    axs.text(0.14, 0.17, 'IBE', horizontalalignment='left', verticalalignment='center',
             transform=axs.transAxes, fontsize=8, rotation=-10)

    axs.text(0.28, 0.82, 'EUR', horizontalalignment='left', verticalalignment='center',
             transform=axs.transAxes, fontsize=10, rotation=-10)

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

    coords2 = []
    for ilon in np.arange(-12, 41.5, 0.5):
        coords2.append((ilon, 34))
    for ilat in np.arange(34, 72.5, 0.5):
        coords2.append((41, ilat))
    for ilon in np.arange(-12, 41.5, 0.5)[::-1]:
        coords2.append((ilon, 72))
    for ilat in np.arange(34, 72.5, 0.5)[::-1]:
        coords2.append((-12, ilat))

    geom_eur2 = geometry.Polygon(coords2)
    axs.add_geometries([geom_eur2], crs=ccrs.PlateCarree(),
                       edgecolor='tab:red', facecolor='None', linewidth=1.5)

    # add horizontal lines
    for ieur in [45, 55, 70]:
        hl = []
        for ilon in np.arange(-11, 40.5, 0.5):
            hl.append((ilon, ieur))
        geom_hl = geometry.LineString(hl)
        axs.add_geometries([geom_hl], crs=ccrs.PlateCarree(), edgecolor='black',
                           facecolor='None', linewidth=1, linestyle='--')

    axs.text(0.12, 0.28, 'S-EUR', horizontalalignment='left', verticalalignment='center',
             transform=axs.transAxes, fontsize=8, rotation=-10)
    axs.text(0.15, 0.39, 'C-EUR', horizontalalignment='left', verticalalignment='center',
             transform=axs.transAxes, fontsize=8, rotation=-10)
    axs.text(0.24, 0.69, 'N-EUR', horizontalalignment='left', verticalalignment='center',
             transform=axs.transAxes, fontsize=8, rotation=-10)

    axs.axis('off')
    fig.savefig('./Figure1c.png', dpi=300, bbox_inches='tight')
    # plt.show()


def plot_fig1d():
    warnings.filterwarnings(action='ignore', message='All-NaN slice encountered')

    fig, axs = plt.subplots(1, 1, figsize=(10, 6))

    data = xr.open_dataset(STATIC_PATH / 'static_Tx99.0p_AUT_SPARTACUS.nc')
    t99 = data.threshold

    mask = xr.open_dataset(MASKS_PATH / 'AUT_masks_SPARTACUS.nc')

    lvls = np.arange(19, 34)

    axs.contourf(mask.nw_mask, cmap='Greys')

    plot_map(fig=fig, ax=axs, data=t99.values, levels=lvls, cbar_ticks=np.arange(19, 33, 3),
             ext='neither')

    axs.add_patch(pat.Rectangle(xy=(473, 73), height=20, width=25, edgecolor='black',
                                fill=False, linewidth=2))
    axs.add_patch(pat.Rectangle(xy=(410, 45), height=92, width=125, edgecolor='black',
                                fill=False, linewidth=2))

    axs.axis('off')
    fig.savefig('./Figure1d.png', dpi=300, bbox_inches='tight')


def plot_fig1e():
    data = xr.open_dataset(STATIC_PATH / 'static_Tx99.0p_FBR_SPARTACUS.nc')
    t99 = data.threshold

    lims = t99.where(t99.notnull(), drop=True)
    t99 = t99.sel(x=lims.x, y=lims.y)

    lvls = np.arange(29, 31.25, 0.25)

    if np.nanmin(t99) < lvls[0] and np.nanmax(t99) < lvls[-1]:
        ext = 'min'
    elif np.nanmin(t99) > lvls[0] and np.nanmax(t99) > lvls[-1]:
        ext = 'max'
    elif np.nanmin(t99) < lvls[0] and np.nanmax(t99) > lvls[-1]:
        ext = 'both'
    else:
        ext = 'neither'

    fig, axs = plt.subplots(1, 1, figsize=(10, 6))
    plot_map(fig=fig, ax=axs, data=t99.values, levels=lvls, cbar_ticks=np.arange(26, 31), ext=ext)

    axs.set_xlim(0, 23)
    axs.set_ylim(0, 18)

    axs.add_patch(pat.Rectangle(xy=(0, 0), height=18, width=23, edgecolor='black',
                                fill=False, linewidth=2))

    fig.savefig('./Figure1e.png', dpi=300, bbox_inches='tight')


def run():
    plot_fig1c()
    plot_fig1d()
    plot_fig1e()


if __name__ == '__main__':
    run()
