"""
create region masks for TEA indicator calculation
author: hst
"""

import argparse
import geopandas as gpd
import numpy as np
import os
from pathlib import Path
from shapely.geometry import Polygon, MultiPolygon
import sys
from tqdm import trange
import xarray as xr

from scripts.general_stuff.general_functions import create_history


def get_opts():
    """
    get CLI parameter

    Returns:
        myopts: CLI parameter

    """

    def file(entry):
        if os.path.isfile(entry):
            return entry
        else:
            raise argparse.ArgumentTypeError(f'{entry} is not a valid file')

    parser = argparse.ArgumentParser()

    parser.add_argument('--region', default='AUT', type=str,
                        help='Region for mask creation.')

    parser.add_argument('--subreg',
                        type=str,
                        help='Optional: Only necessary if selected region is not the entire region '
                             'in the shp file (Austrian states, european countries etc.). '
                             'In case of Austrian states, give name of state. '
                             'In case of european country, give ISO2 code of country.')

    parser.add_argument('--target_sys',
                        default=3416,
                        type=int,
                        help='ID of wanted coordinate System (https://epsg.io) which should be '
                             'used for mask. Default: 3416 (ETRS89 / Austria Lambert).')

    parser.add_argument('--target_ds',
                        default='SPARTACUS',
                        type=str,
                        help='Dataset for which mask should be created. Default: SPARTACUS')

    parser.add_argument('--xy_name',
                        type=str,
                        default='x,y',
                        help='Names of x and y coordinates in testfile, separated by ",". '
                             'Default: x,y')

    parser.add_argument('--shpfile',
                        type=file,
                        help='Shape file of region.')

    parser.add_argument('--testfile',
                        type=file,
                        default='/data/arsclisys/normal/clim-hydro/TEA-Indicators/SPARTACUS/'
                                'SPARTACUS-DAILY_Tx_1961.nc',
                        help='File with coordinate information of target grid.')

    parser.add_argument('--orofile',
                        type=file,
                        default='/data/arsclisys/normal/clim-hydro/TEA-Indicators/SPARTACUS/'
                                'SPARTACUSreg_orography.nc',
                        help='File with orography information of target grid.')

    parser.add_argument('--lsmfile',
                        type=file,
                        default='/data/users/hst/cdrDPS/ERA5/ERA5_LSM.nc',
                        help='File with land sea mask of target grid. Only necessary if mask for '
                             'EUR should be created.')

    parser.add_argument('--outpath',
                        dest='outpath',
                        default='/data/arsclisys/normal/clim-hydro/TEA-Indicators/masks/',
                        help='Path of folder where output data should be saved.')

    myopts = parser.parse_args()

    return myopts


def load_shp(opts):
    """
    load shp file for given region
    Args:
        opts: CLI parameter

    Returns:
        shp: geopandas df of shp file

    """

    # Load shp file
    shp = gpd.read_file(opts.shpfile)

    if opts.subreg:
        try:
            shp = shp[(shp.CNTR_ID == opts.subreg)]
        except AttributeError:
            try:
                shp = shp[(shp.LAND_NAME == opts.subreg)]
            except AttributeError:
                raise AttributeError('The given shape file has neither CNTR_ID nor '
                                     'LAND_NAME information.')

    # Transfer it to the wanted coordinate system
    shp = shp.to_crs(epsg=opts.target_sys)

    return shp


def create_cell_polygons(opts, xvals, yvals, offset):
    """
    create list of polygons for each cell
    Args:
        opts: CLI parameter
        xvals: x-coordinates
        yvals: y-coordinates
        offset: half of grid spacing

    Returns:
        cells: list of cell polygons

    """

    out_region = opts.region
    if opts.subreg:
        out_region = opts.subreg

    path = Path(f'{opts.outpath}polygons/{out_region}_EPSG{opts.target_sys}_{opts.target_ds}/')
    path.mkdir(parents=True, exist_ok=True)
    fname = f'{path}/{out_region}_cells_EPSG{opts.target_sys}_{opts.target_ds}.shp'

    try:
        gdf = gpd.read_file(fname)
        cells = []
        for idx, row in gdf.iterrows():
            cell = {'ix': row['ix'], 'iy': row['iy'], 'geometry': row['geometry']}
            cells.append(cell)
    except:
        cells_list = []
        for ix in trange(len(yvals) - 1, desc='Creating polygons for individual cells'):
            for iy in range(len(xvals) - 1):
                cell = Polygon(
                    [(xvals[iy] - offset, yvals[ix] - offset),
                     (xvals[iy] + offset, yvals[ix] - offset),
                     (xvals[iy] + offset, yvals[ix] + offset),
                     (xvals[iy] - offset, yvals[ix] + offset),
                     (xvals[iy] - offset, yvals[ix] - offset)])
                cells_list.append((ix, iy, cell))

        gdf = gpd.GeoDataFrame(cells_list, columns=['ix', 'iy', 'geometry'])
        gdf.to_file(fname, driver='ESRI Shapefile')

        # Load gdf from file otherwise index error later
        gdf = gpd.read_file(fname)
        cells = []
        for idx, row in gdf.iterrows():
            cell = {'ix': row['ix'], 'iy': row['iy'], 'geometry': row['geometry']}
            cells.append(cell)

    return cells


def create_lt1500m_mask(opts, da_nwmask):
    """
    create mask of all cells with an altitude lower than 1500m
    Args:
        opts: CLI parameter
        da_nwmask: non weightes mask da

    Returns:
        lt1500_mask: lower than 1500m mask da

    """
    orog = xr.open_dataset(opts.orofile)
    if 'altitude' in orog.data_vars:
        orog = orog.altitude
    else:
        orog = orog.orog

    lt1500_mask = da_nwmask.copy()
    lt1500_mask = lt1500_mask.where(orog < 1500)
    lt1500_mask = lt1500_mask.rename('lt1500_mask')
    lt1500_mask.attrs = {'long_name': 'below 1500m mask',
                         'coordinate_sys': f'EPSG:{opts.target_sys}'}

    lt1500_eur = None
    if 'ERA5' in opts.target_ds:
        lt1500_eur = orog.where(orog < 1500)
        lt1500_eur = lt1500_eur.where(lt1500_eur.isnull(), 1)
        lt1500_eur = lt1500_eur.rename('lt1500_mask_EUR')
        lt1500_eur.attrs = {'long_name': 'below 1500m mask (EUR)',
                             'coordinate_sys': f'EPSG:{opts.target_sys}'}

    return lt1500_mask, lt1500_eur


def run_sea(opts):
    """
    create SEA mask (part of SAR that's within AUT)
    Args:
        opts: CLI parameter

    Returns:

    """

    try:
        aut = xr.open_dataset(f'{opts.outpath}AUT_masks_{opts.target_ds}.nc')
        sar = xr.open_dataset(f'{opts.outpath}SAR_masks_{opts.target_ds}.nc')
    except FileNotFoundError:
        raise FileNotFoundError('For SEA mask, run create_region_masks.py for AUT and SAR first.')

    mask = aut['mask'].where(sar['nw_mask'].notnull())
    mask = mask.rename('mask')
    mask.attrs = {'long_name': 'weighted mask', 'coordinate_sys': f'EPSG:{opts.target_sys}'}

    nwmask = mask.copy()
    nwmask = nwmask.where(mask.isnull(), 1)
    nwmask = nwmask.rename('nw_mask')
    nwmask.attrs = {'long_name': 'non weighted mask', 'coordinate_sys': f'EPSG:{opts.target_sys}'}

    lt1500_mask = sar['lt1500_mask'].where(aut['lt1500_mask'].notnull())
    lt1500_mask.attrs = {'long_name': 'below 1500m mask',
                         'coordinate_sys': f'EPSG:{opts.target_sys}'}

    if 'ERA5' in opts.target_ds:
        lsm = aut['LSM_EUR'].copy()
        lt1500_eur = aut['lt1500_mask_EUR'].copy()
        ds = xr.merge([mask, nwmask, lt1500_mask, lsm, lt1500_eur])
    else:
        ds = xr.merge([mask, nwmask, lt1500_mask])
    ds = create_history(cli_params=sys.argv, ds=ds)

    ds.to_netcdf(f'{opts.outpath}{opts.region}_masks_{opts.target_ds}.nc')


def prep_lsm(opts):
    """
    load LSM and convert coordinates
    Args:
        opts: CLI parameter

    Returns:
        lsm: land sea mask
    """
    lsm_raw = xr.open_dataset(opts.lsmfile)

    data = xr.open_dataset(opts.orofile)
    data = data.altitude

    lsm_e = lsm_raw.sel(longitude=slice(180.25, 360))
    lsm_w = lsm_raw.sel(longitude=slice(0, 180))
    lsm_values = np.concatenate((lsm_e.lsm.values[0, :, :], lsm_w.lsm.values[0, :, :]),
                                axis=1)

    lsm_lon = np.arange(-180, 180, 0.25)

    lsm = xr.DataArray(data=lsm_values, dims=('lat', 'lon'), coords={
            'lon': (['lon'], lsm_lon), 'lat': (['lat'], lsm_raw.latitude.values)})

    lsm = lsm.sel(lat=data.lat.values, lon=data.lon.values)

    return lsm


def run_eur(opts):
    """
    create EUR mask
    Args:
        opts: CLI parameter

    Returns:

    """

    if opts.target_ds != 'ERA5':
        raise AttributeError('EUR mask can only be created for ERA5 data.')

    # load LSM and only keep cells with more than 50% land in them
    lsm = prep_lsm(opts=opts)
    lsm = lsm.where(lsm > 0.5)

    # create weighted mask
    mask = lsm.copy()
    mask = mask.where(mask > 0)
    mask = mask.rename('mask')
    mask.attrs = {'long_name': 'weighted mask', 'coordinate_sys': f'EPSG:{opts.target_sys}'}

    # create non weighted mask
    nwmask = mask.copy()
    nwmask = nwmask.where(mask.isnull(), 1)
    nwmask = nwmask.rename('nw_mask')
    nwmask.attrs = {'long_name': 'non weighted mask', 'coordinate_sys': f'EPSG:{opts.target_sys}'}

    # load orography
    orog = xr.open_dataset(opts.orofile)
    orog = orog.altitude
    # create below 1500 m mask
    lt1500_mask = nwmask.copy()
    lt1500_mask = lt1500_mask.where(orog < 1500)
    lt1500_mask = lt1500_mask.rename('lt1500_mask')
    lt1500_mask.attrs = {'long_name': 'below 1500m mask',
                         'coordinate_sys': f'EPSG:{opts.target_sys}'}

    ds = xr.merge([mask, nwmask, lt1500_mask])
    ds = create_history(cli_params=sys.argv, ds=ds)

    ds.to_netcdf(f'{opts.outpath}{opts.region}_masks_{opts.target_ds}.nc')


def run():
    opts = get_opts()

    if opts.region == 'SEA':
        run_sea(opts=opts)
    elif opts.region == 'EUR':
        run_eur(opts=opts)
    else:
        # Load dummy file
        dummy = xr.open_dataset(opts.testfile)
        xy = opts.xy_name.split(',')
        x, y = xy[0], xy[1]

        # Load shp file and transform it to desired coordinate system
        shp = load_shp(opts=opts)

        # Define the cell grid
        xvals, yvals = dummy[x], dummy[y]

        # Get grid spacing
        # Coordinates of ERA5(Land) have some precision trouble
        if opts.target_ds in ['ERA5', 'ERA5Land']:
            dx = set(abs(np.round(xvals[1:].values - xvals[:-1].values, 2)))
            dy = set(abs(np.round(yvals[1:].values - yvals[:-1].values, 2)))
        else:
            dx = set(abs(xvals[1:].values - xvals[:-1].values))
            dy = set(abs(yvals[1:].values - yvals[:-1].values))
        if len(dx) > 1 or len(dx) > 1 or dx != dy:
            raise ValueError('The given test file does not have a regular grid. '
                             'Provide a file with a regular grid.')
        dx, dy = list(dx)[0], list(dy)[0]
        offset = dx / 2

        # Initialize mask array
        mask = np.zeros(shape=(len(yvals), len(xvals)), dtype='float32')

        # The following part is very sensible to the shape file that is used.
        # Lots of trial and error here...
        geom = shp.geometry.iloc[0]
        if not opts.subreg:
            if isinstance(geom, MultiPolygon):
                poly = geom.geoms[1]
            else:
                poly = geom
        else:
            poly = geom

        cells = create_cell_polygons(opts=opts, xvals=xvals, yvals=yvals, offset=offset)

        # Check intersections and calculate mask values
        total_cells = len(cells)
        for i in trange(total_cells, desc='Calculating fractions for cells'):
            icell = cells[i]
            ix, iy, cell = icell['ix'], icell['iy'], icell['geometry']
            intersection = poly.intersection(cell)
            if not intersection.is_empty:
                mask[ix, iy] += intersection.area / cell.area

        # Set cells outside of region to nan
        mask[np.where(mask == 0)] = np.nan

        # Create non-weighted mask
        nw_mask = mask.copy()
        nw_mask[np.where(mask > 0)] = 1

        # Convert to da
        da_mask = xr.DataArray(data=mask, coords={y: ([y], yvals.data), x: ([x], xvals.data)},
                               attrs={'long_name': 'weighted mask',
                                      'coordinate_sys': f'EPSG:{opts.target_sys}'},
                               name='mask')
        da_nwmask = xr.DataArray(data=nw_mask, coords={y: ([y], yvals.data), x: ([x], xvals.data)},
                                 attrs={'long_name': 'non weighted mask',
                                        'coordinate_sys': f'EPSG:{opts.target_sys}'},
                                 name='nw_mask')

        # Create below 1500m mask
        lt1500_mask, lt1500_eur = create_lt1500m_mask(opts=opts, da_nwmask=da_nwmask)

        # add EUR LSM if ERA5(Land) data is used
        if 'ERA5' in opts.target_ds:
            lsm = prep_lsm(opts=opts)
            lsm = lsm.where(lsm > 0.5)
            lsm = lsm.rename('LSM_EUR')
            lsm.attrs = {'long_name': 'land sea mask (EUR)',
                         'coordinate_sys': f'EPSG:{opts.target_sys}'}
            ds = xr.merge([da_mask, da_nwmask, lt1500_mask, lsm, lt1500_eur])
        else:
            ds = xr.merge([da_mask, da_nwmask, lt1500_mask])
        ds = create_history(cli_params=sys.argv, ds=ds)

        out_region = opts.region
        if opts.subreg:
            out_region = opts.subreg

        ds.to_netcdf(f'{opts.outpath}{out_region}_masks_{opts.target_ds}.nc')


if __name__ == '__main__':
    run()
