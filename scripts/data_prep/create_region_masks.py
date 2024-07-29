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

sys.path.append('/home/hst/tea-indicators/scripts/misc/')
from general_functions import create_history


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
                        help='Optional: In case of Austrian states, give name of state. '
                             'In case of european country, give ISO3 code od country.')

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
                        help='File with coordinate information of target grid.')

    parser.add_argument('--outpath',
                        dest='outpath',
                        default='/data/users/hst/TEA-clean/masks/',
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

    """

    path = Path(f'{opts.outpath}polygons/{opts.region}_EPSG{opts.target_sys}_{opts.target_ds}/')
    path.mkdir(parents=True, exist_ok=True)
    fname = f'{path}/{opts.region}_cells_EPSG{opts.target_sys}_{opts.target_ds}.shp'

    try:
        gdf = gpd.read_file(fname)
        cells = []
        for idx, row in gdf.iterrows():
            cell = {'ix': row['ix'], 'iy': row['iy'], 'geometry': row['geometry']}
            cells.append(cell)
    except:
        cells = []
        for ix in trange(len(yvals) - 1, desc='Creating polygons for individual cells'):
            for iy in range(len(xvals) - 1):
                cell = Polygon(
                    [(xvals[iy] - offset, yvals[ix] - offset),
                     (xvals[iy] + offset, yvals[ix] - offset),
                     (xvals[iy] + offset, yvals[ix] + offset),
                     (xvals[iy] - offset, yvals[ix] + offset),
                     (xvals[iy] - offset, yvals[ix] - offset)])
                cells.append((ix, iy, cell))

        gdf = gpd.GeoDataFrame(cells, columns=['ix', 'iy', 'geometry'])
        gdf.to_file(fname, driver='ESRI Shapefile')

    return cells


def run():
    opts = get_opts()

    # Load dummy file
    dummy = xr.open_dataset(opts.testfile)
    xy = opts.xy_name.split(',')
    x, y = xy[0], xy[1]

    # Load shp file and transform it to desired coordinate system
    shp = load_shp(opts=opts)

    # Define the cell grid
    xvals, yvals = dummy[x], dummy[y]

    # Get grid spacing
    dx = set(xvals[1:].values - xvals[:-1].values)
    dy = set(yvals[1:].values - yvals[:-1].values)
    if len(dx) > 1 or len(dx) > 1 or dx != dy:
        raise ValueError('The given test file does not have a regular grid. '
                         'Provide a file with a rugular grid.')
    dx, dy = list(dx)[0], list(dy)[0]
    offset = dx / 2

    # Initialize mask array
    mask = np.zeros(shape=(len(yvals), len(xvals)), dtype='float32')

    geom = shp.geometry.iloc[0]
    if isinstance(geom, MultiPolygon):
        poly = geom.geoms[1]
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

    # Create output dataset
    da_mask = xr.DataArray(data=mask, coords={y: ([y], yvals.data), x: ([x], xvals.data)},
                           attrs={'long_name': 'weighted mask',
                                  'coordinate_sys': f'EPSG:{opts.target_sys}'},
                           name='mask')
    da_nwmask = xr.DataArray(data=nw_mask, coords={y: ([y], yvals.data), x: ([x], xvals.data)},
                             attrs={'long_name': 'non weighted mask',
                                    'coordinate_sys': f'EPSG:{opts.target_sys}'},
                             name='nw_mask')

    ds = xr.merge([da_mask, da_nwmask])
    ds = create_history(cli_params=sys.argv, ds=ds)

    ds.to_netcdf(f'{opts.outpath}{opts.region}_masks_{opts.target_ds}.nc')


if __name__ == '__main__':
    run()
