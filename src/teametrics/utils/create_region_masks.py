#!/usr/bin/env python3
"""
create region masks for TEA indicator calculation
author: hst
"""

import geopandas as gpd
import numpy as np
from pathlib import Path
from shapely.geometry import Polygon, MultiPolygon
from tqdm import trange
import xarray as xr

from ..common.general_functions import create_history_from_cfg, get_gridded_data
from ..common.config import load_opts
from ..calc_TEA import _getopts


def _load_shp(opts):
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
            shp = shp[(shp.CNTR_ID == opts.region)]
        except AttributeError:
            try:
                shp = shp[(shp.LAND_NAME == opts.region)]
            except AttributeError:
                raise AttributeError('The given shape file has neither CNTR_ID nor '
                                     'LAND_NAME information.')

    # Transfer it to the wanted coordinate system
    shp = shp.to_crs(epsg=opts.target_sys)

    return shp


def _create_cell_polygons(opts, xvals, yvals, offset):
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

    path = Path(opts.maskpath) / opts.mask_sub / 'polygons' / f'{out_region}_EPSG{opts.target_sys}_{opts.dataset}/'
    path.mkdir(parents=True, exist_ok=True)
    fname = path / f'{out_region}_cells_EPSG{opts.target_sys}_{opts.dataset}.shp'

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


def create_sea_mask(opts):
    """
    create SEA mask (part of SAR that's within AUT)
    Args:
        opts: Options as defined in config file

    Returns:

    """

    try:
        aut = xr.open_dataset(Path(opts.maskpath) / opts.mask_sub / f'AUT_mask_{opts.dataset}.nc')
        sar = xr.open_dataset(Path(opts.maskpath) / opts.mask_sub / f'SAR_mask_{opts.dataset}.nc')
    except FileNotFoundError:
        raise FileNotFoundError('For SEA mask, run create_region_masks.py for AUT and SAR first.')

    mask = aut['mask'].where(sar['mask'].notnull())
    mask = mask.rename('mask')
    mask.attrs = {'long_name': 'weighted mask', 'coordinate_sys': f'EPSG:{opts.target_sys}'}
    ds = mask.to_dataset()

    create_history_from_cfg(cfg_params=opts, ds=ds)

    _save_output(ds, opts)


def _prep_lsm(opts):
    """
    load Land-Sea-Mask and convert coordinates
    Args:
        opts: CLI parameter

    Returns:
        lsm: land sea mask
    """
    lsm_raw = xr.open_dataset(opts.lsmfile)

    data = xr.open_dataset(opts.orofile)
    data = data.altitude
    
    # get lat resolution
    lat_resolution = round(abs(data.lat.values[0] - data.lat.values[1]), 4)
    
    step = lat_resolution

    # split to eastern and western hemisphere (from 0 ... 360 to -180 .. 180)
    lsm_e = lsm_raw.sel(longitude=slice(180 + step, 360))
    lsm_w = lsm_raw.sel(longitude=slice(0, 180))
    lsm_values = np.concatenate((lsm_e.lsm.values[0, :, :], lsm_w.lsm.values[0, :, :]), axis=1)

    lsm_lon = np.arange(-180, 180, step).astype('float32')
    lsm_lat = lsm_raw.latitude.values

    lsm = xr.DataArray(data=lsm_values, dims=('lat', 'lon'), coords={
        'lon': (['lon'], lsm_lon), 'lat': (['lat'], lsm_lat)})
    
    if opts.dataset == 'ERA5Land':
        lsm = lsm.interp(lon=(np.arange(-1800, 1800, step * 10) / 10),
                         lat=(np.arange(-900, 900, step * 10) / 10)[::-1])

    lsm = lsm.sel(lat=data.lat.values, lon=data.lon.values)

    return lsm


def create_agr_mask(opts):
    """
    create mask for AGRs
    Args:
        opts: CLI parameter

    Returns:

    """
    if 'ERA5' not in opts.dataset:
        raise AttributeError('AGR mask can only be created for ERA5(Land) data.')

    mask = _prep_lsm(opts=opts)
    mask = mask.where(mask > opts.land_frac_min)
    mask = mask.rename('mask')
    mask.attrs = {'long_name': 'weighted mask', 'coordinate_sys': f'EPSG:{opts.target_sys}'}

    # apply altitude threshold if set
    if opts.altitude_threshold != 0:
        mask = _apply_altitude_threshold(mask, opts)
    ds = mask.to_dataset()
    
    create_history_from_cfg(cfg_params=opts, ds=ds)

    _save_output(ds, opts)


def _apply_altitude_threshold(mask, opts):
    """
    apply altitude threshold to mask
    Args:
        mask: mask DataArray
        opts: Config parameters

    Returns:
        mask: mask DataArray with altitude threshold applied

    """
    # load orography
    orog = xr.open_dataset(opts.orofile)
    if 'altitude' in orog.data_vars:
        orog = orog.altitude
    elif 'elevation' in orog.data_vars:
        orog = orog.elevation
    else:
        orog = orog.orog

    mask = mask.where(orog < opts.altitude_threshold)
    return mask


def _find_closest(coords, corner_val, direction):
    """
    Find the closest value in sorted_list to the target with a given direction.
    direction=-1 means the closest on the left (smaller than target)
    direction=1 means the closest on the right (larger than target)
    """
    if direction == 1:
        for i in range(len(coords)):
            if coords[i] > corner_val:
                return coords[i]
        return coords[-1]  # If no larger value found, return the last value
    elif direction == -1:
        for i in reversed(range(len(coords))):
            if coords[i] < corner_val:
                return coords[i]
        return coords[0]  # If no smaller value found, return the first value
    else:
        raise ValueError('Direction must be either -1 or 1.')


def _save_output(ds, opts, out_region=None):
    """
    save output to netcdf files
    Args:
        ds: dataset
        opts: options
        out_region: region acronym (default: opts.region)

    Returns:

    """
    if out_region is None:
        out_region = opts.region
    outpath = Path(opts.maskpath) / opts.mask_sub / f'{out_region}_mask_{opts.dataset}.nc'
    print(f'Saving mask file to {outpath}')
    ds.to_netcdf(outpath)


def create_rectangular_gr(opts):
    """
    create rectangular grid mask using either corners or center coordinates + extent
    Args:
        opts: options as defined in config file

    Returns:

    """
    # load template file
    template_file = get_gridded_data(opts.start, opts.start + 1, opts)
    xy = opts.xy_name.split(',')
    x, y = xy[0], xy[1]
    dx = template_file[x][1] - template_file[x][0]
    dy = abs(template_file[y][1] - template_file[y][0])

    # get corners from CFG file
    if opts.gr_type == 'corners':
        sw_coords = opts.sw_corner.split(',')
        ne_coords = opts.ne_corner.split(',')
        sw_coords = [float(ii) for ii in sw_coords]
        ne_coords = [float(ii) for ii in ne_coords]
    elif opts.gr_type == 'center':
        center_coords = opts.center.split(',')
        center_coords = [float(ii) for ii in center_coords]
        sw_coords = [center_coords[0] - float(opts.we_len) / 2, center_coords[1] - float(opts.ns_len) / 2]
        ne_coords = [center_coords[0] + float(opts.we_len) / 2, center_coords[1] + float(opts.ns_len) / 2]
    else:
        raise ValueError('gr_type must be either "polygon", "corners", or "center".')

    if 'ERA5' in opts.dataset:
        yidxn, yidxx = -1, 0
    else:
        yidxn, yidxx = 0, -1

    xn, xx = sw_coords[0], ne_coords[0]
    yn, yx = sw_coords[1], ne_coords[1]

    # check if corners are within grid
    if any(xv < template_file[x][0] for xv in [xn, xx]) or any(xv > template_file[x][-1] for xv in [xn, xx]):
        raise KeyError('Passed corner(s) are outside of target grid!')
    if any(yv < template_file[y][yidxn] for yv in [yn, yx]) or any(yv > template_file[y][yidxx] for yv in [yn, yx]):
        raise KeyError('Passed corner(s) are outside of target grid!')

    # create non weighted mask array
    nw_mask_arr = np.full((len(template_file[y]), len(template_file[x])), np.nan)
    da_nwmask = xr.DataArray(data=nw_mask_arr, coords={y: ([y], template_file[y].data),
                                                       x: ([x], template_file[x].data)},
                             attrs={'long_name': 'non weighted mask',
                                    'coordinate_sys': f'EPSG:{opts.target_sys}'},
                             name='nw_mask')

    # check if corners are identical with grid points on target grid
    xvals_check = all(xv in template_file[x] for xv in [xn, xx])
    yvals_check = all(yv in template_file[y] for yv in [yn, yx])

    # set values in non-weighted mask within GR to 1 and create weighted mask
    if xvals_check and yvals_check:
        da_nwmask.loc[yn:yx, xn:xx] = 1
        da_mask = da_nwmask.copy()
        da_mask = da_mask.rename('mask')
        da_mask.attrs['long_name'] = 'non weighted mask'
    else:
        # Find the closest x and y for the corners and calculate fractions of cell area
        if 'ERA5' in opts.dataset:
            closest_sw_y = _find_closest(template_file[y][::-1], yn, direction=1)
            closest_ne_y = _find_closest(template_file[y][::-1], yx, direction=-1)
            s_frac = (closest_sw_y - yn) / dy
            n_frac = (yx - closest_ne_y) / dy
        else:
            closest_sw_y = _find_closest(template_file[y], yn, direction=-1)
            closest_ne_y = _find_closest(template_file[y], yx, direction=1)
            s_frac = (yn - closest_sw_y) / dy
            n_frac = (closest_ne_y - yx) / dy
        closest_sw_x = _find_closest(template_file[x], xn, direction=-1)
        closest_ne_x = _find_closest(template_file[x], xx, direction=1)
        w_frac = (xn - closest_sw_x) / dx
        e_frac = (closest_ne_x - xx) / dx

        # set values in non-weighted mask within GR to 1
        if 'ERA5' in opts.dataset:
            da_nwmask.loc[closest_ne_y:closest_sw_y, closest_sw_x:closest_ne_x] = 1
        else:
            da_nwmask.loc[closest_sw_y:closest_ne_y, closest_sw_x:closest_ne_x] = 1

        # create weighted mask
        da_mask = da_nwmask.copy()
        da_mask = da_mask.rename('mask')
        da_mask.attrs['long_name'] = 'non weighted mask'

        # apply fractions to mask to get weighted mask
        da_mask.loc[:, closest_sw_x] = da_mask.loc[:, closest_sw_x] * w_frac
        da_mask.loc[closest_sw_y, :] = da_mask.loc[closest_sw_y, :] * s_frac
        da_mask.loc[:, closest_ne_x] = da_mask.loc[:, closest_ne_x] * e_frac
        da_mask.loc[closest_ne_y, :] = da_mask.loc[closest_ne_y, :] * n_frac

    if opts.altitude_threshold != 0:
        da_mask = _apply_altitude_threshold(da_mask, opts)
    ds_mask = da_mask.to_dataset()

    create_history_from_cfg(cfg_params=opts, ds=ds_mask)

    out_region = f'SW_{xn}_{yn}-NE_{xx}_{yx}'
    _save_output(ds_mask, opts, out_region)


def create_mask_file(opts):
    """
    create mask file for given region
    Args:
        opts: Options as defined in config file

    Returns:

    """
    # Load template file
    template_file = get_gridded_data(opts.start, opts.start + 1, opts)
    xy = opts.xy_name.split(',')
    x, y = xy[0], xy[1]

    # Load shp file and transform it to desired coordinate system
    shp = _load_shp(opts=opts)
    # Define the cell grid
    xvals, yvals = template_file[x], template_file[y]

    # Get grid spacing
    # Coordinates of ERA5(Land) have some precision trouble
    if opts.dataset in ['ERA5', 'ERA5Land']:
        dx = set(abs(np.round(xvals[1:].values - xvals[:-1].values, 2)))
        dy = set(abs(np.round(yvals[1:].values - yvals[:-1].values, 2)))
    else:
        dx = set(abs(xvals[1:].values - xvals[:-1].values))
        dy = set(abs(yvals[1:].values - yvals[:-1].values))
    if len(dx) > 1 or len(dy) > 1 or dx != dy:
        raise ValueError('The given test file does not have a regular grid. '
                         'Provide a file with a regular grid.')
    dx, dy = list(dx)[0], list(dy)[0]
    offset = dx / 2

    # Initialize mask array
    mask = np.zeros(shape=(len(yvals), len(xvals)), dtype='float32')

    if len(shp) == 1:
        pass
    elif 'CNTR_ID' in shp.columns:
        shp = shp[shp.CNTR_ID == opts.region]
    elif 'LAND_NAME' in shp.columns:
        shp = shp[shp.LAND_NAME == opts.region]
    elif 'GEM_NAME' in shp.columns:
        shp = shp[shp.GEM_NAME == opts.region]

    poly = shp.geometry.iloc[0]

    cells = _create_cell_polygons(opts=opts, xvals=xvals, yvals=yvals, offset=offset)

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
    if opts.altitude_threshold != 0:
        da_mask = _apply_altitude_threshold(da_mask, opts)
        
    ds_mask = da_mask.to_dataset()
    create_history_from_cfg(cfg_params=opts, ds=ds_mask)
    out_region = opts.region
    _save_output(ds_mask, opts, out_region)


def run():
    cmd_opts = _getopts()
    
    # load CFG parameter
    opts = load_opts(fname=__file__, config_file=cmd_opts.config_file)
    
    if opts.gr_type != 'polygon':
        create_rectangular_gr(opts=opts)
    elif opts.region == 'SEA':
        create_sea_mask(opts=opts)
    elif opts.region in ['EUR', 'AFR']:
        create_agr_mask(opts=opts)
    else:
        create_mask_file(opts)


if __name__ == '__main__':
    run()
