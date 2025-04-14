#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: hst
"""

import glob
import numpy as np
import os
import sys
import warnings
import xarray as xr

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from scripts.general_stuff.general_functions import create_history_from_cfg, load_opts, get_input_filenames, get_data


def area_grid(opts, masks):
    """
    creates grid where each grid cell gets assigned the size of each grid cell in
    areals (1 areal = 100 km^2)
    Args:
        opts: CLI parameter
        masks: masks

    Returns:
        agrid: area grid
    """

    if opts.dataset == 'SPARTACUS':
        # creates area grid in areals
        agrid = masks['nw_mask'] / 100

    else:
        if opts.dataset == 'ERA5':
            delta_fac = 4  # to get 0.25° resolution
        else:
            delta_fac = 10  # to get 0.1°

        lat = masks.lat.values
        r_mean = 6371
        u_mean = 2 * np.pi * r_mean

        # calculate earth radius at different latitudes
        r_lat = np.cos(np.deg2rad(lat)) * r_mean

        # calculate earth circumference at latitude
        u_lat = 2 * np.pi * r_lat

        # calculate length of 0.25°/0.1° in m for x and y dimension
        x_len = (u_lat / 360) / delta_fac
        y_len = (u_mean / 360) / delta_fac

        # calculate size of cells in areals
        x_len_da = xr.DataArray(data=x_len, coords={'lat': (['lat'], lat)})
        if opts.full_region:
            mask_template = xr.full_like(masks['nw_mask'], 1.)
        else:
            mask_template = masks['nw_mask']
        agrid = mask_template * y_len * x_len_da
        agrid = agrid / 100

    # apply GR mask
    if not opts.full_region:
        agrid = agrid.where(masks['lt1500_mask'] == 1)
        agrid = agrid * masks['mask']
    agrid = agrid.rename('area_grid')
    agrid.attrs = {'units': 'areals'}

    # calculate GR size
    gr_size = agrid.sum()
    gr_size = gr_size.rename('GR_size')
    gr_size.attrs = {'units': 'areals'}

    return agrid, gr_size


def calc_percentiles(opts, threshold_min=None):
    """
    calculate percentile of reference period for each grid point
    Args:
        opts: CLI parameter
        threshold_min: minimum data value for threshold calculation (e.g. 0.99 for precip); optional

    Returns:
        thresh: threshold (percentile) grid

    """

    data = get_data(start=opts.ref_period[0], end=opts.ref_period[1], opts=opts, period=opts.period)
    
    if threshold_min is not None:
        data = data.where(data > threshold_min)

    # calc the chosen percentile for each grid point as threshold
    percent = data.chunk(dict(time=-1)).quantile(q=opts.threshold / 100, dim='time')

    # smooth SPARTACUS precip percentiles (for each grid point calculate the average of all grid
    # points within the given radius)
    if 'smoothing' in opts:
        radius = opts.smoothing
    else:
        radius = 0

    if radius > 0:
        percent_smooth = smooth_data(percent, radius)
        percent = percent_smooth
    
    percent = percent.drop_vars('quantile')
    
    return percent


def smooth_data(data, radius):
    y_size, x_size = data.shape
    percent_smooth_arr = np.full_like(data.values, np.nan)
    percent_tmp = np.zeros((y_size + 2 * radius, x_size + 2 * radius),
                           dtype='float32') * np.nan
    percent_tmp[radius:radius + y_size, radius:radius + x_size] = data
    rad_circ = radius + 0.5
    x_vec = np.arange(0, x_size + 2 * radius)
    y_vec = np.arange(0, y_size + 2 * radius)
    iy_new = 0
    for iy in range(radius, y_size):
        ix_new = 0
        for ix in range(radius, x_size):
            circ_mask = (x_vec[np.newaxis, :] - ix) ** 2 + (y_vec[:, np.newaxis] - iy) ** 2 \
                        < rad_circ ** 2
            percent_smooth_arr[iy_new, ix_new] = np.nanmean(percent_tmp[circ_mask])
            ix_new += 1
        iy_new += 1
    percent_smooth = xr.full_like(data, np.nan)
    percent_smooth[:, :] = percent_smooth_arr
    return percent_smooth


def run():
    warnings.filterwarnings(action='ignore', message='All-NaN slice encountered')
    warnings.filterwarnings(action='ignore', message='Mean of empty slice')

    # load CFG parameter
    opts = load_opts(fname=__file__)

    # load GR masks
    masks = xr.open_dataset(f'{opts.maskpath}/{opts.mask_sub}/{opts.region}_masks_{opts.dataset}.nc')

    # create area grid
    area, gr_size = area_grid(opts=opts, masks=masks)

    # calculate thresholds
    if opts.threshold_type == 'abs':
        thr_grid = None
    else:
        print('Calculating percentiles...')
        if opts.precip:
            # TODO: get rid of precip option and use parameter instead
            threshold_min = 0.99
        else:
            threshold_min = None
        thr_grid = calc_percentiles(opts=opts, threshold_min=threshold_min)
        
        vname = f'{opts.parameter}-p{opts.threshold}ANN Ref1961-1990'
        if opts.precip:
            vname = f'{opts.parameter}-p{opts.threshold}WAS WetDOYs > 1 mm Ref1961-1990'
        
        thr_grid = thr_grid.rename('threshold')
        thr_grid.attrs = {'units': opts.unit, 'methods_variable_name': vname, 'percentile': f'{opts.threshold}p'}
        
        # apply GR mask
        if not opts.full_region:
            if 'ERA' in opts.dataset and opts.region != 'EUR' and 'agr' in opts:
                mask = masks['lt1500_mask_EUR']
            else:
                mask = masks['lt1500_mask'] * masks['mask']
                
            for dim in thr_grid.dims:
                if dim not in mask.dims:
                    continue
                if thr_grid[dim].dtype != mask[dim].dtype:
                    thr_grid[dim] = thr_grid[dim].astype(mask[dim].dtype)
            
            thr_grid = thr_grid.where(mask > 0)
        
    # combine to single dataset
    if thr_grid is not None:
        ds_out = xr.merge([area, gr_size, thr_grid], join='left')
    else:
        ds_out = xr.merge([area, gr_size], join='left')
    del ds_out.attrs['units']
    ds_out = create_history_from_cfg(cfg_params=opts, ds=ds_out)

    # add additional attributes
    ds_out.attrs['region'] = opts.region
    ds_out.attrs['dataset'] = opts.dataset
    ds_out.attrs['coordinate_sys'] = masks.attrs['coordinate_sys']

    # save output
    if opts.full_region:
        outname = f'{opts.outpath}static_{opts.param_str}_{opts.region}_{opts.dataset}_full.nc'
    else:
        outname = f'{opts.outpath}static_{opts.param_str}_{opts.region}_{opts.dataset}.nc'

    ds_out.to_netcdf(outname)


if __name__ == '__main__':
    run()
