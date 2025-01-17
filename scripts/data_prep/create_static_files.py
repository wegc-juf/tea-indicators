#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: hst
"""

import argparse
import glob
import numpy as np
import os
import sys
import warnings
import xarray as xr

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from scripts.general_stuff.general_functions import create_history, load_opts, extend_tea_opts


def get_opts():
    """
    get CLI parameter
    Returns:
        myopts: CLI parameter
    """

    def dir_path(path):
        if os.path.isdir(path):
            return path
        else:
            raise argparse.ArgumentTypeError(f'{path} is not a valid path!')

    parser = argparse.ArgumentParser()

    parser.add_argument('--parameter',
                        default='Tx',
                        type=str,
                        help='Parameter for which the TEA indices should be calculated'
                             '[default: Tx].')

    parser.add_argument('--precip',
                        action='store_true',
                        help='Set if chosen parameter is a precipitation parameter.')

    parser.add_argument('--season-length',
                        dest='season_length',
                        default=366,
                        type=int,
                        help='Number of days in season for threshold calculation. For whole year, '
                             'use 366 [default], for WAS (Apr-Oct) 214.')

    parser.add_argument('--threshold',
                        default=99,
                        type=float,
                        help='Threshold in degrees Celsius, mm, or as percentile [default: 99].')

    parser.add_argument('--smoothing',
                        default=0,
                        type=int,
                        help='Radius for spatial smoothing of threshold grid in km [default: 0].'
                             'Used for precipitation parameter from SPARTACUS data.')

    parser.add_argument('--threshold-type',
                        dest='threshold_type',
                        type=str,
                        choices=['perc', 'abs'],
                        default='perc',
                        help='Pass "perc" (default) if percentiles should be used as thresholds or '
                             '"abs" for absolute thresholds.')

    parser.add_argument('--unit',
                        default='degC',
                        type=str,
                        help='Physical unit of chosen parameter.')

    parser.add_argument('--region',
                        default='AUT',
                        type=str,
                        help='Geo region [options: AUT (default), Austrian state name, '
                             'or ISO2 code of european country].')

    parser.add_argument('--inpath',
                        dest='inpath',
                        default='/data/arsclisys/normal/clim-hydro/TEA-Indicators/SPARTACUS/',
                        type=dir_path,
                        help='Path of input data.')

    parser.add_argument('--maskpath',
                        dest='maskpath',
                        default='/data/arsclisys/normal/clim-hydro/TEA-Indicators/masks/',
                        type=dir_path,
                        help='Path of folder where GR masks are located.')

    parser.add_argument('--outpath',
                        dest='outpath',
                        default='/data/arsclisys/normal/clim-hydro/TEA-Indicators/static/',
                        type=dir_path,
                        help='Path of folder where static file should be saved.')

    parser.add_argument('--dataset',
                        dest='dataset',
                        default='SPARTACUS',
                        choices=['SPARTACUS', 'ERA5', 'ERA5Land'],
                        type=str,
                        help='Input dataset [default: SPARTACUS].')

    myopts = parser.parse_args()

    return myopts


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
        agrid = masks['nw_mask'] * y_len * x_len_da
        agrid = agrid / 100

    # apply GR mask
    agrid = agrid.where(masks['lt1500_mask'] == 1)
    agrid = agrid * masks['mask']
    agrid = agrid.rename('area_grid')
    agrid.attrs = {'units': 'areals'}

    # calculate GR size
    gr_size = agrid.sum()
    gr_size = gr_size.rename('GR_size')
    gr_size.attrs = {'units': 'areals'}

    return agrid, gr_size


def load_ref_data(opts, masks, ds_params, gr_size):
    """
    load and prepare data for percentile calculation
    Args:
        opts: CLI parameter
        masks: GR masks
        ds_params: dict woth dataset dependent dimension names
        gr_size: size of GR in areals

    Returns:
        data: data of reference period
    """

    dys = np.arange(0, opts.season_length)

    xn, yn = ds_params[opts.dataset]['xname'], ds_params[opts.dataset]['yname']

    ref_period = np.arange(1961, 1991)
    data_ref = xr.DataArray(data=np.zeros((len(ref_period), len(dys),
                                           len(masks[yn]), len(masks[xn])),
                                          dtype='float32') * np.nan,
                            coords={'year': (['year'], ref_period),
                                    'dys': (['dys'], dys),
                                    yn: ([yn], masks[yn].values),
                                    xn: ([xn], masks[xn].values)})

    param_str = ''
    if opts.dataset == 'SPARTACUS' and opts.precip:
        param_str = 'RR'
    elif opts.dataset == 'SPARTACUS' and not opts.precip:
        param_str = opts.parameter

    idx = 0
    for yr in ref_period:
        files = sorted(glob.glob(f'{opts.inpath}*{param_str}*{yr}.nc'))
        if len(files) > 1:
            raise FileNotFoundError(f'There are multiple files for {yr} in the given input '
                                    f'directory. Please check and rerun.')
        file = files[0]
        data_yr = xr.open_dataset(file)
        if opts.dataset == 'SPARTACUS' and opts.parameter == 'P24h_7to7':
            data_yr = data_yr.rename({'RR': opts.parameter})
        data_param = data_yr[opts.parameter]

        # in case of European wide data, set all cells outside of region to nan (ERA5 data are not
        # smoothed --> we don't need data outside the GR and can apply mask here to reduce
        # memory usage)
        # For larger GRs, also store some margins because of special treatment of GRs > 100 areals.
        if 'ERA5' in opts.dataset and gr_size <= 100:
            data_param = data_param.where(masks['lt1500_mask'] == 1)
        elif 'ERA5' in opts.dataset and gr_size > 100:
            valid_cells = masks['lt1500_mask'].where(masks['lt1500_mask'] > 0, drop=True)
            min_lat, max_lat = valid_cells.lat.min().values - 2, valid_cells.lat.max().values + 2
            min_lon, max_lon = valid_cells.lon.min().values - 2, valid_cells.lon.max().values + 2
            data_param = data_param.sel(lat=slice(max_lat, min_lat), lon=slice(min_lon, max_lon))
            # adjust size of DataArray accordingly
            if len(data_param.lon) != len(data_ref.lon) or len(data_param.lat) != len(data_ref.lon):
                data_ref = xr.DataArray(data=np.zeros((len(ref_period), len(dys),
                                                       len(data_param[yn]), len(data_param[xn])),
                                                      dtype='float32') * np.nan,
                                        coords={'year': (['year'], ref_period),
                                                'dys': (['dys'], dys),
                                                yn: ([yn], data_param[yn].values),
                                                xn: ([xn], data_param[xn].values)})

        # only select WAS and wet days for precip
        if opts.precip:
            data_param = data_param.sel(time=slice(f'{yr}-04-01', f'{yr}-10-31'))
            data_param = data_param.where(data_param > 0.99)
            data = data_param.values
        else:
            # check length of year and add 29th of February if necessary
            if len(data_param.time) != 366:
                t29th = np.zeros((1, len(data_param[yn]), len(data_param[xn]))) * np.nan
                data = np.append(data_param.sel(
                    time=slice(f'{yr}-01-01', f'{yr}-02-28')).values, t29th, axis=0)
                data = np.append(data, data_param.sel(
                    time=slice(f'{yr}-03-01', f'{yr}-12-31')).values, axis=0)
            else:
                data = data_param.values

        # combine years to 4D dataset
        data_ref[idx, :, :, :] = data

        idx += 1

    return data_ref


def calc_percentiles(opts, masks, gr_size):
    """
    caluclate percentile of reference period for each grid point
    Args:
        opts: CLI parameter
        masks: GR masks
        gr_size: size of GR in areals

    Returns:
        thresh: threshold (percentile) grid

    """

    params = {'SPARTACUS': {'xname': 'x', 'yname': 'y'},
              'ERA5': {'xname': 'lon', 'yname': 'lat'},
              'ERA5Land': {'xname': 'lon', 'yname': 'lat'}}

    data = load_ref_data(opts=opts, masks=masks, ds_params=params, gr_size=gr_size)

    # calc the chosen percentile for each grid point as threshold
    # (TMax-p99ANN AllDOYs Ref1961-1990 & P24H-p95WAS WetDOYs > 1 mm Ref1961-1990).
    percent = data.quantile(q=opts.threshold / 100, dim=('year', 'dys'))

    # smooth SPARTACUS precip percentiles (for each grid point calculate the average of all grid
    # points within the given radius)
    radius = opts.smoothing

    if radius == 0:
        percent_smooth = percent.copy()
    else:
        percent_smooth_arr = np.full_like(percent.values, np.nan)
        y_size = len(data[params[opts.dataset]['yname']])
        x_size = len(data[params[opts.dataset]['xname']])
        percent_tmp = np.zeros((y_size + 2 * radius, x_size + 2 * radius),
                               dtype='float32') * np.nan
        percent_tmp[radius:radius + y_size, radius:radius + x_size] = percent

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

        percent_smooth = xr.full_like(percent, np.nan)
        percent_smooth[:, :] = percent_smooth_arr

    vname = f'{opts.parameter}-p{opts.threshold}ANN Ref1961-1990'
    if opts.precip:
        vname = f'{opts.parameter}-p{opts.threshold}WAS WetDOYs > 1 mm Ref1961-1990'

    percent_smooth = percent_smooth.drop('quantile')

    # apply GR mask
    if 'ERA' in opts.dataset and opts.region != 'EUR' and gr_size > 100:
        percent_smooth = percent_smooth.where(masks['lt1500_mask_EUR'] == 1)
    else:
        percent_smooth = percent_smooth.where(masks['lt1500_mask'] == 1)
        percent_smooth = percent_smooth * masks['mask']
    percent_smooth = percent_smooth.rename('threshold')
    percent_smooth.attrs = {'units': opts.unit, 'methods_variable_name': vname,
                            'percentile': f'{opts.threshold}p'}

    return percent_smooth


def run():
    warnings.filterwarnings(action='ignore', message='All-NaN slice encountered')
    warnings.filterwarnings(action='ignore', message='Mean of empty slice')

    # opts = get_opts()
    opts = load_opts(script_name=sys.argv[0].split('/')[-1].split('.py')[0])
    opts = extend_tea_opts(opts)

    # load GR masks
    masks = xr.open_dataset(f'{opts.maskpath}{opts.region}_masks_{opts.dataset}.nc')

    # create area grid
    area, gr_size = area_grid(opts=opts, masks=masks)

    # calculate thresholds
    if opts.threshold_type == 'abs':
        thr_grid = xr.full_like(masks['nw_mask'], opts.threshold)
        thr_grid = thr_grid.where(masks['lt1500_mask'] == 1)
        if 'ERA' in opts.dataset and gr_size > 100:
            thr_grid = thr_grid
        else:
            thr_grid = thr_grid * masks['mask']
        thr_grid = thr_grid.rename('threshold')
        thr_grid.attrs = {'units': opts.unit, 'abs_threshold': f'{opts.threshold}{opts.unit}'}
    else:
        thr_grid = calc_percentiles(opts=opts, masks=masks, gr_size=gr_size)

    # combine to single dataset
    ds_out = xr.merge([area, gr_size, thr_grid], join='left')
    del ds_out.attrs['units']
    ds_out = create_history(cli_params=sys.argv, ds=ds_out)

    # add additional attributes
    ds_out.attrs['region'] = opts.region
    ds_out.attrs['dataset'] = opts.dataset
    ds_out.attrs['coordinate_sys'] = masks.attrs['coordinate_sys']

    # save output
    outname = f'{opts.outpath}static_{opts.param_str}_{opts.region}_{opts.dataset}.nc'

    ds_out.to_netcdf(outname)


if __name__ == '__main__':
    run()
