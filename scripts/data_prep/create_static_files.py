#!/opt/virtualenv3.11/bin/python3
# -*- coding: utf-8 -*-
"""
@author: hst
"""

import argparse
import cftime as cft
import os

import numpy as np
import xarray as xr


def get_opts():
    """
    get CLI parameter
    Returns:
        opts: CLI parameter
    """

    def dir_path(path):
        if os.path.isdir(path):
            return path
        else:
            raise argparse.ArgumentTypeError(f'{path} is not a valid path!')

    def file(entry):
        if os.path.isfile(entry):
            return entry
        else:
            raise argparse.ArgumentTypeError(f'{entry} is not a valid testfile!')

    parser = argparse.ArgumentParser()

    parser.add_argument('--parameter',
                        default='T',
                        choices=['T', 'P'],
                        type=str,
                        help='Parameter for which the TEA indices should be calculated'
                             '[default: T].')

    parser.add_argument('--threshold',
                        default=99,
                        type=float,
                        help='Threshold in degrees Celsius, mm, or as percentile [default: 99].')

    parser.add_argument('--threshold_type',
                        type=str,
                        choices=['perc', 'abs'],
                        default='perc',
                        help='Pass "perc" (default) if percentiles should be used as thresholds or '
                             '"abs" for absolute thresholds.')

    parser.add_argument('--precip_var',
                        default='P24h_7to7',
                        type=str,
                        choices=['Px1h', 'P24h', 'Px1h_7to7', 'P24h_7to7'],
                        help='Precipitation variable used [default: P24h_7to7].')

    parser.add_argument('--region',
                        default='AUT',
                        type=str,
                        help='Geo region [options: AUT (default), Austrian state name, '
                             'or ISO2 code of european country].')

    # parser.add_argument('--testfile',
    #                     default='/data/users/hst/cdrDPS/spartacus/Tx2000.nc',
    #                     type=file,
    #                     help='One of the files that will be used for the TEA index calculation '
    #                          'later on.')

    parser.add_argument('--maskpath',
                        dest='maskpath',
                        default='/data/users/hst/TEA-clean/masks/',
                        type=dir_path,
                        help='Path of folder where GR masks are located.')

    parser.add_argument('--outpath',
                        dest='outpath',
                        default='/data/users/hst/TEA-clean/static/',
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


def tfile_param(opts):
    """
    get variable name depending on opts
    :param opts: CLI parameter
    :return:
    """

    params = {
        'SPARTACUS': {'T': 'Tx', 'P': 'RR'},
        'SPARTACUSv2': {'T': 'Tx', 'P': 'RR'},
        'SPARTACUSv3': {'T': 'Tx', 'P': 'RR'},
        'SPARTACUSv2RRhr': {'T': None, 'P': 'RRhr'},
        'ERA5': {'T': 'Tmax', 'P': 'Precip24Hsum'},
        'ERA5Land': {'T': 'Tmax', 'P': 'Precip24Hsum'},
        'WEGN': {'T': 'T_zAvgTerr_DM', 'P': 'Precip_DS'},
        'EOBS': {'T': 'tx', 'P': 'rr'},
        'REGEN': {'T': None, 'P': 'p'},
    }

    return params[opts.dataset][opts.parameter]


def load_testfile(opts):
    """
    load a single file to get the grid dimensions
    :param opts: CLI parameter
    :return: shape of grid
    """

    test = xr.open_dataset(opts.testfile)

    vname = tfile_param(opts=opts)

    if vname is None:
        raise AttributeError(f'There is no {opts.dataset} {opts.parameter} data!')

    if vname not in test.data_vars:
        test = test['Rx1h'][0, :, :]
    else:
        test = test[vname][0, :, :]

    shape_testfile = np.shape(test)

    return shape_testfile


def threshold_grid(opts, grid, threshold, mask):
    """
    creates a grid where each grid cell gets assigned the same threshold value
    :param opts: CLI parameter
    :param grid: size of grid
    :param threshold: value to assign each grid point
    :param mask: region mask
    :return: threshold grid
    """

    print('Creating threshold grid...')

    # create grid the same size as testfile
    thr_grid = np.ones((grid[0], grid[1]), dtype='float16') * threshold
    thr_grid = thr_grid * mask

    # save threshold grid
    path_out = (f'{opts.outpath}/thresholds/threshold_grid_{opts.region}_T_{threshold}C'
                f'_{opts.dataset}.npy')
    np.save(path_out, thr_grid)


def area_grid(opts, grid):
    """
    creates  and saves a grid where each grid cell gets assigned the size of each grid cell in
    km2 per 100 km2
    :param opts: CLI parameter
    :param grid: size of grid
    :return: area_grid
    """

    print('Creating area grid...')

    if 'SPARTACUS' in opts.dataset:
        opts.dataset = opts.dataset + 'reg'
        # creates area grid in km2 per 100 km2
        a_grid = np.ones((grid[0], grid[1]), dtype='float32') / 100
    elif opts.dataset == 'WEGN':
        a_grid = np.ones((grid[0], grid[1]), dtype='float32') * 0.01 / 100
    else:
        data = xr.open_dataset(opts.testfile)
        if opts.dataset == 'ERA5':
            delta_fac = 4  # to get 0.25° resolution
        else:
            delta_fac = 10  # to get 0.1°
        try:
            lat = data.lat.values
        except AttributeError:
            lat = data.latitude.values
        r_mean = 6371
        u_mean = 2 * np.pi * r_mean
        # calculate earth radius at different latitudes
        r_lat = np.cos(np.deg2rad(lat)) * r_mean
        # calculate earth circumference at latitude
        u_lat = 2 * np.pi * r_lat
        # calculate length of 0.25°/0.1° in m for x and y dimension
        x_len = (u_lat / 360) / delta_fac
        a_grid = np.ones((grid[0], grid[1])) * (u_mean / 360) / delta_fac
        # calculate area of each grid cell
        for ix, ixval in enumerate(a_grid[0, :]):
            a_grid[:, ix] = ixval * x_len
        # convert to km2 per 100 km2
        a_grid = a_grid / 100

    a_grid = a_grid.astype('float32')

    path_out = f'{opts.outpath}area_grid_{opts.dataset}.npy'
    np.save(path_out, a_grid)

    return a_grid


def load_mask(opts):
    """
    load GR masks
    Args:
        opts: CLI parameter

    Returns:
        mask_ds: ds with mask, nw_mask, and orography mask
    """

    mask_ds = xr.open_dataset(f'{opts.maskpath}{opts.region}_masks_{opts.dataset}.nc')

    return mask_ds


def tfile_params(opts):
    """
    get parameter for temperature files
    :param opts: CLI parameter
    :return:
    """

    params = {
        'SPARTACUS': {'fpath': '/data/users/hst/cdrDPS/spartacus/Tx', 'vname': 'Tx'},
        'SPARTACUSreg': {'fpath': '/data/users/hst/cdrDPS/spartacus/Tx', 'vname': 'Tx'},
        'SPARTACUSv2reg': {'fpath': '/data/users/hst/cdrDPS/spartacus_v2/Tx', 'vname': 'Tx'},
        'SPARTACUSv3reg': {'fpath': '/data/users/hst/cdrDPS/spartacus_v3/Tx', 'vname': 'Tx'},
        'ERA5': {'fpath': '/data/users/hst/cdrDPS/ERA5/ERA5_', 'vname': 'Tmax'},
        'ERA5Land': {'fpath': '/data/users/hst/cdrDPS/ERA5Land/ERA5Land_', 'vname': 'Tmax'},
        'WEGN': {'fpath': '/var/wegnet/wegc203058/netcdf/WEA/WN_L2_DD_v8_UTM_TF1_UTC_',
                 'vname': 'T_zAvgTerr_DM'}
    }

    return params[opts.dataset]['fpath'], params[opts.dataset]['vname']


def load_temp_percentile_ref_data(opts, grid):
    """
    loads and reshapes reference data
    :param opts: CLI parameter
    :param grid: size of grid
    :return: reshaped reference data (4d array)
    """

    print('Loading temperature reference data...')

    if opts.dataset != 'WEGN':
        start_ref = "1961"
        end_ref = "1990"
    else:
        start_ref = "2007"
        end_ref = "2021"
    ref_period = np.arange(int(start_ref), int(end_ref) + 1)
    tref = np.zeros((30, 366, grid[0], grid[1]), dtype='float32') * np.nan

    idx = 0

    file_path, var = tfile_params(opts=opts)

    for yr in ref_period:
        if opts.dataset != 'WEGN':
            file = f'{file_path}{yr}.nc'
            tyr = xr.open_dataset(file)
            tx = tyr[var].values
        else:
            files = f'{file_path}{yr}-*.nc'
            tyr = xr.open_mfdataset(files)
            tx = tyr[var].values - 273.15

        # check length of year and add 29th of February if necessary
        if len(tx) != 366:
            t29th = np.zeros((1, grid[0], grid[1])) * np.nan
            t = np.append(tx[0:59, :, :], t29th, axis=0)
            t = np.append(t, tx[59:, :, :], axis=0)
        else:
            t = tx

        # combine years to 4d dataset
        tref[idx, :, :, :] = t

        idx += 1

    return tref


def load_temp_percentile_eobs_ref_data(grid):
    """
    loads and reshapes reference data
    :param grid: size of grid
    :return: reshaped reference data (4d array)
    """

    print('Loading temperature reference data...')

    start_ref = '1961'
    end_ref = '1990'

    ref_period = np.arange(int(start_ref), int(end_ref) + 1)
    tref = np.zeros((30, 366, grid[0], grid[1]), dtype='float32') * np.nan

    data = xr.open_dataset('/home/hst/data/EOBS/v27.0/EOBS_tx_1950to2022.nc')
    data = data.tx

    for iyr, yr in enumerate(ref_period):
        tx = data.sel(time=slice(f'{yr}-01-01', f'{yr}-12-31'))

        # check length of year and add 29th of February if necessary
        if len(tx) != 366:
            t29th = np.zeros((1, grid[0], grid[1])) * np.nan
            t = np.append(tx[0:59, :, :], t29th, axis=0)
            t = np.append(t, tx[59:, :, :], axis=0)
        else:
            t = tx

        # combine years to 4d dataset
        tref[iyr, :, :, :] = t

    return tref


def pfile_params(opts):
    """
    get params for precip files depending on opts
    :param opts: CLI parameter
    :return:
    """

    params = {
        'SPARTACUS': {'filep': '/data/users/hst/cdrDPS/spartacus/RR', 'vname': 'RR'},
        'SPARTACUS_Px1H': {'filep': '/home/hst/data/spartacus/Rx1h/Rx1h_', 'vname': 'Rx1h'},
        'SPARTACUSreg': {'filep': '/data/users/hst/cdrDPS/spartacus/RR', 'vname': 'RR'},
        'SPARTACUSreg_Px1H': {'filep': '/home/hst/data/spartacus/Rx1h/Rx1h_', 'vname': 'Rx1h'},
        'SPARTACUSv2reg': {'filep': '/data/users/hst/cdrDPS/spartacus_v2/RR', 'vname': 'RR'},
        'SPARTACUSv3reg': {'filep': '/data/users/hst/cdrDPS/spartacus_v3/RR', 'vname': 'RR'},
        'SPARTACUSv2RRhrreg': {'filep': '/data/users/hst/cdrDPS/spartacus_v2_RRhr/RRhr',
                               'vname': 'RRhr'},
        'SPARTACUSv2RRhrreg_Px1H': {'filep': '/home/hst/data/spartacus/Rx1h/SPCUSv2RRhr/Rx1h_',
                                    'vname': 'Rx1h'},
        'WEGN': {'filep': '/var/wegnet/wegc203058/netcdf/WEA/WN_L2_DD_v8_UTM_TF1_UTC_',
                 'vname': 'Precip_DS'},
        'ERA5Land': {'filep': '/data/users/hst/cdrDPS/ERA5Land/ERA5Land_',
                     'vname': opts.precip_var},
        'ERA5': {'filep': '/data/users/hst/cdrDPS/ERA5/ERA5_', 'vname': opts.precip_var}
    }

    if '24H' in opts.precip_var:
        fp, vn = params[opts.dataset]['filep'], params[opts.dataset]['vname']
    else:
        fp, vn = params[f'{opts.dataset}_Px1H']['filep'], params[f'{opts.dataset}_Px1H']['vname']

    return fp, vn


def load_precip_ref_data(opts, _grid):
    """
    loads and reshapes reference data
    :param opts: CLI parameter
    :param _grid: size of grid
    :return: reshaped reference data (4d array)
    """

    print("Loading precipitation data...")

    ref_per_new = False

    if opts.dataset != 'WEGN':
        start_ref = '1961'
        end_ref = '1990'
        yrs = 30
        if ref_per_new:
            end_ref = '1980'
            yrs = 20
        ref_per = [start_ref, end_ref]
        ref_period = np.arange(int(start_ref), int(end_ref) + 1)
        rr_ref = np.zeros((yrs, 214, _grid[0], _grid[1]), dtype='float32') * np.nan
    else:
        start_ref = '2007'
        end_ref = '2020'
        ref_per = [start_ref, end_ref]
        ref_period = np.arange(int(start_ref), int(end_ref) + 1)
        rr_ref = np.zeros((14, 214, _grid[0], _grid[1]), dtype='float32') * np.nan
    idx = 0

    file_path, var = pfile_params(opts=opts)

    for yr in ref_period:
        if opts.dataset != 'WEGN':
            file = f'{file_path}{yr}.nc'
            rr_yr = xr.open_dataset(file)
        else:
            files = f'{file_path}{yr}-*.nc'
            rr_yr = xr.open_mfdataset(files)

        if opts.dataset in ['ERA5', 'ERA5Land']:
            rr_yr['time'] = cft.num2pydate(rr_yr['time'], units='days since 1961-01-01',
                                           calendar='gregorian')

        # select only WAS values for percentile calculation
        rr_yr = rr_yr.sel(time=slice(f'{yr}-04-01', f'{yr}-10-31'))

        rrx = rr_yr[var].values

        # remove days with RR < 0.99 mm
        if opts.precip_var in ['Precip1Hmax', 'Precip1Hmax_7to7']:
            idx_dry = np.where(rrx < 0.29)
        else:
            idx_dry = np.where(rrx < 0.99)

        rrx[idx_dry[0], idx_dry[1], idx_dry[2]] = np.nan

        # combine years to 4d dataset
        rr_ref[idx, :, :, :] = rrx

        idx += 1

    return rr_ref, ref_per


def load_precip_eobs_ref_data(grid):
    """
    loads and reshapes reference data
    :param grid: size of grid
    :return: reshaped reference data (4d array)
    """

    print("Loading precipitation data...")

    start_ref = '1961'
    end_ref = '1990'
    ref_period = np.arange(int(start_ref), int(end_ref) + 1)
    rr_ref = np.zeros((30, 214, grid[0], grid[1]), dtype='float32') * np.nan

    data = xr.open_dataset(f'/home/hst/data/EOBS/v27.0/EOBS_rr_1950to2022.nc')
    data = data.where(data.rr > 0.99)

    for iyr, yr in enumerate(ref_period):
        # select only WAS values for percentile calculation
        rr_yr = data.sel(time=slice(f'{yr}-04-01', f'{yr}-10-31'))

        # combine years to 4d dataset
        rr_ref[iyr, :, :, :] = rr_yr.rr.values

    return rr_ref


def calc_percentiles(opts, data, mask_nw, omask, ref_per):
    """
    calculates percentiles for each grid point as threshold for precipitation
    :param opts: CLI parameter
    :param data: precipitation data (output of load_precip_ref_data)
    :param mask_nw: non-weighted mask of geo region
    :param omask: orography mask
    :param ref_per: start and end str of ref period
    :return: spatially smoothed percentiles as thresholds
    """

    print("Calculating percentiles...")

    # calc the chosen percentile for each grid point as threshold
    percent = np.nanpercentile(data, float(opts.threshold), axis=(0, 1))

    percent = percent * omask

    # smooth percentiles (for each grid point calculate the average of all grid points within
    # the given radius
    if opts.dataset in ['SPARTACUS', 'SPARTACUSreg'] and opts.parameter == 'P':
        radius = 7
    else:
        radius = 0

    if radius == 0:
        percent_smooth = percent
    else:
        percent_smooth = np.full_like(percent, np.nan)
        y_size = percent.shape[0]
        x_size = percent.shape[1]
        percent_tmp = np.zeros((y_size + 2 * radius, x_size + 2 * radius), dtype='float32') * np.nan
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
                percent_smooth[iy_new, ix_new] = np.nanmean(percent_tmp[circ_mask])
                ix_new += 1
            iy_new += 1

    percent_smooth = percent_smooth * mask_nw

    if opts.dataset in ['ERA5', 'ERA5Land'] and opts.parameter == 'P':
        parameter = opts.precip_var
    elif opts.dataset in ['SPARTACUS', 'SPARTACUSreg',
                          'SPARTACUSv2reg', 'SPARTACUSv2RRhrreg', 'EOBS'] and opts.parameter == 'P':
        parameter = 'Precip24Hsum_7to7'
    else:
        parameter = opts.parameter

    ref_str = ''
    if ref_per[0] != '1961' or ref_per[1] != '1990':
        ref_str = f'_Ref{ref_per[0]}-{ref_per[1]}'

    path_out = (f'{opts.outpath}thresholds/threshold_grid_{opts.region}_{parameter}'
                f'_{opts.threshold}percentile_{opts.dataset}{ref_str}.npy')

    np.save(path_out, percent_smooth)

    return percent_smooth


def orog_mask(opts, _reg_mask):
    """
    loads orography mask
    :param opts: CLI parameter
    :param _reg_mask: mask of geo region
    """

    _reg_mask[np.where(_reg_mask == 0)] = np.nan

    if 'SPARTACUS' in opts.dataset:
        ver = ''
        if opts.dataset in ['SPARTACUSv2reg', 'SPARTACUSv2RRhrreg']:
            ver = 'v2'
        elif opts.dataset == 'SPARTACUSv3reg':
            ver = 'v3'
        mask_bel1500 = xr.open_dataset(
            f'/data/users/hst/cdrDPS/orographies/SPARTACUS{ver}_orography_mask_reg.nc')
        mask_bel1500 = mask_bel1500.orog.values * _reg_mask

    elif opts.dataset in ['EOBS', 'REGEN']:
        mask_bel1500 = xr.open_dataset(
            '/home/hst/data/EOBS/v27.0/EOBS_altitude.nc')
        mask_bel1500 = mask_bel1500.where(mask_bel1500.elevation < 1500, 0)
        mask_bel1500 = mask_bel1500.where(mask_bel1500.elevation == 0, 1)
        mask_bel1500 = mask_bel1500.elevation.values * _reg_mask

    elif opts.region == 'COM' or opts.dataset == 'WEGN':
        mask_bel1500 = np.ones(np.shape(_reg_mask))

    else:
        data = xr.open_dataset(opts.testfile)
        z = data.altitude.values[0, :, :]
        mask_bel1500 = np.ones(np.shape(z), dtype='float32')
        mask_bel1500[np.where(z > 1500)] = np.nan
        mask_bel1500 = mask_bel1500 * _reg_mask

    path_out = f'{opts.outpath}masks/orography_mask_{opts.region}_{opts.dataset}.npy'
    np.save(path_out, mask_bel1500)

    return mask_bel1500


def check_wegn_input(opts):
    """
    checks if all input parameter are set correctly
    :param opts: inpput parameter
    :return:
    """

    if opts.region != 'FBR':
        print('There is no WEGN data for {}! Region is set to FBR.'.format(opts.region))
        opts.region = 'FBR'

    return opts


def run():
    opts = get_opts()

    masks = load_mask(opts=opts)

    grid_size = load_testfile(opts=opts)

    area = area_grid(opts=opts, grid=grid_size)

    reg_mask, reg_mask_nw = load_mask(opts=opts, grid=grid_size, area=area)

    omask = orog_mask(opts=opts, _reg_mask=reg_mask_nw)

    if opts.parameter == 'T' and opts.threshold_type == 'abs':
        threshold_grid(opts=opts, threshold=float(opts.threshold), grid=grid_size, mask=reg_mask_nw)

    elif opts.parameter == 'T' and opts.threshold_type == 'perc':
        if opts.dataset != 'EOBS':
            temp = load_temp_percentile_ref_data(opts=opts, grid=grid_size)
        else:
            temp = load_temp_percentile_eobs_ref_data(grid=grid_size)
        calc_percentiles(opts=opts, data=temp, mask_nw=reg_mask_nw, omask=omask)
    else:
        if opts.dataset != 'EOBS':
            precip, ref_per_yrs = load_precip_ref_data(opts=opts, _grid=grid_size)
        else:
            precip, ref_per_yrs = load_precip_eobs_ref_data(grid=grid_size)
        calc_percentiles(opts=opts, data=precip, mask_nw=reg_mask_nw, omask=omask,
                         ref_per=ref_per_yrs)


if __name__ == '__main__':
    run()
