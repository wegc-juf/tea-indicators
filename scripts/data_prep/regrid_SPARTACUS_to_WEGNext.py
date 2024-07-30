#!/opt/virtualenv3.7/bin/python3
# -*- coding: utf-8 -*-
"""
@author: hst

"""

import argparse
import glob
import numpy as np
import os
from pathlib import Path
import pyproj
import sys
from tqdm import trange
import xarray as xr

sys.path.append('/home/hst/tea-indicators/scripts/misc/')
from general_functions import create_history


def get_opts():
    """
    loads input parameter
    :return: opts
    """

    def dir_path(path):
        if os.path.isdir(path):
            return path
        else:
            raise argparse.ArgumentTypeError(f'{path} is not a valid path.')

    def file(entry):
        if os.path.isfile(entry):
            return entry
        else:
            raise argparse.ArgumentTypeError(f'{entry} is not a valid file.')

    parser = argparse.ArgumentParser()

    parser.add_argument('--parameter',
                        default='Tx',
                        type=str,
                        choices=['Tx', 'Tn', 'RR', 'TX'],
                        help='Parameter for which the SPARTACUS data should be regridded '
                             '[options: Tx (default), Tn, RR, RRhr].')

    parser.add_argument('--inpath',
                        default='/data/reloclim/backup/ZAMG_SPARTACUS/data/v202108/',
                        type=dir_path,
                        help='Path of folder where data is located.')

    parser.add_argument('--orography',
                        action='store_true',
                        help='Set if orography should be regridded.')

    parser.add_argument('--orofile',
                        default='/data/reloclim/backup/ZAMG_INCA/data/original/'
                                'INCA_orog_corrected_y_dim.nc',
                        type=file,
                        help='Orography file only necessary if "orography" is set to true.')

    parser.add_argument('--outpath',
                        default='/data/users/hst/TEA-clean/SPARTACUS/',
                        help='Path of folder where output data should be saved.')

    myopts = parser.parse_args()

    return myopts


def define_wegn_grid_1000x1000(opts):
    """
    create 1 km resolution grid over SPARTACUS region with identical points of
    WegenerNet grid within the FBR region
    Args:
        opts: CLI parameter

    Returns:
        grid: dummy da with new grid coordinates

    """

    # Load sample SPARTACUS data
    original_grid = xr.open_dataset(
        os.path.join(f'{opts.inpath}', f'{opts.parameter}2000.nc'))

    # Open WEGN sample data
    wegnet = xr.open_mfdataset(os.path.join('/data/users/hst/cdrDPS/wegnet/',
                                            'WN_L2_DD_v7_UTM_TF1_UTC_2020-08.nc'),
                               combine='by_coords')
    # Select 2020-08-15 (just a random date)
    wegnet_first = wegnet.isel(time=14)

    # New coordinates in 1 km x 1 km instead of 200 m x 200 m
    fbr_x = wegnet_first.X.values[2:-2:5]
    fbr_y = wegnet_first.Y.values[2:-2:5]

    # Change SPARTACUS projection to UTM
    x_spa, y_spa = epsg3416_to_utm_grid(original_grid.x, original_grid.y)

    # Round corners to 1000
    xmin, xmax = np.round(np.min(x_spa) / 1000) * 1000, np.round(np.max(x_spa) / 1000) * 1000
    ymin, ymax = np.round(np.min(y_spa) / 1000) * 1000, np.round(np.max(y_spa) / 1000) * 1000

    # FBR corners
    xmin_fbr, xmax_fbr = np.min(fbr_x), np.max(fbr_x)
    ymin_fbr, ymax_fbr = np.min(fbr_y), np.max(fbr_y)

    left = np.arange(xmin_fbr, xmin, -1000)[::-1]
    right = np.arange(xmax_fbr, xmax, 1000)
    bottom = np.arange(ymin_fbr, ymin, -1000)[::-1]
    top = np.arange(ymax_fbr, ymax, 1000)

    x_new = np.concatenate((left[:-1], fbr_x, right[1:]), axis=0)
    y_new = np.concatenate((bottom[:-1], fbr_y, top[1:]), axis=0)

    # Create DataArray with new grid and dummy values
    dummy_data = np.zeros((len(y_new), len(x_new)))

    grid = xr.Dataset(data_vars=dict(data=(["y", "x"], dummy_data), ),
                      coords=dict(x=(["x"], x_new), y=(["y"], y_new), ), )

    return grid


def utm_to_epsg3416_grid(x, y):
    """
     x ... 1d array
     y ... 1d array
     newx, newy ... meshgrid
     """
    transformer = pyproj.Transformer.from_crs('EPSG:32633', 'EPSG:3416')
    ny, nx = len(y), len(x)
    x, y = np.meshgrid(x, y)
    newy, newx = transformer.transform(x.flatten(), y.flatten())
    newx = np.asarray(newx).reshape((ny, nx))
    newy = np.asarray(newy).reshape((ny, nx))
    return newx, newy


def epsg3416_to_utm_grid(x, y):
    """
    x ... 1d array
    y ... 1d array
    newx, newy ... meshgrid
    """
    transformer = pyproj.Transformer.from_crs('EPSG:3416', 'EPSG:32633')
    ny, nx = len(y), len(x)
    y, x = np.meshgrid(y, x)
    newx, newy = transformer.transform(y.flatten(), x.flatten())
    newy = np.sort(newy)
    newx = np.asarray(newx).reshape((nx, ny))
    newx = newx.T
    newy = np.asarray(newy).reshape((ny, nx))

    return newx, newy


def regrid_spartacus(opts, ds_in, method="linear"):
    """
    regrid SPARTACUS data (projection EPSG3416) to the extended FBR region (projection UTM33N)
    Args:
        opts: CLI parameter
        ds_in: ds for regridding
        method: interpolation method (default: linear)

    Returns:
        ds_regridded: regridded ds
    """

    # Define extended WEGN grid
    grid = define_wegn_grid_1000x1000(opts=opts)
    x_new_utm = grid.x.values
    y_new_utm = grid.y.values

    # Get remapping arrays
    # UTM to EPSG because input dataset is SPARTACUS, which is in EPSG3416
    x_new_epsg3416, y_new_epsg3416 = utm_to_epsg3416_grid(x_new_utm, y_new_utm)
    x = xr.DataArray(x_new_epsg3416, dims=["y", "x"], coords={"x": x_new_utm, "y": y_new_utm})
    y = xr.DataArray(y_new_epsg3416, dims=["y", "x"], coords={"x": x_new_utm, "y": y_new_utm})

    # Interpolation
    ds_regridded = ds_in.interp(x=x, y=y, method=method)

    return ds_regridded


def regrid_orog(opts):
    """
    regrid orography to new grid
    :return:
    """

    orog_file = xr.open_dataset(opts.orofile)
    oro_new = regrid_spartacus(opts=opts, ds_in=orog_file.orog, method='linear')
    oro_new = oro_new.assign_attrs(grid_mapping='UTM33N')
    oro_new = create_history(cli_params=sys.argv, ds=oro_new)
    oro_new.attrs['crs'] = 'EPSG:32633'

    path = Path(f'{opts.outpath}')
    path.mkdir(parents=True, exist_ok=True)
    oro_new.to_netcdf(f'{opts.outpath}SPARTACUSreg_orography.nc')


def run():
    opts = get_opts()

    if opts.orography:
        regrid_orog(opts=opts)
    else:
        input_files = sorted(glob.glob(f'{opts.inpath}/*{opts.parameter}*.nc'))
        for ifile in trange(len(input_files), desc='Regridding files'):
            filename = input_files[ifile].split('/')[-1]

            # Open SPARTACUS file
            ds = xr.open_dataset(input_files[ifile], engine='netcdf4')

            # Regrid to extended WEGN grid
            ds_new = regrid_spartacus(opts=opts, ds_in=ds, method='linear')
            ds_new = ds_new.assign_attrs(grid_mapping='UTM33N')

            # Add history to attributes and change crs to EPSG:32633
            ds_new = create_history(cli_params=sys.argv, ds=ds_new)
            ds_new.attrs['crs'] = 'EPSG:32633'

            # Rename variables if necessary
            if opts.parameter == 'TX':
                ds_new = ds_new.rename({'TX': 'Tx'})
                opts.parameter = 'Tx'
            elif opts.parameter == 'TN':
                ds_new = ds_new.rename({'TN': 'Tn'})

            # Save output
            encoding = {opts.parameter: {'dtype': 'int16', 'scale_factor': 0.1,
                                         '_FillValue': -9999}}

            path = Path(f'{opts.outpath}')
            path.mkdir(parents=True, exist_ok=True)
            ds_new.to_netcdf(f'{opts.outpath}{filename}', encoding=encoding,
                             engine='netcdf4')


if __name__ == '__main__':
    run()
