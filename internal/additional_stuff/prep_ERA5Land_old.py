#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: hst
"""

import argparse

from pathlib import Path
from metpy import calc
import numpy as np
import os
import sys
from tqdm import trange
import xarray as xr

try:
    from common.general_functions import create_history_from_cli_params
except ImportError:
    from src.teametrics.common.general_functions import create_history_from_cli_params


def get_opts():
    """
    loads CLI parameter
    Returns:
        myopts: CLI parameter
    """
    parser = argparse.ArgumentParser()

    def dir_path(path):
        if os.path.isdir(path):
            return path
        else:
            raise argparse.ArgumentTypeError(f'{path} is not a valid path.')

    parser.add_argument('--inpath',
                        default='/data/arsclisys/normal/ERA5_land/hourly/',
                        type=dir_path,
                        help='Input directory.')

    parser.add_argument('--outpath',
                        default='/data/users/hst/TEA-clean/ERA5Land/',
                        type=dir_path,
                        help='Output directory.')

    parser.add_argument('--orog-file',
                        default='/data/users/hst/cdrDPS/orographies/ERA5Land_geopotential.nc',
                        dest='orog_file',
                        help='Orography file.')

    myopts = parser.parse_args()

    return myopts


def calc_altitude(ds_in, orog_file):
    """
    calc altitude from geopotential
    Args:
        ds_in: input ds
        orog_file: filename of orography data

    Returns:
        altitude: altitude

    """
    data = xr.open_dataset(ds_in)

    # altitude
    ds_geop = xr.open_dataset(orog_file)

    lat_min = data.latitude.values.min()
    lat_max = data.latitude.values.max()
    lon_min = data.longitude.values.min()
    lon_max = data.longitude.values.max()

    ds_geop_aut = ds_geop.sel(longitude=slice(lon_min, lon_max), latitude=slice(lat_max, lat_min))
    ds_geop_aut = ds_geop_aut.reset_coords(drop=True)

    # height
    altitude = ds_geop_aut.z.resample(time='1D').mean() / 9.80665
    altitude = altitude.rename('altitude')

    altitude = altitude[0, :, :]

    return altitude


def resample_temperature(ds_in):
    """
    resample temperature to daily data and save it in °C
    Args:
        ds_in: input temperature data

    Returns:
        tav: mean daily temperature in °C
        tmin: daily min temperature in °C
        tmax: daily max temperature in °C

    """

    ds_in = ds_in.shift(time=1)
    t_resampled = ds_in.resample(time='1D')
    tav = t_resampled.mean() - 273.15
    tav = tav.rename('T')
    tmin = t_resampled.min() - 273.15
    tmin = tmin.rename('Tn')
    tmax = t_resampled.max() - 273.15
    tmax = tmax.rename('Tx')

    # Set attributes
    tmin.attrs = {'units': '°C', 'long_name': 'daily minimum temperature'}
    tmax.attrs = {'units': '°C', 'long_name': 'daily maximum temperature'}
    tav.attrs = {'units': '°C', 'long_name': 'daily average temperature'}

    return tav, tmin, tmax


def resample_precipitation(ds_in, shift=0):
    """
    resample precipitation to daily data and save it in mm
    Args:
        ds_in: precipitation data
        shift: hours by which data should be shifted

    Returns:
        p24h: 24h precipitation sum in mm
        p1h: daily max hourly precipitation sum in mm

    """

    ds_in = ds_in.shift(time=(1+shift))
    # precip is given in m in ERA5Land data, change to mm
    p24h = ds_in.resample(time='1D').max() * 1000

    # shift data again to get hourly precipitation amount
    ds_shift = ds_in.shift(time=(1+shift))
    diff = ds_shift - ds_in
    # find maximum hourly precipitation amount
    px1h = diff.resample(time='1D').max() * 1000

    p24h_name, px1h_name, shift_str = 'P24h', 'Px1h', ''
    if shift > 0:
        p24h_name = f'P24h_{shift}to{shift}'
        px1h_name = f'Px1h_{shift}to{shift}'
        shift_str = f' ({shift}to{shift})'

    p24h = p24h.rename(p24h_name)
    px1h = px1h.rename(px1h_name)

    p24h.attrs = {'units': 'mm', 'long_name': f'daily precipitation sum{shift_str}'}
    px1h.attrs = {'units': 'mm', 'long_name': f'daily maximum hourly precipitation{shift_str}'}

    return p24h, px1h


def calc_wind(ds_in):
    """
    calculate wind speed in m s**-1

    Args:
        ds_in: input data

    Returns:
        wind: wind speed in m s**-1

    """

    data_u, data_v = ds_in.u10, ds_in.v10

    data_u = data_u.shift(time=1)
    data_v = data_v.shift(time=1)
    ucom = data_u.resample(time='1D').mean()
    vcom = data_v.resample(time='1D').mean()
    wind = np.sqrt(ucom ** 2 + vcom ** 2)

    wind = wind.rename('WindSpeed')
    wind.attrs = {'units': 'm s**-1', 'long_name': f'10m wind speed'}

    return wind


def resample_pressure(ds_in):
    """
    resamples pressure data to daily data
    Args:
        ds_in: hourly pressure data

    Returns:
        pressure: daily pressure data
    """

    ds_in = ds_in.shift(time=1)
    pressure = ds_in.resample(time='1D').mean()
    pressure = pressure.rename('p')
    pressure.attrs = {'units': 'Pa', 'long_name': 'surface pressure'}

    return pressure


def calc_specific_hum(t_dp, pressure):
    """
    calculates the specific humidity and resamples it to daily data
    Args:
        t_dp: hourly dewpoint temperature data
        pressure: hourly pressure data

    Returns:
        q: daily specific humidity data
    """

    t_dp = t_dp.shift(time=1)
    pressure = pressure.shift(time=1)
    q = calc.specific_humidity_from_dewpoint(pressure, t_dp)
    q = q.resample(time='1D').mean()

    q = q.rename('q')
    q.attrs = {'units': '1', 'long_name': 'specific humidity'}

    return q


def run():
    opts = get_opts()

    files = sorted(Path(opts.inpath).glob('*ERA5Land*nc'))

    altitude = calc_altitude(ds_in=files[0], orog_file=opts.orog_file)

    # Save altitude in separate file
    altitude = create_history_from_cli_params(cli_params=sys.argv, ds=altitude, dsname='ERA5Land')
    alt_out = altitude.copy()
    alt_out = alt_out.rename({'latitude': 'lat', 'longitude': 'lon'})
    alt_out.to_netcdf(Path(opts.outpath) / 'ERA5Land_orography.nc')

    for ifile in trange(len(files), desc='Preparing ERA5Land data'):
        file = files[ifile]
        ds_in = xr.open_dataset(file, mask_and_scale=True)
        filename = file.name

        # Temperature
        tav, tmin, tmax = resample_temperature(ds_in=ds_in.t2m)

        # Precipitation
        p24h, p1h = resample_precipitation(ds_in=ds_in.tp)
        p24h_7to7, p1h_7to7 = resample_precipitation(ds_in=ds_in.tp, shift=7)

        # Wind
        wind = calc_wind(ds_in=ds_in)

        # Surface pressure
        pressure = resample_pressure(ds_in=ds_in.sp)

        # Specific humidity
        humidity = calc_specific_hum(t_dp=ds_in.d2m, pressure=ds_in.sp)

        # Create output ds
        ds_out = xr.merge([tav, tmin, tmax, p24h, p1h, p24h_7to7, p1h_7to7, wind, pressure,
                           humidity, altitude])
        ds_out = create_history_from_cli_params(cli_params=sys.argv, ds=ds_out, dsname='ERA5Land')
        ds_out = ds_out.rename({'latitude': 'lat', 'longitude': 'lon'})

        ds_out['lat'] = (np.arange(ds_out.lat[-1] * 10, (ds_out.lat[0] * 10) + 1) / 10)[::-1]
        ds_out['lon'] = (np.arange(ds_out.lon[0] * 10, (ds_out.lon[-1] * 10) + 1) / 10)

        ds_out.to_netcdf(Path(opts.outpath) / filename)


if __name__ == '__main__':
    run()
