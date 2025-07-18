#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: hst
"""

import argparse
import gc
import glob
from metpy import calc
import numpy as np
import os
import sys
from tqdm import trange
import warnings
import xarray as xr

from common.general_functions import create_history_from_cli_params


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
                        default='/data/users/hst/TEA-clean/ERA5/raw/',
                        type=dir_path,
                        help='Input directory.')

    parser.add_argument('--outpath',
                        default='/data/users/hst/TEA-clean/ERA5/',
                        type=dir_path,
                        help='Output directory.')

    parser.add_argument('--start',
                        default=2025,
                        type=int,
                        help='First year that should be prepped.')

    parser.add_argument('--end',
                        default=2025,
                        type=int,
                        help='Last year that should be prepped.')

    parser.add_argument('--africa',
                        action='store_true',
                        help='Set if African data should be prepped.')

    myopts = parser.parse_args()

    return myopts


def calc_altitude_dt(opts, africa=False):
    """
    calculates altitude in m and time difference to UTC
    Args:
        opts: CLI parameter
        africa: set if african data should be prepped

    Returns:
        altitude: altitude grid
        dt_utc: difference to UTC on grid
        tz: time zones array
    """

    geop_files = sorted(glob.glob(f'{opts.inpath}*geopotential*'))

    ds_alt = xr.open_dataset(geop_files[0])
    altitude = ds_alt.z[0, :, :] / 9.80665
    altitude = altitude.rename('altitude')
    altitude = altitude.rename({'valid_time': 'time'})
    altitude.attrs = {'units': 'm', 'long_name': 'altitude'}
    altitude = altitude.drop(['expver', 'number'])

    nlat = 169
    if africa:
        nlat = 305

    lon_grid = xr.DataArray(data=np.tile(ds_alt.longitude.values, (nlat, 1)),
                            dims={'latitude': (['latitude'], ds_alt.latitude.values),
                                  'longitude': (['longitude'], ds_alt.longitude.values)},
                            coords={'latitude': (['latitude'], ds_alt.latitude.values),
                                    'longitude': (['longitude'], ds_alt.longitude.values)})
    dt_utc = lon_grid / 15
    dt_utc = dt_utc.round().astype('int')

    tz = sorted(set(list(dt_utc.stack(z=('longitude', 'latitude')).values)))

    return altitude, dt_utc, tz


def resample_temperature(data, delta, tz):
    """
    resample temperature to daily data and save it in °C
    Args:
        data: 2m temperature data
        delta: time offset to UTC
        tz: list of time zones

    Returns:
        tav: mean daily temperature in °C
        tmin: daily min temperature in °C
        tmax: daily max temperature in °C

    """

    # T is given in K in ERA5 data, change to °C
    data = data - 273.15

    tav = xr.full_like(data, 0)
    tmin = xr.full_like(data, 0)
    tmax = xr.full_like(data, 0)

    for itz, tz in enumerate(tz):
        tz_data = data.where(delta == tz, 0)
        tz_data = tz_data.shift(valid_time=tz)
        t_resampled = tz_data.resample(valid_time='1D')

        tav_tz = t_resampled.mean()
        tmin_tz = t_resampled.min()
        tmax_tz = t_resampled.max()

        tav = tav + tav_tz
        tmin = tmin + tmin_tz
        tmax = tmax + tmax_tz

    tav = tav.rename('T')
    tmin = tmin.rename('Tn')
    tmax = tmax.rename('Tx')

    # Set attributes
    tmin.attrs = {'units': '°C', 'long_name': 'daily minimum temperature'}
    tmax.attrs = {'units': '°C', 'long_name': 'daily maximum temperature'}
    tav.attrs = {'units': '°C', 'long_name': 'daily average temperature'}

    return tav, tmin, tmax


def resample_precipitation(data, delta, tz, shift=0):
    """
    resample precipitation to daily data and save it in mm
    Args:
        data: precipitation data
        delta: time offset to UTC
        tz: list of time zones
        shift: hours by which data should be shifted

    Returns:
        p24h: 24h precipitation sum in mm
        p1h: daily max hourly precipitation sum in mm

    """

    # Precipitation is given in m in ERA5 data, change to mm
    data = data * 1000

    p24h, px1h = xr.full_like(data, 0), xr.full_like(data, 0)
    for itz, tz in enumerate(tz):
        tz_data = data.where(delta == tz, 0)
        tz_data = tz_data.shift(valid_time=(tz + shift))

        tz_p24h = tz_data.resample(valid_time='1D').sum()
        tz_p1h = tz_data.resample(valid_time='1D').max()

        p24h = p24h + tz_p24h
        px1h = px1h + tz_p1h

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


def calc_wind(data, delta, tz):
    """
    calculate wind speed in m s**-1

    Args:
        data: input data
        delta: time offset to UTC
        tz: list of time zones

    Returns:
        wind: wind speed in m s**-1

    """

    data_u, data_v = data.u10, data.v10

    wind = xr.full_like(data_u, 0)
    for itz, tz in enumerate(tz):
        tz_data_u = data_u.where(delta == tz, 0)
        tz_data_u = tz_data_u.shift(valid_time=tz)
        tz_data_v = data_v.where(delta == tz, 0)
        tz_data_v = tz_data_v.shift(valid_time=tz)

        ucom = tz_data_u.resample(valid_time='1D').mean()
        vcom = tz_data_v.resample(valid_time='1D').mean()
        tz_wind = np.sqrt(ucom ** 2 + vcom ** 2)

        wind = wind + tz_wind

    wind = wind.rename('WindSpeed')
    wind.attrs = {'units': 'm s**-1', 'long_name': f'10m wind speed'}

    return wind


def resample_pressure(data, delta, tz):
    """
    resamples pressure data to daily data
    Args:
        data: hourly pressure data
        delta: time offset to UTC
        tz: list of time zones

    Returns:
        pressure: daily pressure data
    """

    pressure = xr.full_like(data, 0)
    for itz, tz in enumerate(tz):
        tz_data = data.where(delta == tz, 0)
        tz_data = tz_data.shift(valid_time=tz)

        tz_pressure = tz_data.resample(valid_time='1D').mean()
        pressure = pressure + tz_pressure

    pressure = pressure.rename('p')
    pressure.attrs = {'units': 'Pa', 'long_name': 'surface pressure'}

    return pressure


def calc_specific_hum(t_dp, pressure, delta, tz):
    """
    calculates the specific humidity and resamples it to daily data
    Args:
        t_dp: hourly dewpoint temperature data
        pressure: hourly pressure data
        delta: time offset to UTC
        tz: list of time zones

    Returns:
        q: daily specific humidity data
    """

    q = xr.full_like(pressure, 0)
    for itz, tz in enumerate(tz):
        tz_data_tdp = t_dp.where(delta == tz, 0)
        tz_data_tdp = tz_data_tdp.shift(valid_time=tz)
        tz_data_p = pressure.where(delta == tz, 0)
        tz_data_p = tz_data_p.shift(valid_time=tz)

        q_tz = calc.specific_humidity_from_dewpoint(tz_data_p, tz_data_tdp)
        q_tz = q_tz.resample(valid_time='1D').mean()
        q_tz = q_tz.where(q_tz.notnull(), 0)

        q = q + q_tz

    q.attrs = {'units': '1', 'long_name': 'specific humidity'}

    # create new da that is not a quantity (leads to errors in merge later)
    q = xr.DataArray(data=q.values, coords=q.coords, dims=q.dims, attrs=q.attrs, name='q')

    return q


def run():
    warnings.filterwarnings(action='ignore', message='invalid value encountered in divide')
    # warnings.filterwarnings(action='ignore', message='invalid value encountered in true_divide')
    warnings.filterwarnings(action='ignore', message='overflow encountered in exp')

    opts = get_opts()

    if opts.end > opts.start:
        years = np.arange(opts.start, opts.end + 1)
    elif opts.start == opts.end:
        years = [opts.start]
    else:
        raise UserWarning('End year is smaller than start year.')

    altitude, delta_utc, time_zones = calc_altitude_dt(opts=opts, africa=opts.africa)

    # Save altitude in separate file
    create_history_from_cli_params(cli_params=sys.argv, ds=altitude)
    # alt_out = altitude.copy()
    # alt_out = alt_out.rename({'latitude': 'lat', 'longitude': 'lon'})
    # alt_out.to_netcdf(f'{opts.outpath}ERA5_orography.nc')

    tname, pname = '2m_temperature', 'total_precipitation'
    if opts.africa:
        tname, pname = 'instant', 'accum'

    for iyr in trange(len(years), desc='Preparing ERA5 data'):
        basename = f'{opts.inpath}ERA5_{years[iyr]}'

        # Temperature
        da_t2m = xr.open_dataarray(f'{basename}_{tname}.nc',
                                   mask_and_scale=True).chunk()
        da_t2m = da_t2m.drop(['expver'])
        tav, tmin, tmax = resample_temperature(data=da_t2m, delta=delta_utc, tz=time_zones)
        da_t2m.close()
        del da_t2m

        # Precipitation
        da_tp = xr.open_dataarray(f'{basename}_{pname}.nc',
                                  mask_and_scale=True).chunk()
        da_tp = da_tp.drop(['expver'])
        p24h, p1h = resample_precipitation(data=da_tp, delta=delta_utc, tz=time_zones)
        p24h_7to7, p1h_7to7 = resample_precipitation(data=da_tp, delta=delta_utc, tz=time_zones,
                                                     shift=7)
        da_tp.close()
        del da_tp

        dataarrays = [tav, tmin, tmax, p24h, p1h, p24h_7to7, p1h_7to7, altitude]

        if not opts.africa:
            # Wind
            windfiles = sorted(glob.glob(f'{basename}_10m_*_component_of_wind.nc'))
            ds_wind = xr.open_mfdataset(windfiles, mask_and_scale=True).chunk()
            ds_wind = ds_wind.drop(['expver'])
            wind = calc_wind(data=ds_wind, delta=delta_utc, tz=time_zones)
            ds_wind.close()
            del ds_wind

            # Surface pressure
            da_p = xr.open_dataarray(f'{basename}_surface_pressure.nc',
                                     mask_and_scale=True).chunk()
            da_p = da_p.drop(['expver'])
            pressure = resample_pressure(data=da_p, delta=delta_utc, tz=time_zones)

            # Specific humidity
            da_dp = xr.open_dataarray(f'{basename}_2m_dewpoint_temperature.nc',
                                      mask_and_scale=True).chunk()
            da_dp = da_dp.drop(['expver'])
            humidity = calc_specific_hum(t_dp=da_dp, pressure=da_p, delta=delta_utc,
                                         tz=time_zones)
            da_p.close()
            del da_p
            da_dp.close()
            del da_dp

            dataarrays = [tav, tmin, tmax, p24h, p1h, p24h_7to7, p1h_7to7,
                        wind, pressure, humidity, altitude]

        # Create output ds
        ds_out = xr.merge(dataarrays)
        create_history_from_cli_params(cli_params=sys.argv, ds=ds_out)
        ds_out = ds_out.drop(['time', 'number'])
        ds_out = ds_out.rename({'valid_time':'time', 'latitude': 'lat', 'longitude': 'lon'})

        ds_out.to_netcdf(f'{opts.outpath}ERA5_{years[iyr]}.nc')

        # collect garbage
        gc.collect()


def manually_merge_2024_data():
    """
    running all vars at once crashed all the time, therefore variables need to be merged manually
    Returns:

    """
    ds1 = xr.open_dataset('/data/users/hst/TEA-clean/ERA5/ERA5_2024_mainVARS.nc')
    ds2 = xr.open_dataset('/data/users/hst/TEA-clean/ERA5/ERA5_2024_auxVARS.nc')
    ds3 = xr.open_dataset('/data/users/hst/TEA-clean/ERA5/ERA5_2024_auxVARS-sh.nc')

    ds2 = ds2.drop(['altitude'])
    ds3 = ds3.drop(['altitude'])

    ds_out = xr.merge([ds1, ds2, ds3])
    ds_out.attrs['history'] = ds1.attrs['history']
    ds_out.to_netcdf(f'/data/users/hst/TEA-clean/ERA5/ERA5_2024.nc')


if __name__ == '__main__':
    run()
    # manually_merge_2024_data()
