#!/opt/virtualenv3.7/bin/python3
# -*- coding: utf-8 -*-
"""
@author: hst
@reviewer: juf (2024-10-15)
"""

import argparse
import os

import glob
import numpy as np
import pandas as pd
import xarray as xr


def getopts():
    """
    get arguments
    :return: command line parameters
    """

    def dir_path(path):
        if os.path.isdir(path):
            return path
        else:
            raise argparse.ArgumentTypeError(f"{path} is not a valid path")

    parser = argparse.ArgumentParser()

    parser.add_argument('--region',
                        default='AUT',
                        choices=['AUT', 'SEA', 'FBR', 'Niederösterreich'],
                        type=str,
                        help='Geo region options: [AUT (default), FBR, SEA, and Niederösterreich].')

    parser.add_argument('--threshold',
                        default='30',
                        help='Threshold in degrees Celsius [default: 30].')

    parser.add_argument('--dataset',
                        type=str,
                        default='ERA5',
                        choices=['ERA5', 'ERA5Land'],
                        help='Dataset: ERA5 (default) or ERA5Land.')

    parser.add_argument('--folder-input',
                        dest='folder_input',
                        default='/data/users/hst/TEA-clean/TEA/paper_data/daily_basis_variables/',
                        type=dir_path,
                        help='Path of folder where data is located.')

    parser.add_argument('--folder-output',
                        dest='outpath',
                        default='/data/users/hst/TEA-clean/paper_data/energy_content/',
                        help='Path of folder where output data should be saved.')

    myopts = parser.parse_args()

    return myopts


def prep_era5_data():
    """
    prepare ERA5 data: switch order of latitude, convert units, apply mask
    :return:
    """
    years = np.arange(1961, 1963)

    for yr in years:
        print(yr)
        if yr < 2023:
            try:
                raw = xr.open_dataset(f'/data/arsclisys/normal/ERA5/hourly/'
                                      f'single_levels/ERA5_{yr}.nc',
                                      mask_and_scale=True)
            except FileNotFoundError:
                raw = xr.open_dataset(f'/data/arsclisys/normal/ERA5/hourly/'
                                      f'single_levels/preliminary/ERA5pre_{yr}.nc',
                                      mask_and_scale=True)
            rname_dict = {'latitude': 'lat', 'longitude': 'lon'}
        else:
            raw = xr.open_mfdataset(f'/data/users/hst/TEA-clean/ERA5/raw/'
                                    f'ERA5_{yr}_2m_temperature.nc')
            rname_dict = {'valid_time': 'time', 'latitude': 'lat', 'longitude': 'lon'}

        mask = xr.open_dataset('/data/arsclisys/normal/clim-hydro/TEA-Indicators/masks/'
                               'AUT_masks_ERA5.nc')

        era5 = raw.t2m - 273.15
        era5 = era5.rename('T')
        era5 = era5.rename(rname_dict)

        era5 = era5.sel(lat=slice(49.5, 46), lon=slice(9, 17.5))
        era5 = era5.where(mask['lt1500_mask'] == 1)

        # shift to adjust for AUT time zone
        era5 = era5.shift(time=1)

        era5.to_netcdf(f'/data/arsclisys/normal/clim-hydro/TEA-Indicators/ERA5/hourly/'
                       f'T_ERA5_hourly_{yr}.nc')


def prep_era5land_data():
    """
    prepare ERA5 data: switch order of latitude, convert units, apply mask
    :return:
    """

    years = np.arange(1961, 2025)

    for yr in years:
        print(yr)
        if yr < 2023:
            raw = xr.open_dataset(f'/data/arsclisys/normal/ERA5_land/hourly/'
                                  f'ERA5Land_{yr}.nc',
                                  mask_and_scale=True)
            rname_dict = {'latitude': 'lat', 'longitude': 'lon'}
        else:
            raw = xr.open_mfdataset(f'/data/users/hst/TEA-clean/ERA5Land/raw/'
                                    f'ERA5Land_{yr}_2m_temperature.nc')
            rname_dict = {'valid_time': 'time', 'latitude': 'lat', 'longitude': 'lon'}

        mask = xr.open_dataset('/data/arsclisys/normal/clim-hydro/TEA-Indicators/masks/'
                               'AUT_masks_ERA5Land.nc')

        era5l = raw.t2m - 273.15
        era5l = era5l.rename('T')
        era5l = era5l.rename(rname_dict)

        era5l = era5l.sel(lat=slice(49.5, 46), lon=slice(9, 17.5))
        if yr < 2023:
            era5l = era5l.where(mask['lt1500_mask'] == 1)
        else:
            era5l = era5l * mask['lt1500_mask'].values

        # shift to adjust for AUT time zone
        era5l = era5l.shift(time=1)

        # ensure that coords are correct
        era5l['lat'] = (np.arange(era5l.lat[-1] * 10, (era5l.lat[0] * 10) + 1) / 10)[::-1]
        era5l['lat'].attrs = {}
        era5l['lon'] = (np.arange(era5l.lon[0] * 10, (era5l.lon[-1] * 10) + 1) / 10)
        era5l['lon'].attrs = {}

        era5l.to_netcdf(f'/data/arsclisys/normal/clim-hydro/TEA-Indicators/ERA5Land/hourly/'
                        f'T_ERA5Land_hourly_{yr}.nc')


def load_tea_data(opts):
    """
    load TEA data
    :param opts: CLI parameter
    :return: TEA data
    """

    files = sorted(glob.glob(
        f'{opts.folder_input}DBV_Tx{opts.threshold}.0p_annual_{opts.region}_{opts.dataset}_*.nc'))

    data = xr.open_mfdataset(files, data_vars='minimal')
    for ivar in data.data_vars:
        if ivar not in ['DTEA_GR', 'DTEC_GR', 'DTEM_GR', 'DTEC', 'DTEM']:
            data = data.drop_vars(ivar)

    return data


def load_era5_data(opts, yr, reg):
    """
    load ERA5 data
    :param opts: CLI parameter
    :param yr: year
    :param reg: region
    :return: data
    """

    data = xr.open_dataarray(f'/data/arsclisys/normal/clim-hydro/TEA-Indicators/{opts.dataset}/'
                             f'hourly/T_{opts.dataset}_hourly_{yr}.nc')
    mask = xr.open_dataset(f'/data/arsclisys/normal/clim-hydro/TEA-Indicators/masks/'
                           f'{reg}_masks_{opts.dataset}.nc')
    data = data.where(mask['lt1500_mask'] == 1)

    return data


def calc_hourly_tem(data, thresh, orog):
    """
    calc hourly TEM
    :param data: hourly data
    :param thresh: threshold grid
    :param orog: orography grid
    :return: tem (hourly TEM)
    """

    data = data.where(orog == 1)

    # calc hourly TEM (Equation 10_3)
    ex_da = data.where(data > thresh)
    tem = ex_da - thresh

    return tem


def exceedance_hours(tem, area):
    """
    calc daily number-of-exceedance hours (Eq. 10_1 & Eq. 10_3)
    and daily number-of-exceedance hours of GR (Eq. 10_7)
    :param tem: TEM da
    :param area: area grid
    :return:
    """
    # check ok 2024-10-11

    # initialize Nhours array (Eq. 10_1)
    n_hours = xr.full_like(tem, 0)

    # set n_hours of all cells with TEM > 0 to 1 and count daily exceedance hours (Eq. 10_3)
    n_hours = n_hours.where(tem.isnull(), 1)
    n_hours = n_hours.sum(dim='time')
    n_hours = n_hours.rename('Nhours')

    # calc NhoursGR (Eq. 10_7)
    a_nl = area.where(n_hours > 0)
    a_gr = a_nl.sum(dim=('lat', 'lon'))

    n_hours_gr = np.sum((a_nl / a_gr) * n_hours)
    n_hours_gr = n_hours_gr.rename('Nhours_GR')

    return n_hours_gr


def exceedance_times(tem, area):
    """
    calc times of exceedances
    :param tem: hourly TEM
    :param area: area grid
    :return:
    """
    # check ok 2024-10-15

    hrs_arr = xr.full_like(tem, 1) * xr.DataArray(data=np.arange(0, 24),
                                                  dims={'time': (['time'], tem.time.values)},
                                                  coords={'time': (['time'], tem.time.values)})
    hrs_arr = hrs_arr.where(tem.notnull())

    # find t_hfirst and t_hlast (Eq. 10_4 & 10_6)
    thfirst = hrs_arr.min(dim='time')
    thfirst = thfirst.rename('t_hfirst')
    thlast = hrs_arr.max(dim='time')
    thlast = thlast.rename('t_hlast')

    # find t_hmax (Eq. 10_5)
    tem_nonan = tem.where(tem.notnull(), 0)
    max_tem_time_idx = tem_nonan.argmax(dim='time')
    thmax = hrs_arr.isel(time=max_tem_time_idx)
    thmax = thmax.rename('t_hmax')

    # calc GR variables (Eq. 10_7 & 10_8)
    a_nl = area.where(thfirst > 0)
    a_gr = a_nl.sum(dim=('lat', 'lon'))

    thfirst_gr = np.sum((a_nl / a_gr) * thfirst)
    thfirst_gr = thfirst_gr.rename('t_hfirst_GR')
    thlast_gr = np.sum((a_nl / a_gr) * thlast)
    thlast_gr = thlast_gr.rename('t_hlast_GR')
    thmax_gr = np.sum((a_nl / a_gr) * thmax)
    thmax_gr = thmax_gr.rename('t_hmax_GR')

    return thfirst_gr, thlast_gr, thmax_gr


def run():
    opts = getopts()
    tea = load_tea_data(opts=opts)

    static = xr.open_dataset(f'/data/arsclisys/normal/clim-hydro/TEA-Indicators/static/'
                             f'static_Tx{opts.threshold}.0degC_{opts.region}_{opts.dataset}.nc')
    masks = xr.open_dataset(f'/data/arsclisys/normal/clim-hydro/TEA-Indicators/masks/'
                            f'{opts.region}_masks_{opts.dataset}.nc')

    thresh = static['threshold']
    area = static['area_grid']
    orog = masks['lt1500_mask']

    df = pd.DataFrame(columns=['doy', 'Nhours', 't_hfirst', 't_hlast', 't_hmax'])

    years = np.arange(1961, 2025)
    for iyr in years:
        # load ERA5 data, slice other data
        era5_hrl = load_era5_data(opts=opts, yr=iyr, reg=opts.region)
        tea_yr = tea.sel(days=slice(str(iyr) + '-01-01', str(iyr) + '-12-31'))

        # get exceedance days from TEA data
        ex_days = tea_yr.where((tea_yr.DTEC_GR == 1).compute(), drop=True).days

        # step through exceedance days
        for exdy in ex_days:
            era5_hrl_day = era5_hrl.sel(time=slice(exdy,
                                                   exdy + np.timedelta64(23, 'h')))

            tem = calc_hourly_tem(data=era5_hrl_day, thresh=thresh, orog=orog)

            n_hrs = exceedance_hours(tem=tem, area=area)
            tfirst, tlast, tmax = exceedance_times(tem=tem, area=area)

            # add values to df
            ex_date = pd.Timestamp(exdy.values).date()
            df.loc[ex_date, 'doy'] = ex_date.timetuple().tm_yday
            df.loc[ex_date, 'Nhours'] = n_hrs.values
            df.loc[ex_date, 't_hfirst'] = tfirst.values
            df.loc[ex_date, 't_hlast'] = tlast.values
            df.loc[ex_date, 't_hmax'] = tmax.values
            # parts of equation 16
            df.loc[ex_date, 'h_rise'] = df.loc[ex_date, 't_hmax'] - df.loc[
                ex_date, 't_hfirst'] + 0.5
            df.loc[ex_date, 'h_set'] = df.loc[ex_date, 't_hlast'] - df.loc[ex_date, 't_hmax'] + 0.5

    # compute CTP variables (Eq. 16)
    df.index = pd.to_datetime(df.index)
    df_ctp = df.resample('1AS-Apr').mean()
    df_ctp = df_ctp.drop(columns=['doy'])

    # get first and last exceedance doy (Eq. 13_4 & 13_5)
    df_doy_first = df.resample('1AS-Apr').min()
    df_doy_last = df.resample('1AS-Apr').max()
    df_ctp['doy_first'] = df_doy_first['doy'].values
    df_ctp['doy_last'] = df_doy_last['doy'].values

    # calc annual exposure period (Eq. 13_6)
    df_ctp['delta_y'] = (df_ctp['doy_last'] - df_ctp['doy_first'] + 1) / 30.5

    df_ctp = df_ctp.rename(columns={'Nhours': 'h_avg', 'h_rise': 'hrise_avg', 'h_set': 'hset_avg'})

    df_ctp.to_csv(f'{opts.outpath}'
                  f'SupplVars_EnergyContent_{opts.region}_Tx'
                  f'_{opts.threshold}.0degC_{opts.dataset}.csv')


if __name__ == '__main__':
    # prep_era5_data()
    # prep_era5land_data()
    run()
