"""
@author: hst
"""

import datetime as dt
import glob
import numpy as np
import xarray as xr


def calc_regional_ahc(ahc_zall):
    """
    calculate AHC for predefined regions
    :param ahc_zall: full AHC (surface upwards)
    :return: zAll AHC for different regions (dict)
    """
    # define regions
    regions = ['NH20to90N', 'NH35to70N', 'GLOB']
    lats = {'ET': [20, 90], 'ML': [35, 70], 'GLOB': [-90, 90]}
    lons = {'ET': [-180, 180], 'ML': [-180, 180], 'GLOB': [-180, 180]}

    reg_ahc = {}
    ahc_zall_eur_map = None
    for ireg, reg in enumerate(lats.keys()):
        reg_ahc[reg] = {}
        ahc_zall_reg = ahc_zall.sel(latitude_bins=slice(lats[reg][0], lats[reg][1]),
                                    longitude_bins=slice(lons[reg][0], lons[reg][1]))
        reg_ahc[reg][f'ahc_zall_{regions[ireg]}'] = ahc_zall_reg.sum(dim=('latitude_bins',
                                                                          'longitude_bins'))

    return reg_ahc, ahc_zall_eur_map


def calc_anomaly(data):
    """
    calculate AHC anomaly
    :param data: AHC data
    :return: AHC anomaly
    """

    data8022 = data.sel(time=slice('1980-01-01', '2022-12-31'))
    anom = data.groupby('time.month') - data8022.groupby('time.month').mean('time')
    anom = anom.rename('ahc_anomaly')

    return anom


def run():
    data = xr.open_dataset('/data/users/max/backup/4_hst/20250806_ahc/'
                           'wegc_era5_ahc_gridded_194001-202504_created-20250806-191551.nc')

    data = data['ahc'][0, :, :, :].sel(time=slice('1961-01-01', '2024-12-31'))

    regional_ahc, ahc_zall_eur_map = calc_regional_ahc(ahc_zall=data)

    anom = xr.Dataset()
    for ireg in regional_ahc.keys():
        var = list(regional_ahc[ireg].keys())[0]
        ahc_anom = calc_anomaly(data=regional_ahc[ireg][var])
        ahc_anom_was = ahc_anom.where(ahc_anom.time.dt.month.isin(np.arange(4, 11)), drop=True)

        reg = var.split('_')[-1]
        anom[f'ahc_anom_{reg}_ANN'] = ahc_anom.resample(time='1YS').mean()
        anom[f'ahc_anom_{reg}_WAS'] = ahc_anom_was.resample(time='1YS').mean()

    anom['history'] = f'{dt.datetime.now():%FT%H:%M:%S} prepare_AHC_data.py'
    anom.to_netcdf(f'/data/users/hst/cdrDPS/AHC/ahc_anomalies_1961to2024.nc')

def compare_old_new():
    old = xr.open_dataset('/data/users/hst/cdrDPS/AHC/OLD/AHC_anomalies_1960to2022.nc')
    new = xr.open_dataset('/data/users/hst/cdrDPS/AHC/ahc_anomalies_1960to2024.nc')

    import matplotlib.pyplot as plt
    plt.figure()
    plt.plot(old['ahc_anom_zall_GLOB_ANN'].time, old['ahc_anom_zall_GLOB_ANN'] / 1e21, label='old')
    plt.plot(new['ahc_anom_GLOB_ANN'].time, new['ahc_anom_GLOB_ANN'], label='new')
    plt.show()


if __name__ == "__main__":
    run()
