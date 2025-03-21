import matplotlib.pyplot as plt
import pandas as pd
import pyreadr
import xarray as xr


def prep_hourly_data(provider='gsa'):
    data = pyreadr.read_r(f'/data/users/hst/additional_stuff/heavyrainfall_Haslinger2025/data/raw/'
                          f'ts_hourly_{provider}.rds')
    df = data[None]

    time = sorted(list(set(df['date'])))
    stations = list(set(df['id']))
    stations = sorted([int(istation[3:]) for istation in stations])

    df_out = pd.DataFrame(index=time, columns=stations)

    str_stat = 'gh_'
    if provider == 'hyd':
        str_stat = 'hh_'

    for station in stations:
        df_station = df.loc[df['id'] == f'{str_stat}{station}']
        df_out.loc[df_station['date'], station] = df_station['rr'].values

    df_out.to_csv(f'/data/users/hst/additional_stuff/heavyrainfall_Haslinger2025/data/raw/TEAprep/'
                  f'ts_hourly_{provider}.csv')


def check_station_syr(provider='gsa'):
    df = pd.read_csv(f'/data/users/hst/additional_stuff/heavyrainfall_Haslinger2025/'
                     f'data/raw/TEAprep/ts_hourly_{provider}.csv', index_col=0)
    df = df.iloc[1:, :]
    syrs = pd.DataFrame(index=df.columns, columns=['syr'])

    for station in df.columns:
        sdf = df.loc[:, station]
        sdf = sdf.loc[sdf.notnull()]
        syr = sdf.index[0][:4]
        syrs.loc[station, 'syr'] = int(syr)

    plt.hist(syrs['syr'], bins=range(1940, 2025, 5))
    plt.show()


def get_fbr_sea_stations(region='FBR'):
    data = pyreadr.read_r(f'/data/users/hst/additional_stuff/heavyrainfall_Haslinger2025/data/raw/'
                          f'station_locations.rds')
    df = data[None]

    # add station IDs
    df = df.loc[df['timescale'] == 'hourly']
    df_gsa = df.loc[df['provider'] == 'GSA']
    df_hyd = df.loc[df['provider'] == 'HYD']
    ids = []
    for igsa in range(len(df_gsa.index)):
        ids.append(f'gsa_{igsa + 1}')
    for ihyd in range(len(df_hyd.index)):
        ids.append(f'hyd_{ihyd + 1}')
    df.loc[:, 'id'] = ids

    # load region mask
    reg = xr.open_dataset(f'/data/arsclisys/normal/clim-hydro/TEA-Indicators/masks/'
                          f'{region}_masks_ERA5Land.nc')
    reg = reg.nw_mask.where(reg.nw_mask == 1, drop=True)
    reg_lats = [reg.lat.min().values, reg.lat.max().values]
    reg_lons = [reg.lon.min().values, reg.lon.max().values]

    # find station within region
    stations = df.loc[df['lat'] >= reg_lats[0]]
    stations = stations.loc[stations['lat'] <= reg_lats[1]]
    stations = stations.loc[stations['lon'] >= reg_lons[0]]
    stations = stations.loc[stations['lon'] <= reg_lons[1]]

    stations.to_csv(f'/data/users/hst/additional_stuff/heavyrainfall_Haslinger2025/data/raw/TEAprep/'
                    f'{region}_stations.csv')


def add_gsa_altitude_to_metadata():
    data = pyreadr.read_r(f'/data/users/hst/additional_stuff/heavyrainfall_Haslinger2025/data/raw/'
                          f'station_locations.rds')
    df = data[None]
    df = df.loc[df['provider'] == 'GSA']

    gsa = pd.read_csv('/data/users/hst/cdrDPS/station_data/metadata/'
                      'stations_metadata.csv',
                      index_col=0)

    alts = []
    for istation in df.index:
        lat, lon = df.loc[istation, 'lat'], df.loc[istation, 'lon']
        lat_diffs = abs(gsa.loc[:, 'Breite [°N]'] - lat)
        lon_diffs = abs(gsa.loc[:, 'Länge [°E]'] - lon)
        diff = lat_diffs + lon_diffs
        idx_min = diff.idxmin()
        alts.append(gsa.loc[idx_min, 'Höhe [m]'])

    df['altitude'] = alts

    df.to_csv('/data/users/hst/additional_stuff/heavyrainfall_Haslinger2025/data/raw/TEAprep/'
              'stations_metadata.csv')


if __name__ == '__main__':
    # prep_hourly_data(provider='gsa')
    # add_gsa_altitude_to_metadata()
    # check_station_syr(provider='hyd')
    get_fbr_sea_stations(region='FBR')
