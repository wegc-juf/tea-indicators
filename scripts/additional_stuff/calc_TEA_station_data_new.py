"""
@author: hst
calculate TEA indicators with data from Haslinger et al. 2025
"""

import datetime as dt
import numpy as np
import os
import pandas as pd
from scipy.stats import gmean
import sys
import xarray as xr

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from scripts.general_stuff.general_functions import ref_cc_params
from scripts.general_stuff.var_attrs import get_attrs
from scripts.calc_indices.calc_daily_basis_vars import calc_dteec_1d

PARAMS = ref_cc_params()


def load_data(region, pvar):
    """
    load station data
    Args:

    Returns:
        data: interpolated station data

    """

    if region != 'AUT':
        stations = pd.read_csv(
            f'/data/users/hst/additional_stuff/heavyrainfall_Haslinger2025/data/raw/'
            f'TEAprep/{region}_stations.csv', index_col=0)
        if len(stations.index) == 0:
            raise KeyError(f'There are no stations in {region} available!')

    data_hyd = pd.read_csv(
        f'/data/users/hst/additional_stuff/heavyrainfall_Haslinger2025/data/raw/'
        f'TEAprep/ts_hourly_hyd.csv', index_col=0)
    data_hyd['time'] = pd.to_datetime(data_hyd.index)
    data_hyd = data_hyd.set_index('time')
    col_names_hyd = [f'hyd_{istation}' for istation in data_hyd.columns]
    data_hyd.columns = col_names_hyd

    data_gsa = pd.read_csv(
        f'/data/users/hst/additional_stuff/heavyrainfall_Haslinger2025/data/raw/'
        f'TEAprep/ts_hourly_gsa.csv', index_col=0)
    data_gsa['time'] = pd.to_datetime(data_gsa.index)
    data_gsa = data_gsa.set_index('time')
    col_names_gsa = [f'gsa_{istation}' for istation in data_gsa.columns]
    data_gsa.columns = col_names_gsa

    data = pd.concat([data_hyd, data_gsa], axis=1)

    # remove line with NaT as index
    data = data.iloc[:-1, :]

    # select relevant stations
    if region != 'AUT':
        data = data.loc[:, stations['id'].values]

    # shift data by 7h to get 7to7 data
    data = data.shift(-7, freq='H')

    if pvar == 'P24h_7to7':
        # resample to daily data and only keep wet days
        data = data.resample('D').sum()
        data = data.where(data >= 1)
    else:
        # resample to daily data and only keep wet hours
        data = data.resample('D').max()
        data = data.where(data >= 0.3)

    # calculate threshold value
    ref_data = data[(data.index.year >= int(PARAMS['REF']['start'][:4]))
                    & (data.index.year <= int(PARAMS['REF']['end'][:4]))]
    threshs = ref_data.quantile(0.95)

    # only select WAS data
    data = data[(data.index.month >= 4) & (data.index.month <= 10)]

    return data, threshs


def calculate_event_count(dtec, gr=False):
    """
    calculate DTEEC(_GR) according to equations 4 and 5
    Args:
        dtec: daily threshold exceedance count
        gr: set if GR data is used

    Returns:

    """

    if gr:
        dtec = xr.DataArray(data=dtec.values,
                            coords={'days': (['days'], dtec.index.values)})
        dteec_np = calc_dteec_1d(dtec_cell=dtec.values)
        dteec = xr.DataArray(dteec_np, coords=dtec.coords, dims=dtec.dims)
        gr_var_str = '_GR'
    else:
        dtec = xr.DataArray(data=dtec.values,
                            coords={'days': (['days'], dtec.index.values),
                                'stations': (['stations'], dtec.columns)})
        dteec = xr.full_like(dtec, np.nan)
        dtec_2d = dtec.values
        # loop through all rows and calculate DTEEC
        for iy in range(len(dtec_2d[0, :])):
            dtec_row = dtec_2d[:, iy]
            # skip all nan rows
            if np.isnan(dtec_row).all():
                continue
            dteec_row = np.apply_along_axis(calc_dteec_1d, axis=0, arr=dtec_row)
            dteec[:, iy] = dteec_row
        gr_var_str = ''

    dteec = dteec.rename(f'DTEEC{gr_var_str}')

    return dteec


def calc_daily_basis_vars(data, thresh):
    """
    calculate daily basis variables
    Args:
        data: daily station data
        thresh: 95th percentile of reference period (threshold for exceedance)

    Returns:
        basics: daily basis variables
        data: updated data

    """

    # equation 01
    dtec = data.where(data > thresh, 0.0)
    dtec = dtec.where(dtec == 0, 1.0)
    dtec = dtec.where(dtec == 1)

    # calculate dtec_gr (equation 03)
    dtec_gr = dtec.max(axis=1)

    # equation 07
    dtem = data - thresh
    dtem = dtem.where(dtem > 0, 0.0)

    dtem_gr = dtem.sum(axis=1)

    # calc dteec
    dteec = calculate_event_count(dtec=dtec, gr=False)
    dteec_gr = calculate_event_count(dtec=dtec_gr, gr=True)

    basics = xr.Dataset(data_vars={'DTEC': (['days', 'stations'], dtec),
                                   'DTEC_GR': (['days'], dtec_gr),
                                   'DTEM': (['days', 'stations'], dtem),
                                   'DTEM_GR': (['days'], dtem_gr),
                                   'DTEEC': (['days', 'stations'], dteec.values),
                                   'DTEEC_GR': (['days'], dteec_gr.values)},
                        coords={'days': (['days'], data.index.values),
                                'stations': (['stations'], data.columns)})

    basics['DTEM'] = basics['DTEM'].where(basics['DTEM'] > 0)
    basics['DTEM_GR'] = basics['DTEM_GR'].where(basics['DTEM_GR'] > 0)

    return basics


def assign_ctp_coords(data):
    """
    create dictionary of all start & end dates, the chosen frequency and period
    Args:
        data: data array

    Returns:

    """

    freq = 'AS-APR'

    pstarts = pd.date_range(data.days[0].values, data.days[-1].values,
                            freq=freq).to_series()
    pends = pd.date_range(data.days[0].values, data.days[-1].values,
                              freq='A-OCT').to_series()

    # add ctp as coordinates to enable using groupby later
    # map the 'days' coordinate to 'ctp'
    def map_to_ctp(dy, starts, ends):
        for start, end, ctp in zip(starts, ends, starts):
            if start <= dy <= end:
                return ctp
        return np.nan

    days_to_ctp = []
    for day in data.days.values:
        ctp_dy = map_to_ctp(dy=day, starts=pstarts, ends=pends)
        days_to_ctp.append(ctp_dy)

    data.coords['ctp'] = ('days', days_to_ctp)

    # group into CTPs
    data_per = data.groupby('ctp')

    return data, data_per


def calc_ctp_indicators(data, region, pvar):

    pdata = data.sum('days')

    ef = pdata.DTEEC
    ef_gr = pdata.DTEEC_GR

    ed = pdata['DTEC']
    ed_gr = pdata['DTEC_GR']
    ed_avg = ed / ef
    ed_avg_gr = ed_gr / ef_gr

    em_avg = data.median('days')['DTEM']
    em_avg_gr = data.median('days')['DTEM_GR']
    em_avg = em_avg.interpolate_na(dim='ctp')
    em_avg_gr = em_avg_gr.interpolate_na(dim='ctp')

    # add attributes and combine to one dataset
    ef = ef.rename('EF')
    ef = ef.assign_attrs(get_attrs(vname='EF'))
    ef_gr = ef_gr.rename('EF_GR')
    ef_gr = ef_gr.assign_attrs(get_attrs(vname='EF_GR'))

    ed_avg = ed_avg.rename('EDavg')
    ed_avg.attrs = get_attrs(vname='EDavg')
    ed_avg_gr = ed_avg_gr.rename('EDavg_GR')
    ed_avg_gr.attrs = get_attrs(vname='EDavg_GR')

    em_avg = em_avg.rename('EMavg')
    em_avg.attrs = get_attrs(vname='EMavg')
    em_avg_gr = em_avg_gr.rename('EMavg_GR')
    em_avg_gr.attrs = get_attrs(vname='EMavg_GR')

    ctp = xr.merge([ef, ef_gr, ed_avg, ed_avg_gr, em_avg, em_avg_gr])

    # calc compound vars
    tex = ctp.EF * ctp.EDavg * ctp.EMavg
    tex = tex.rename('tEX')
    ctp['tEX'] = tex

    tex_gr = ctp.EF_GR * ctp.EDavg_GR * ctp.EMavg_GR
    tex_gr = tex_gr.rename('tEX_GR')
    ctp['tEX_GR'] = tex_gr

    fm = ctp.EF * ctp.EMavg
    fm = fm.rename('FM')
    ctp['FM'] = fm

    fm_gr = ctp.EF_GR * ctp.EMavg_GR
    fm_gr = fm_gr.rename('FM_GR')
    ctp['FM_GR'] = fm_gr

    # save output
    history = f'{dt.datetime.now():%FT%H:%M:%S} calc_TEA_station_data.py'
    ctp.attrs['history'] = history
    ctp.to_netcdf(f'/data/users/hst/additional_stuff/heavyrainfall_Haslinger2025/data/TEA/'
                  f'CTP_{region}_{pvar}.nc')

    return ctp


def calc_dec_indicators(data, region, pvar):
    # drop compund vars
    data = data.drop_vars(['tEX', 'tEX_GR', 'FM', 'FM_GR'])
    weights = xr.DataArray([1, 1, 1, 1, 1, 1, 1, 1, 1, 1], dims=['window']) / 10

    # equation 23 (decadal averaging)
    for vvar in data.data_vars:
        data[vvar] = data.rolling(ctp=10, center=True).construct('window')[vvar].dot(
            weights)
        data[vvar].attrs = get_attrs(vname=vvar, dec=True)
        
    # add compound vars again
    tex = data.EF * data.EDavg * data.EMavg
    tex = tex.rename('tEX')
    data['tEX'] = tex

    tex_gr = data.EF_GR * data.EDavg_GR * data.EMavg_GR
    tex_gr = tex_gr.rename('tEX_GR')
    data['tEX_GR'] = tex_gr

    fm = data.EF * data.EMavg
    fm = fm.rename('FM')
    data['FM'] = fm

    fm_gr = data.EF_GR * data.EMavg_GR
    fm_gr = fm_gr.rename('FM_GR')
    data['FM_GR'] = fm_gr
    
    history = f'{dt.datetime.now():%FT%H:%M:%S} calc_TEA_station_data.py'
    data.attrs['history'] = history
    data.to_netcdf(f'/data/users/hst/additional_stuff/heavyrainfall_Haslinger2025/data/TEA/'
                   f'DEC_{region}_{pvar}.nc')

    return data


def calc_af(data, region, pvar):
    ref_ds = data.sel(ctp=slice(PARAMS['REF']['start_cy'], PARAMS['REF']['end_cy']))
    
    af = xr.full_like(data, np.nan)
    for vvar in data.data_vars:
        if vvar in ['tEX', 'tEX_GR', 'FM', 'FM_GR']:
            continue
        ref = gmean(ref_ds[vvar])
        af[vvar] = data[vvar] / ref
        
    # calc compound vars
    tex = af.EF * af.EDavg * af.EMavg
    tex = tex.rename('tEX')
    af['tEX'] = tex

    tex_gr = af.EF_GR * af.EDavg_GR * af.EMavg_GR
    tex_gr = tex_gr.rename('tEX_GR')
    af['tEX_GR'] = tex_gr

    fm = af.EF * af.EMavg
    fm = fm.rename('FM')
    af['FM'] = fm

    fm_gr = af.EF_GR * af.EMavg_GR
    fm_gr = fm_gr.rename('FM_GR')
    af['FM_GR'] = fm_gr

    rename_dict = {vname: f'{vname}_AF' for vname in af.data_vars}
    af = af.rename(rename_dict)

    for vvar in af.data_vars:
        af_cc = gmean(af[vvar].sel(ctp=slice(PARAMS['CC']['start_cy'], PARAMS['CC']['end_cy'])))
        af[f'{vvar}_CC'] = af_cc

    history = f'{dt.datetime.now():%FT%H:%M:%S} calc_TEA_station_data.py'
    af.attrs['history'] = history
    af.to_netcdf(f'/data/users/hst/additional_stuff/heavyrainfall_Haslinger2025/data/TEA/'
                 f'AF_{region}_{pvar}.nc')


def run():
    region = 'SEA'
    pvar = 'Px1h_7to7'

    data, p95 = load_data(region=region, pvar=pvar)

    # calc DBV variables
    dbv = calc_daily_basis_vars(data=data, thresh=p95)
    dbv, dbv_per = assign_ctp_coords(data=dbv)

    # calc CTP variables
    ctp = calc_ctp_indicators(data=dbv_per, region=region, pvar=pvar)
    
    # calc DEC variables
    dec = calc_dec_indicators(data=ctp, region=region, pvar=pvar)

    # calc AF
    calc_af(data=dec, region=region, pvar=pvar)


if __name__ == '__main__':
    run()
