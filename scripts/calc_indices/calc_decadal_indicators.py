import glob
import logging
import numpy as np
from pathlib import Path
import pandas as pd
import re
import sys
import xarray as xr

from scripts.general_stuff.var_attrs import get_attrs
from scripts.general_stuff.general_functions import create_history
from scripts.general_stuff.TEA_logger import logger

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


def load_ctp_data(opts, suppl, tea):
    """
    load CTP data
    Args:
        opts: CLI parameter
        suppl: set if supplementary variables should be processed
        tea: TEA object

    Returns:
        data: CTP ds
    """

    ctppath = f'{opts.outpath}ctp_indicator_variables/'

    sdir, suppl_str = '', ''
    if suppl:
        sdir = 'supplementary/'
        suppl_str = 'suppl'

    def is_in_period(filename, start, end):
        match = re.search(pattern=r'(\d{4})to(\d{4})', string=filename)
        if match:
            file_start, file_end = int(match.group(1)), int(match.group(2))
            return file_start <= end and file_end >= start
        else:
            return False
    
    filenames = (f'{ctppath}{sdir}CTP{suppl_str}_{opts.param_str}_{opts.region}_{opts.period}'
                 f'_{opts.dataset}_*_new.nc')
    files = sorted(glob.glob(filenames))
    files = [file for file in files if is_in_period(filename=file, start=opts.start, end=opts.end)]

    data = xr.open_mfdataset(paths=files, data_vars='minimal')
    tea.load_CTP_results(filenames)

    # check if more data than chosen period is loaded and select correct period if so
    if 'new' in filenames:
        date_index = -2
    else:
        date_index = -1
    syr, eyr = int(files[0].split('_')[date_index][:4]), int(files[-1].split('_')[date_index][6:10])
    if opts.start != syr or opts.end != eyr:
        data = data.sel(ctp=slice(f'{opts.start}-01-01', f'{opts.end}-12-31'))

    return data


def adjust_doy(data):
    """
    adjust doy_first(_GR) and doy_last(_GR) (Eq. 24)
    Args:
        data: ds

    Returns:
        data: ds with adjusted doy vars
    """
    data['doy_first'] = data['doy_first'] - 0.5 * (
            30.5 * data['AEP'] - (data['doy_last'] - data['doy_first'] + 1))
    data['doy_last'] = data['doy_last'] + 0.5 * (
            30.5 * data['AEP'] - (data['doy_last'] - data['doy_first'] + 1))

    if 'doy_first_GR' in data.data_vars:
        data['doy_first_GR'] = data['doy_first_GR'] - 0.5 * (
                30.5 * data['AEP_GR'] - (data['doy_last_GR'] - data['doy_first_GR'] + 1))
        data['doy_last_GR'] = data['doy_last_GR'] + 0.5 * (
                30.5 * data['AEP_GR'] - (data['doy_last_GR'] - data['doy_first_GR'] + 1))

    return data


def save_output(opts, data, su, sl, suppl):
    """
    save decadal-mean output
    Args:
        opts: CLI parameter
        data: ds
        su: upper spread ds
        sl: lower spread ds
        suppl: True if supplementary variables are processed

    Returns:

    """
    sdir, suppl_str = '', ''
    if suppl:
        sdir = 'supplementary/'
        suppl_str = 'suppl'

    data = create_history(cli_params=sys.argv, ds=data)

    path = Path(f'{opts.outpath}dec_indicator_variables/supplementary/')
    path.mkdir(parents=True, exist_ok=True)
    data.to_netcdf(f'{opts.outpath}dec_indicator_variables/{sdir}'
                   f'DEC{suppl_str}_{opts.param_str}_{opts.region}_{opts.period}_{opts.dataset}'
                   f'_{opts.start}to{opts.end}.nc')

    if su:
        sl = create_history(cli_params=sys.argv, ds=sl)
        su = create_history(cli_params=sys.argv, ds=su)
        sl.to_netcdf(f'{opts.outpath}dec_indicator_variables/{sdir}'
                     f'DEC{suppl_str}_sLOW_{opts.param_str}_{opts.region}_{opts.period}'
                     f'_{opts.dataset}_{opts.start}to{opts.end}.nc')
        su.to_netcdf(f'{opts.outpath}dec_indicator_variables/{sdir}'
                     f'DEC{suppl_str}_sUPP_{opts.param_str}_{opts.region}_{opts.period}'
                     f'_{opts.dataset}_{opts.start}to{opts.end}.nc')


def calc_spread_estimators(data, dec_data):
    """
    calculate spread estimator time series (Eq. 25)
    Args:
        data: non averaged data
        dec_data: decadal-mean data

    Returns:

    """

    supp, slow = xr.full_like(dec_data, np.nan), xr.full_like(dec_data, np.nan)
    for icy, cy in enumerate(data.time):
        # skip first and last 5 years
        if icy < 5 or icy > len(data.time) - 4:
            continue
        pdata = data.isel(time=slice(icy - 5, icy + 5))
        cupp = xr.where(pdata > dec_data.isel(time=icy), 1, 0)

        cupp_sum = cupp.sum(dim='time')
        cupp_sum = cupp_sum.where(cupp_sum > 0, 1)
        supp_per = np.sqrt((1 / (cupp_sum.max()))
                           * ((cupp * (data - dec_data.isel(time=icy)) ** 2).sum()))

        clow_sum = (1 - cupp).sum(dim='time')
        clow_sum = clow_sum.where(clow_sum > 0, 1)
        slow_per = np.sqrt((1 / (clow_sum.max()))
                           * (((1 - cupp) * (data - dec_data.isel(time=icy)) ** 2).sum()))

        supp.loc[{'time': cy}] = supp_per
        slow.loc[{'time': cy}] = slow_per

    for vvar in supp.data_vars:
        supp[vvar].attrs = get_attrs(vname=vvar, spread='upper')
    for vvar in slow.data_vars:
        supp[vvar].attrs = get_attrs(vname=vvar, spread='lower')

    rename_dict_supp = {vvar: f'{vvar}_supp' for vvar in supp.data_vars}
    rename_dict_slow = {vvar: f'{vvar}_slow' for vvar in slow.data_vars}
    supp = supp.rename(rename_dict_supp)
    slow = slow.rename(rename_dict_slow)

    return supp, slow


def rolling_decadal_mean(data):
    """
    apply rolling decadal mean
    Args:
        data: annual data

    Returns:
        data: decadal-mean data
    """

    weights = xr.DataArray([1, 1, 1, 1, 1, 1, 1, 1, 1, 1], dims=['window']) / 10

    # equation 23 (decadal averaging)
    for vvar in data.data_vars:
        data[vvar] = data.rolling(time=10, center=True).construct('window')[vvar].dot(
            weights)
        data[vvar].attrs = get_attrs(vname=vvar, dec=True)

    return data


def calc_decadal_indicators(opts, suppl=False, tea=None):
    """
    calculate decadal-mean ctp indicator variables (Eq. 23)
    Args:
        opts: CLI parameter
        suppl: set if supplementary variables should be processed
        tea: TEA object

    Returns:

    """
    data = load_ctp_data(opts=opts, suppl=suppl, tea=tea)
    logger.info("Calculating decadal indicators")
    tea.calc_decadal_indicators(calc_spread=opts.spreads)

    dec_data = data.copy()

    dec_data = rolling_decadal_mean(data=dec_data)

    # equation 24 (re-adjusting doy vars)
    if 'doy_first' in data.data_vars:
        dec_data = adjust_doy(data=dec_data)

    su, sl = None, None
    if opts.spreads:
        logging.info(f'Calculating spread estimators.')
        su, sl = calc_spread_estimators(data=data, dec_data=dec_data)

    save_output(opts=opts, data=dec_data, su=su, sl=sl, suppl=suppl)
