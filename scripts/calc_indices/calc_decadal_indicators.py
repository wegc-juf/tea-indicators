import glob
import logging
import numpy as np
from pathlib import Path
import pandas as pd
import re
import sys, os
import xarray as xr
import warnings

from scripts.general_stuff.var_attrs import get_attrs
from scripts.general_stuff.general_functions import create_history, compare_to_ref
from scripts.general_stuff.TEA_logger import logger
from TEA import TEAIndicators

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


def load_ctp_data(opts, tea):
    """
    load CTP data
    Args:
        opts: CLI parameter
        tea: TEA object

    Returns:
        data: CTP ds
    """

    ctppath = f'{opts.outpath}/ctp_indicator_variables/'

    def is_in_period(filename, start, end):
        match = re.search(pattern=r'(\d{4})to(\d{4})', string=filename)
        if match:
            file_start, file_end = int(match.group(1)), int(match.group(2))
            return file_start <= end and file_end >= start
        else:
            return False
    
    filenames = (f'{ctppath}CTP_{opts.param_str}_{opts.region}_{opts.period}'
                 f'_{opts.dataset}_*_new.nc')
    files = sorted(glob.glob(filenames))
    files = [file for file in files if is_in_period(filename=file, start=opts.start, end=opts.end)]

    tea.load_CTP_results(files)


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


def save_output(opts, data, su, sl):
    """
    save decadal-mean output
    Args:
        opts: CLI parameter
        data: ds
        su: upper spread ds
        sl: lower spread ds

    Returns:

    """
    data = create_history(cli_params=sys.argv, ds=data)

    path = Path(f'{opts.outpath}/dec_indicator_variables/')
    path.mkdir(parents=True, exist_ok=True)
    data.to_netcdf(f'{opts.outpath}dec_indicator_variables/'
                   f'DEC_{opts.param_str}_{opts.region}_{opts.period}_{opts.dataset}'
                   f'_{opts.start}to{opts.end}.nc')

    if su:
        sl = create_history(cli_params=sys.argv, ds=sl)
        su = create_history(cli_params=sys.argv, ds=su)
        sl.to_netcdf(f'{opts.outpath}dec_indicator_variables/'
                     f'DEC_sLOW_{opts.param_str}_{opts.region}_{opts.period}'
                     f'_{opts.dataset}_{opts.start}to{opts.end}.nc')
        su.to_netcdf(f'{opts.outpath}dec_indicator_variables/'
                     f'DEC_sUPP_{opts.param_str}_{opts.region}_{opts.period}'
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
        supp_per = np.sqrt((1 / cupp_sum)
                           * ((cupp * (pdata - dec_data.isel(time=icy)) ** 2).sum()))

        clow_sum = (1 - cupp).sum(dim='time')
        clow_sum = clow_sum.where(clow_sum > 0, 1)
        slow_per = np.sqrt((1 / clow_sum)
                           * (((1 - cupp) * (pdata - dec_data.isel(time=icy)) ** 2).sum()))

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


def calc_decadal_indicators(opts, tea):
    """
    calculate decadal-mean ctp indicator variables (Eq. 23)
    Args:
        opts: CLI parameter
        tea: TEA object

    Returns:

    """
    outpath_new = (f'{opts.outpath}/dec_indicator_variables/'
                   f'DEC_{opts.param_str}_{opts.region}_{opts.period}_{opts.dataset}'
                   f'_{opts.start}to{opts.end}_new.nc')
    if opts.recalc_decadal or not os.path.exists(outpath_new):
        load_ctp_data(opts=opts, tea=tea)
        logger.info("Calculating decadal indicators")
        tea.calc_decadal_indicators(calc_spread=opts.spreads, drop_annual_results=True)
        path = Path(f'{opts.outpath}/dec_indicator_variables/')
        path.mkdir(parents=True, exist_ok=True)
        logger.info(f'Saving decadal indicators to {outpath_new}')
        tea.save_decadal_results(outpath_new)
    else:
        logger.info(f'Loading decadal indicators from {outpath_new}. To recalculate use --recalc-decadal')
        tea.load_decadal_results(outpath_new)

    if opts.compare_to_ref:
        file_ref = outpath_new.replace('.nc', '_ref.nc')
        compare_to_ref_decadal(tea=tea, filename_ref=file_ref)


def compare_to_ref_decadal(tea, filename_ref):
    """
    compare results to reference file
    TODO: move this to test routine
    Args:
        tea: TEA object
        filename_ref: reference file
    """
    if os.path.exists(filename_ref):
        logger.info(f'Comparing results to reference file {filename_ref}')
        tea_ref = TEAIndicators()
        tea_ref.load_decadal_results(filename_ref)
        for vvar in tea.decadal_results.data_vars:
            attrs = tea.decadal_results[vvar].attrs
            if vvar in tea_ref.decadal_results.data_vars:
                diff = tea.decadal_results[vvar] - tea_ref.decadal_results[vvar]
                max_diff = diff.max(skipna=True).values
                if max_diff > 1e-6:
                    logger.warning(f'Maximum difference in {vvar} is {max_diff}')
            else:
                logger.warning(f'{vvar} not found in reference file.')
    else:
        logger.warning(f'Reference file {filename_ref} not found.')


def calc_amplification_factors(opts, tea):
    """
    calculate amplification factors
    Args:
        opts: command line parameters
        tea: TEA object

    Returns:

    """
    # calculate amplification factors
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        logger.info('Calculating amplification factors.')
        tea.calc_amplification_factors()
    
    path = Path(f'{opts.outpath}/dec_indicator_variables/amplification/')
    path.mkdir(parents=True, exist_ok=True)
    out_path = (f'{opts.outpath}/dec_indicator_variables/amplification/'
                f'AF_{opts.param_str}_{opts.region}_{opts.period}_{opts.dataset}'
                f'_{opts.start}to{opts.end}_new.nc')
    
    # compare to reference file
    if opts.compare_to_ref:
        ref_path = out_path.replace('_new.nc', '_new_ref.nc')
        ref_data = xr.open_dataset(ref_path)
        logger.info(f'Comparing amplification factors to reference file {ref_path}')
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            compare_to_ref(tea.amplification_factors, ref_data)
    
    # save amplification factors
    logger.info(f'Saving amplification factors to {out_path}')
    tea.save_amplification_factors(out_path)


