import glob
import logging
from pathlib import Path
import re
import os
import xarray as xr
import warnings

from common.var_attrs import get_attrs
from common.general_functions import compare_to_ref
from common.TEA_logger import logger
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

    if 'Agr' in str(type(tea)):
        grg_str = 'GRG-'
    else:
        grg_str = ''
    if 'station' in opts:
        name = opts.station
    else:
        name = opts.region
    filenames = (f'{ctppath}/CTP_{opts.param_str}_{grg_str}{name}_{opts.period}'
                 f'_{opts.dataset}_*.nc')
    files = sorted(glob.glob(filenames))
    files = [file for file in files if is_in_period(filename=file, start=opts.start, end=opts.end) if 'ref' not in file]

    # TODO: optimize tea._calc_spread_estimators

    tea.load_ctp_results(files, use_dask=opts.use_dask)


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


def calc_decadal_indicators(opts, tea, outpath=None):
    """
    calculate decadal-mean ctp indicator variables (Eq. 23)
    Args:
        opts: CLI parameter
        tea: TEA object
        outpath: output path (default: opts.outpath/dec_indicator_variables/)

    Returns:

    """
    if outpath is None:
        if 'station' in opts:
            name = opts.station
        else:
            name = opts.region
        outpath = _get_decadal_outpath(opts, name)

    if opts.recalc_decadal or not os.path.exists(outpath):
        load_ctp_data(opts=opts, tea=tea)
        logger.info("Calculating decadal indicators")
        tea.calc_decadal_indicators(calc_spread=opts.spreads, drop_annual_results=True)
        path = Path(f'{opts.outpath}/dec_indicator_variables/')
        path.mkdir(parents=True, exist_ok=True)
        logger.info(f'Saving decadal indicators to {outpath}')
        tea.save_decadal_results(outpath)
    else:
        logger.info(f'Loading decadal indicators from {outpath}. To recalculate use --recalc-decadal')
        tea.load_decadal_results(outpath)

    if opts.compare_to_ref:
        file_ref = outpath.replace('.nc', '_ref.nc')
        compare_to_ref_decadal(tea=tea, filename_ref=file_ref)


def _get_decadal_outpath(opts, region):
    if 'agr' in opts:
        agr_str = 'AGR-'
    else:
        agr_str = ''
    outpath = (f'{opts.outpath}/dec_indicator_variables/'
               f'DEC_{opts.param_str}_{agr_str}{region}_{opts.period}_{opts.dataset}'
               f'_{opts.start}to{opts.end}.nc')
    return outpath


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


def calc_amplification_factors(opts, tea, outpath=None):
    """
    calculate amplification factors
    Args:
        opts: command line parameters
        tea: TEA object
        outpath: output path (default: opts.outpath/dec_indicator_variables/amplification/)

    Returns:

    """
    if outpath is None:
        if 'station' in opts:
            name = opts.station
        else:
            name = opts.region
        outpath = _get_amplification_outpath(opts, name)

    # calculate amplification factors
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        logger.info('Calculating amplification factors.')
        tea.calc_amplification_factors(ref_period=opts.ref_period, cc_period=opts.cc_period)

    path = Path(f'{opts.outpath}/dec_indicator_variables/amplification/')
    path.mkdir(parents=True, exist_ok=True)

    # compare to reference file
    if opts.compare_to_ref:
        ref_path = outpath.replace('.nc', '_ref.nc')
        ref_data = xr.open_dataset(ref_path)
        logger.info(f'Comparing amplification factors to reference file {ref_path}')
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            compare_to_ref(tea.amplification_factors, ref_data)

    # save amplification factors
    logger.info(f'Saving amplification factors to {outpath}')
    tea.save_amplification_factors(outpath)


def _get_amplification_outpath(opts, region):
    """
    get amplification factors output path
    Args:
        opts: options
        region: region name (str)

    Returns:
        outpath: output path (str)

    """
    if 'agr' in opts:
        agr_str = 'AGR-'
    else:
        agr_str = ''
    outpath = (f'{opts.outpath}/dec_indicator_variables/amplification/'
               f'AF_{opts.param_str}_{agr_str}{region}_{opts.period}_{opts.dataset}'
               f'_{opts.start}to{opts.end}.nc')
    return outpath
