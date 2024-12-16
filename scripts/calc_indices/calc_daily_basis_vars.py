import glob
import logging
import numpy as np
import os
from pathlib import Path
import xarray as xr
import numpy as np

from scripts.general_stuff.var_attrs import get_attrs
from scripts.general_stuff.TEA_logger import logger
from TEA import TEAIndicators


def check_tmp_dir(opts):
    """
    check if tmp directory is empty and ask if files should be deleted
    Args:
        opts: CLI parameter

    Returns:

    """

    def is_directory_empty(directory):
        return not any(os.scandir(directory))

    def delete_files_in_directory(directory):
        for entry in os.scandir(directory):
            if entry.is_file():
                os.remove(entry.path)

    tmp_dir = f'{opts.outpath}daily_basis_variables/tmp/'

    # Check each directory and interact with the user if necessary
    if not is_directory_empty(tmp_dir):
        logging.info(f'Tmp directory is not empty, files will be deleted first.')
        delete_files_in_directory(tmp_dir)


def calc_daily_basis_vars(opts, static, data, large_gr=False, cell=None, mask=None):
    """
    compute daily basis variables following chapter 3 of TEA methods
    Args:
        opts: CLI parameter
        static: static ds
        data: data
        large_gr: set if called from calc_TEA_largeGR (saves output in different directory)
        cell: lat and lon of cell (only relevant if called from calc_TEA_largeGR).
        mask: mask grid for masking out regions (nan values are masked out)

    Returns:
        basic_vars: ds with daily basis variables (DTEC, DTEM, DTEA) both gridded and for GR
        dtem_max: da daily maximum threshold exceedance magnitude

    """

    cell_str = ''
    if large_gr:
        cell_str = f'_lat{cell[0]}_lon{cell[1]}'

    path = Path(f'{opts.outpath}/daily_basis_variables/tmp/')
    path.mkdir(parents=True, exist_ok=True)

    # check if tmp directory is empty
    if not large_gr:
        check_tmp_dir(opts)

    TEA = TEAIndicators(input_data_grid=data, threshold_grid=static['threshold'], area_grid=static['area_grid'],
                        mask=mask,
                        # set min area to < 1 grid cell area so that all exceedance days are considered
                        min_area=0.0001, low_extreme=opts.low_extreme, unit=opts.unit)
    logger.info('Calculating daily basis variables')
    TEA.calc_daily_basis_vars()
    
    # get custom attributes
    
    bv_outpath = (f'{opts.outpath}/daily_basis_variables/'
                  f'DBV_{opts.param_str}_{opts.region}_{opts.period}_{opts.dataset}'
                  f'_{opts.start}to{opts.end}_new.nc')
    if large_gr:
        bv_outpath = (f'{opts.tmppath}/daily_basis_variables/'
                      f'DBV{cell_str}_{opts.param_str}_{opts.region}_{opts.period}_{opts.dataset}'
                      f'_{opts.start}to{opts.end}_new.nc')

    logger.info(f'Saving daily basis variables to {bv_outpath}')
    TEA.save_daily_results(bv_outpath)
    
    return TEA
