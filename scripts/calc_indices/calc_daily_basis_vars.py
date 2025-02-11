import glob
import logging
import numpy as np
import os
from pathlib import Path
import xarray as xr
import numpy as np

from scripts.general_stuff.TEA_logger import logger
from TEA import TEAIndicators


def calc_daily_basis_vars(opts, static, data, mask):
    """
    compute daily basis variables following chapter 3 of TEA methods
    Args:
        opts: config parameters
        static: static ds
        data: data
        mask: mask

    Returns:
        TEA: TEA object

    """
    TEA = TEAIndicators(input_data_grid=data, threshold_grid=static['threshold'], area_grid=static['area_grid'],
                        mask=mask,
                        # set min area to < 1 grid cell area so that all exceedance days are considered
                        min_area=0.0001, low_extreme=opts.low_extreme, unit=opts.unit)
    
    logger.info('Calculating daily basis variables')
    TEA.calc_daily_basis_vars()
    
    bv_outpath = f'{opts.outpath}/daily_basis_variables'
    if not os.path.exists(bv_outpath):
        os.makedirs(bv_outpath)
    bv_filename = (f'{bv_outpath}/'
                   f'DBV_{opts.param_str}_{opts.region}_{opts.period}_{opts.dataset}'
                   f'_{opts.start}to{opts.end}.nc')

    logger.info(f'Saving daily basis variables to {bv_filename}')
    TEA.save_daily_results(bv_filename)
    
    return TEA
