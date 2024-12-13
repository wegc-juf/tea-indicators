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


def save_dtec_dtea(opts, tea, static, cstr):
    """
    save DTEC, DTEC_GR, and DTEA_GR to tmp files
    Args:
        opts: CLI parameter
        tea: TEA object
        static: static files
        cstr: cell string to add to filename (only if called from calc_TEA_largeGR)
    """
    dtem = tea.daily_results.DTEM
    dtec = tea.daily_results.DTEC
    dtea = tea.daily_results.DTEA
    dtea_gr = tea.daily_results.DTEA_GR
    dtec_gr = tea.daily_results.DTEC_GR
    
    outpath = f'{opts.outpath}/daily_basis_variables/tmp/'
    
    # check if outpath exists and create it if not
    Path(outpath).mkdir(parents=True, exist_ok=True)

    # calc area fraction
    area_frac = (dtea_gr / static['GR_size']) * 100
    area_frac = area_frac.rename('DTEA_frac')

    areas = xr.merge([dtea_gr, area_frac])
    areas.to_netcdf(f'{outpath}DTEA{cstr}_{opts.param_str}_{opts.region}_{opts.period}_{opts.dataset}'
                    f'_{opts.start}to{opts.end}.nc')

    dtecs = xr.merge([dtec, dtec_gr])
    dtecs.to_netcdf(f'{outpath}DTEC{cstr}_{opts.param_str}_{opts.region}_{opts.period}_{opts.dataset}'
                    f'_{opts.start}to{opts.end}.nc')


def save_event_count(opts, tea, cstr):
    """
    save DTEEC(_GR) to netcdf file
    Args:
        opts: CLI parameter
        tea: TEA object
        cstr: string for subcell
    """
    vars = ['DTEEC', 'DTEEC_GR']
    for var in vars:
        if 'GR' in var:
            dteec = tea.daily_results.DTEEC_GR
        else:
            dteec = tea.daily_results.DTEEC

        outname = (f'{opts.outpath}daily_basis_variables/tmp/'
                   f'{dteec.name}{cstr}_{opts.param_str}_{opts.region}_{opts.period}_{opts.dataset}'
                   f'_{opts.start}to{opts.end}.nc')
        dteec.to_netcdf(outname)


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


def calc_daily_basis_vars(opts, static, data, large_gr=False, cell=None):
    """
    compute daily basis variables following chapter 3 of TEA methods
    Args:
        opts: CLI parameter
        static: static ds
        data: data
        large_gr: set if called from calc_TEA_largeGR (saves output in different directory)
        cell: lat and lon of cell (only relevant if called from calc_TEA_largeGR).

    Returns:
        basic_vars: ds with daily basis variables (DTEC, DTEM, DTEA) both gridded and for GR
        dtem_max: da daily maximum threshold exceedance magnitude

    """

    cell_str = ''
    if large_gr:
        cell_str = f'_lat{cell[0]}_lon{cell[1]}'

    path = Path(f'{opts.outpath}daily_basis_variables/tmp/')
    path.mkdir(parents=True, exist_ok=True)

    # check if tmp directory is empty
    if not large_gr:
        check_tmp_dir(opts)

    TEA = TEAIndicators(input_data_grid=data, threshold_grid=static['threshold'], area_grid=static['area_grid'],
                        # set min area to < 1 grid cell area so that all exceedance days are considered
                        min_area=0.0001, low_extreme=opts.low_extreme, unit=opts.unit)
    logger.info('Calculating daily basis variables')
    TEA.calc_daily_basis_vars()
    
    # get custom attributes
    
    bv_outpath = (f'{opts.outpath}/daily_basis_variables/'
                  f'DBV_{opts.param_str}_{opts.region}_{opts.period}_{opts.dataset}'
                  f'_{opts.start}to{opts.end}.nc')
    if large_gr:
        large_gr_path = Path(f'{opts.tmppath}/daily_basis_variables/')
        large_gr_path.mkdir(parents=True, exist_ok=True)
        bv_outpath = (f'{opts.tmppath}daily_basis_variables/'
                      f'DBV{cell_str}_{opts.param_str}_{opts.region}_{opts.period}_{opts.dataset}'
                      f'_{opts.start}to{opts.end}.nc')

    if opts.save_old:
        # combine all basic variables into one ds
        bv_files = sorted(glob.glob(
            f'{opts.outpath}/daily_basis_variables/tmp/'
            f'*{cell_str}_{opts.param_str}_{opts.region}_{opts.period}_{opts.dataset}'
            f'_{opts.start}to{opts.end}.nc'))
        bv_ds = xr.open_mfdataset(bv_files, data_vars='minimal')
        logger.info(f'Saving daily basis variables to {bv_outpath}')
        bv_ds.to_netcdf(bv_outpath)
        for file in bv_files:
            os.system(f'rm {file}')

    bv_outpath_new = bv_outpath.replace('.nc', '_new.nc')
    logger.info(f'Saving daily basis variables to {bv_outpath_new}')
    TEA.save_daily_results(bv_outpath_new)
    
    return TEA