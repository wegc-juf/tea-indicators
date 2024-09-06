import glob
import logging
import numpy as np
import os
from pathlib import Path
import xarray as xr


def calc_dtec_dtea(opts, dtem, static, cstr):
    """
    calculate DTEC, DTEC_GR, and DTEA_GR and save it to tmp files
    Args:
        opts: CLI parameter
        dtem: Daily Threshold Exceedance Magnitude
        static: static files
        cstr: cell string to add to filename (only if called from calc_TEA_largeGR)

    Returns:
        dtec: Daily Threshold Exceedance Count
        dtec_gr: Daily Threshold Exceedance Count (GR)
        dtea_gr: Daily Threshold Exceedance AREA (GR)
    """
    outpath = f'{opts.outpath}daily_basis_variables/tmp/'

    # equation 01
    # store DTEM for all DTEC == 1
    dtec = dtem.where(dtem.isnull(), 1)
    dtec = dtec.rename('DTEC')
    dtec.attrs = {'long_name': 'daily threshold exceedance count', 'units': '1'}

    # equation 02_1 not needed (cells with TEC == 0 are already nan in tem)
    # equation 02_2
    dtea = dtec * static['area_grid']

    # equation 06
    # calculate DTEA_GR
    dtea_gr = dtea.sum(axis=(1, 2), skipna=True)
    dtea_gr = dtea_gr.rename('DTEA_GR')
    dtea_gr = dtea_gr.assign_attrs({'long_name': 'daily threshold exceedance area',
                                    'units': 'areals'})

    # calc area fraction
    area_frac = (dtea_gr / static['GR_size']) * 100
    area_frac = area_frac.rename('DTEA_frac')

    areas = xr.merge([dtea_gr, area_frac])
    areas.to_netcdf(f'{outpath}DTEA{cstr}_{opts.param_str}_{opts.region}_{opts.period}_{opts.dataset}'
                    f'_{opts.start}to{opts.end}.nc')

    # calculate dtec_gr (continues equation 03)
    dtec_gr = dtec.notnull().any(dim=static['threshold'].dims)
    dtec_gr = dtec_gr.where(dtec_gr == True)
    dtec_gr = dtec_gr.rename(f'{dtec.name}_GR')
    dtec_gr = dtec_gr.assign_attrs({'long_name': 'daily threshold exceedance count (GR)',
                                    'units': '1'})

    dtecs = xr.merge([dtec, dtec_gr])
    dtecs.to_netcdf(f'{outpath}DTEC{cstr}_{opts.param_str}_{opts.region}_{opts.period}_{opts.dataset}'
                    f'_{opts.start}to{opts.end}.nc')

    return dtec, dtec_gr, dtea_gr


def calc_dteec_1d(dtec_cell):
    # Convert to a NumPy array and change NaN to 0
    dtec_np = np.nan_to_num(dtec_cell, nan=0)

    # Find the starts and ends of sequences (change NaNs to 0 before the diff operation)
    change = np.diff(np.concatenate(
        ([np.zeros((1,) + dtec_np.shape[1:]), dtec_np, np.zeros((1,) + dtec_np.shape[1:])]),
        axis=0), axis=0)
    starts = np.where(change == 1)
    ends = np.where(change == -1)

    # Calculate the middle points (as flat indices)
    middle_indices = (starts[0] + ends[0] - 1) // 2

    # Create an output array filled with NaNs
    events_np = np.full(dtec_cell.shape, np.nan)

    # Set the middle points to 1 (use flat indices to index into the 3D array)
    events_np[middle_indices] = 1

    return events_np


def calculate_event_count(opts, dtec, cstr, da_out=False):
    """
    calculate DTEEC(_GR) according to equations 4 and 5
    Args:
        opts: CLI parameter
        dtec: daily threshold exceedance count
        cstr: string for subcell
        da_out: set to true if da is written to output

    Returns:

    """

    if 'GR' in dtec.name:
        dteec_np = calc_dteec_1d(dtec_cell=dtec.values)
        dteec = xr.DataArray(dteec_np, coords=dtec.coords, dims=dtec.dims)
        gr_str, gr_var_str = ' (GR)', '_GR'
    else:
        dteec = xr.full_like(dtec, np.nan)
        dtec_3d = dtec.values
        # loop through all rows and calculate DTEEC
        for iy in range(len(dtec_3d[0, :, 0])):
            dtec_row = dtec_3d[:, iy, :]
            # skip all nan rows
            if np.isnan(dtec_row).all():
                continue
            dteec_row = np.apply_along_axis(calc_dteec_1d, axis=0, arr=dtec_row)
            dteec[:, iy, :] = dteec_row
        gr_str, gr_var_str = '', ''

    dteec = dteec.rename(f'DTEEC{gr_var_str}')
    dteec.attrs = {'long_name': f'daily threshold exceedance event count{gr_str}', 'units': '1'}

    if not da_out:
        outname = (f'{opts.outpath}daily_basis_variables/tmp/'
                   f'{dteec.name}{cstr}_{opts.param_str}_{opts.region}_{opts.period}_{opts.dataset}'
                   f'_{opts.start}to{opts.end}.nc')
        dteec.to_netcdf(outname)
    else:
        return dteec


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

    if opts.parameter == 'T':
        data_unit = '°C'
    else:
        data_unit = 'mm'

    cell_str = ''
    if large_gr:
        cell_str = f'_lat{cell[0]}_lon{cell[1]}'

    path = Path(f'{opts.outpath}daily_basis_variables/tmp/')
    path.mkdir(parents=True, exist_ok=True)

    # check if tmp directory is empty
    if not large_gr:
        check_tmp_dir(opts)

    # calculate DTEM
    # equation 07
    dtem = data - static['threshold']
    dtem = dtem.where(dtem > 0).astype('float32')
    dtem = dtem.rename('DTEM')
    dtem.attrs = {'long_name': 'daily threshold exceedance magnitude', 'units': data_unit}

    dtec, dtec_gr, dtea_gr = calc_dtec_dtea(opts=opts, dtem=dtem, static=static, cstr=cell_str)

    # equation 08
    # calculate dtem_gr (area weighted DTEM)
    area_fac = static['area_grid'] / dtea_gr.T
    dtem_gr = (dtem * area_fac).sum(axis=(1, 2), skipna=True)
    dtem_gr = dtem_gr.rename(f'{dtem.name}_GR')
    dtem_gr = dtem_gr.assign_attrs({'long_name': 'daily threshold exceedance magnitude (GR)',
                                    'units': data_unit})

    # equation 09
    # save maximum DTEM
    dtem_max = dtem.max(dim=static['threshold'].dims)
    dtem_max = dtem_max.rename('DTEM_Max')
    dtem_max = dtem_max.assign_attrs({'long_name': 'daily maximum grid cell exceedance magnitude',
                                      'units': data_unit})

    dtems = xr.merge([dtem, dtem_gr, dtem_max])
    outname = (f'{opts.outpath}daily_basis_variables/tmp/'
               f'DTEM{cell_str}_{opts.param_str}_{opts.region}_{opts.period}_{opts.dataset}'
               f'_{opts.start}to{opts.end}.nc')
    dtems.to_netcdf(outname)

    # equations 4 and 5
    # calculate DTEEC(_GR)
    calculate_event_count(opts=opts, dtec=dtec, cstr=cell_str)
    calculate_event_count(opts=opts, dtec=dtec_gr, cstr=cell_str)

    # combine all basic variables into one ds
    bv_files = sorted(glob.glob(
        f'{opts.outpath}daily_basis_variables/tmp/'
        f'*{cell_str}_{opts.param_str}_{opts.region}_{opts.period}_{opts.dataset}'
        f'_{opts.start}to{opts.end}.nc'))
    bv_ds = xr.open_mfdataset(bv_files, data_vars='minimal')

    bv_outpath = (f'{opts.outpath}daily_basis_variables/'
                  f'DBV_{opts.param_str}_{opts.region}_{opts.period}_{opts.dataset}'
                  f'_{opts.start}to{opts.end}.nc')
    if large_gr:
        large_gr_path = Path(f'{opts.tmppath}daily_basis_variables/')
        large_gr_path.mkdir(parents=True, exist_ok=True)
        bv_outpath = (f'{opts.tmppath}daily_basis_variables/'
                      f'DBV{cell_str}_{opts.param_str}_{opts.region}_{opts.period}_{opts.dataset}'
                      f'_{opts.start}to{opts.end}.nc')

    bv_ds.to_netcdf(bv_outpath)

    for file in bv_files:
        os.system(f'rm {file}')