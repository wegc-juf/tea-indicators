import glob
from itertools import repeat
from multiprocessing import Pool, get_context
import math
import logging
import numpy as np
import os
from pathlib import Path
import sys
import warnings
import xarray as xr

warnings.filterwarnings(action='ignore', message='All-NaN slice encountered')
warnings.filterwarnings(action='ignore', message='divide by zero encountered in divide')

sys.path.append('/home/hst/tea-indicators/scripts/misc/')
from general_functions import create_history
from calc_daily_basis_vars import calc_daily_basis_vars, calculate_event_count
from calc_TEA import (assign_ctp_coords, calc_event_frequency, calc_supplementary_event_vars,
                      calc_event_duration, calc_exceedance_magnitude, calc_exceedance_area_tex_sev)


def select_cell(opts, lat, lon, data, static, masks):
    """
    select data of current subcell and weight edges
    Args:
        opts: CLI parameter
        lat: current lat
        lon: current lon
        data: EUR data
        static: static ds
        masks: mask ds

    Returns:
        cell_data: data of cell
        land_frac: fraction covered by land below 1500m
        cell_static: static of cell

    """

    if opts.parameter == 'T':
        lat_off = 1
        lon_off = 1 / np.cos(np.deg2rad(lat))
    else:
        lat_off = 0.5
        lon_off = (1 / np.cos(np.deg2rad(lat))) * 0.5

    fac = 0.25
    if opts.dataset == 'ERA5Land':
        fac = 0.1

    lon_min = math.floor((lon - lon_off) * 4) / 4
    lon_max = math.ceil((lon + lon_off) * 4) / 4

    # select data for cell
    cell_data = data.sel(lat=slice(lat + lat_off, lat - lat_off),
                         lon=slice(lon_min, lon_max))
    cell_lsm = masks['lt1500_mask'].sel(lat=slice(lat + lat_off, lat - lat_off),
                                        lon=slice(lon_min, lon_max))
    cell_data = cell_data.where(cell_lsm == 1)

    # calc fraction by which most left column is smaller in area than orig grid
    orig_col = fac * (fac * len(cell_data.lat))
    col_width_w = fac - ((lon - lon_off) - cell_data.lon[0])
    small_col_w = col_width_w * (fac * len(cell_data.lat))
    frac_w = small_col_w / orig_col

    # calc fraction by which most left column is smaller in area than orig grid
    col_width_e = fac - (cell_data.lon[-1] - (lon + lon_off))
    small_col_e = col_width_e * (fac * len(cell_data.lat))
    frac_e = small_col_e / orig_col

    cell_lsm[:, 0].values = cell_lsm[:, 0].values * frac_w.values
    cell_lsm[:, -1].values = cell_lsm[:, -1].values * frac_e.values

    # calculate fraction covered by valid cells (land below 1500 m)
    land_frac = cell_lsm.sum() / np.size(cell_lsm)

    # select static data for cell
    cell_static = static.sel(lat=slice(lat + lat_off, lat - lat_off),
                             lon=slice(lon_min, lon_max))

    for vvar in ['area_grid', 'threshold']:
        cell_static[vvar][:, 0] = cell_static[vvar][:, 0] * frac_w.values
        cell_static[vvar][:, -1] = cell_static[vvar][:, -1] * frac_e.values

    cell_static['GR_size'] = cell_static['area_grid'].sum().values

    return cell_data, land_frac, cell_static


def dbv_to_new_grid(opts, dbv, cell, dbv_ds):
    """
    average variables to new grid
    Args:
        opts: CLI parameters
        dbv: daily basis variables
        cell: lat and lon of current cell
        dbv_ds: DBV ds for output

    Returns:

    """

    # GR vars of 0.5° sub-cells are used as new grid values
    # --> drop grid values on native 0.25° grid
    dvars = [vvar for vvar in dbv.data_vars if 'GR' not in vvar]
    dbv = dbv.drop(dvars)

    # rename vars (remove 'GR' from var name and attrs)
    rename_dict = {}
    for vvar in dbv.data_vars:
        new_name = vvar.split('_GR')[0]
        rename_dict[vvar] = new_name
        dbv[vvar].attrs['long_name'] = dbv[vvar].attrs['long_name'].split(' (GR)')[0]
    dbv = dbv.rename(rename_dict)

    # assign lat and lon coords to data (to combine back to grid later on)
    if dbv_ds is None:
        dbv_ds = dbv.copy()
        dbv_ds = dbv_ds.assign_coords({'lat': (['lat'], [cell[0]]), 'lon': (['lon'], [cell[1]])})
        for vvar in dbv_ds.data_vars:
            dbv_ds[vvar] = dbv_ds[vvar].expand_dims({'lat': 1, 'lon': 1})
    else:
        tmp_ds = dbv.copy()
        tmp_ds = tmp_ds.assign_coords({'lat': (['lat'], [cell[0]]), 'lon': (['lon'], [cell[1]])})
        dbv_ds = xr.concat([dbv_ds, tmp_ds], dim='lon')

    # drop time (no idea where this is coming from in the first place...)
    dbv_ds = dbv_ds.drop('time')

    dbv_ds = create_history(cli_params=sys.argv, ds=dbv_ds)

    # save tmp files
    # remove previous file
    os.system(f'rm {opts.tmppath}/daily_basis_variables/'
              f'DBV_lat{cell[0]}_lon{cell[1]}_{opts.param_str}_{opts.region}_{opts.period}'
              f'_{opts.dataset}_{opts.start}to{opts.end}.nc')

    path = Path(f'{opts.tmppath}daily_basis_variables/')
    path.mkdir(parents=True, exist_ok=True)
    dbv_ds.to_netcdf(f'{opts.tmppath}/daily_basis_variables/'
                     f'DBV_lat{cell[0]}_lon{cell[1]}_{opts.param_str}_{opts.region}_{opts.period}'
                     f'_{opts.dataset}_{opts.start}to{opts.end}.nc')

    return dbv_ds


def ctp_to_new_grid(opts, ef, ed, em, ea, svars, em_suppl, cell, ds, ds_suppl):
    """
    average variables to new grid
    Args:
        opts: CLI parameters
        ef: event frequency ds
        ed: event duration ds
        em: event magnitude ds
        ea: event area ds
        svars: supplementary variables dataset
        em_suppl: supplementary event magnitude ds
        cell: lat and lon of current cell
        ds: output ds
        ds_suppl: output supplementary ds

    Returns:

    """

    # combine to output dataset
    ds_out = xr.merge([ef, ed, em, ea])
    ds_out_suppl = xr.merge([svars, em_suppl])

    ds_out['ctp'] = ds_out['ctp'].assign_attrs(
        {'long_name': f'climatic time period ({opts.period})'})
    ds_out_suppl['ctp'] = ds_out_suppl['ctp'].assign_attrs(
        {'long_name': f'climatic time period ({opts.period})'})

    # GR vars of 0.5° sub-cells are used as new grid values
    # --> drop grid values on native 0.25° grid
    dvars = [vvar for vvar in ds_out.data_vars if 'GR' not in vvar]
    dvars_suppl = [vvar for vvar in ds_out_suppl.data_vars if 'GR' not in vvar]
    ds_out = ds_out.drop(dvars)
    ds_out_suppl = ds_out_suppl.drop(dvars_suppl)

    # rename vars (remove 'GR' from var name and attrs)
    rename_dict = {}
    for vvar in ds_out.data_vars:
        new_name = vvar.split('_GR')[0]
        rename_dict[vvar] = new_name
        ds_out[vvar].attrs['long_name'] = ds_out[vvar].attrs['long_name'].split(' (GR)')[0]
    ds_out = ds_out.rename(rename_dict)

    rename_dict_suppl = {}
    for vvar in ds_out_suppl.data_vars:
        new_name = vvar.split('_GR')[0]
        rename_dict_suppl[vvar] = new_name
        ds_out_suppl[vvar].attrs['long_name'] = ds_out_suppl[vvar].attrs['long_name'].split(
            ' (GR)')[0]
    ds_out_suppl = ds_out_suppl.rename(rename_dict_suppl)

    # assign lat and lon coords to data (to combine back to grid later on)
    if ds is None:
        ds = ds_out.copy()
        ds = ds.assign_coords({'lat': (['lat'], [cell[0]]), 'lon': (['lon'], [cell[1]])})
        for vvar in ds.data_vars:
            ds[vvar] = ds[vvar].expand_dims({'lat': 1, 'lon': 1})
    else:
        tmp_ds = ds_out.copy()
        tmp_ds = tmp_ds.assign_coords({'lat': (['lat'], [cell[0]]), 'lon': (['lon'], [cell[1]])})
        ds = xr.concat([ds, tmp_ds], dim='lon')

    if ds_suppl is None:
        ds_suppl = ds_out_suppl.copy()
        ds_suppl = ds_suppl.assign_coords(
            {'lat': (['lat'], [cell[0]]), 'lon': (['lon'], [cell[1]])})
        for vvar in ds_suppl.data_vars:
            ds_suppl[vvar] = ds_suppl[vvar].expand_dims({'lat': 1, 'lon': 1})
    else:
        tmp_ds_suppl = ds_out_suppl.copy()
        tmp_ds_suppl = tmp_ds_suppl.assign_coords({'lat': (['lat'], [cell[0]]), 'lon': (['lon'],
                                                                                        [cell[1]])})
        ds_suppl = xr.concat([ds_suppl, tmp_ds_suppl], dim='lon')

    # drop time (no idea where this is coming from in the first place...)
    ds = ds.drop('time')
    ds_suppl = ds_suppl.drop('time')

    ds = create_history(cli_params=sys.argv, ds=ds)
    ds_suppl = create_history(cli_params=sys.argv, ds=ds_suppl)

    # save tmp files
    path = Path(f'{opts.tmppath}ctp_indicator_variables/supplementary/')
    path.mkdir(parents=True, exist_ok=True)
    ds.to_netcdf(f'{opts.tmppath}ctp_indicator_variables/'
                 f'CTP_lat{cell[0]}_lon{cell[1]}_{opts.param_str}_{opts.region}_{opts.period}'
                 f'_{opts.dataset}_{opts.start}to{opts.end}.nc')
    ds_suppl.to_netcdf(f'{opts.tmppath}ctp_indicator_variables/supplementary/'
                       f'CTPsuppl_lat{cell[0]}_lon{cell[1]}_{opts.param_str}_{opts.region}'
                       f'_{opts.period}_{opts.dataset}_{opts.start}to{opts.end}.nc')

    return ds, ds_suppl


def calc_tea_lat(opts, data, static, masks, lat):
    logging.info(f'Processing lat {lat}')

    if opts.dataset == 'ERA5':
        lons = np.arange(-12, 40.5, 0.5)
    else:
        lons = np.arange(9, 18, 0.5)

    ds = None
    ds_suppl = None
    dbv_ds = None

    # step through all longitudes
    for ilon, lon in enumerate(lons):
        cell_data, land_frac, cell_static = select_cell(opts=opts, lat=lat, lon=lon, data=data,
                                                        static=static, masks=masks)

        # skip cells with less than 50% land in it
        if land_frac.values <= 0.5 or land_frac.isnull():
            continue

        calc_daily_basis_vars(opts=opts, static=cell_static, data=cell_data, large_gr=True,
                              cell=[lat, lon])

        dbv = xr.open_dataset(
            f'{opts.tmppath}daily_basis_variables/'
            f'DBV_lat{lat}_lon{lon}_{opts.param_str}_{opts.region}_{opts.period}_{opts.dataset}'
            f'_{opts.start}to{opts.end}.nc')

        # apply criterion that DTEA_GR > DTEA_min and all GR variables use same dates,
        # dtea_min is given in areals (1 areal = 100 km2)
        dtea_min = 1
        for vvar in dbv.data_vars:
            if vvar == 'DTEEC_GR':
                # Amin criterion sometimes splits up events --> run DTEEC_GR detection again
                dbv[vvar] = calculate_event_count(opts=opts, dtec=dbv['DTEC_GR'], da_out=True,
                                                  cstr=f'_lat{lat}_lon{lon}')
            elif 'GR' in vvar:
                dbv[vvar] = dbv[vvar].where(dbv['DTEA_GR'] > dtea_min)

        # get dates for climatic time periods (CTP) and assign coords to dbv
        dbv, dbv_per = assign_ctp_coords(opts, data=dbv)

        # calculate EF and corresponding supplementary variables
        ef = calc_event_frequency(pdata=dbv_per)
        svars = calc_supplementary_event_vars(data=dbv)

        # calculate ED
        ed = calc_event_duration(pdata=dbv_per, ef=ef)

        # calculate EM
        em, em_suppl = calc_exceedance_magnitude(opts=opts, pdata=dbv_per, ed=ed)

        # calculate EA
        ea = calc_exceedance_area_tex_sev(opts=opts, data=dbv, ed=ed, em=em)

        # calc dbv on new grid and save tmp files
        dbv_ds = dbv_to_new_grid(opts=opts, dbv=dbv, cell=[lat, lon], dbv_ds=dbv_ds)

        # calc variables on new grid and save tmp files
        ds, ds_suppl = ctp_to_new_grid(opts=opts, ef=ef, ed=ed, em=em, ea=ea, svars=svars,
                                       em_suppl=em_suppl, cell=[lat, lon],
                                       ds=ds, ds_suppl=ds_suppl)

    # save output files
    if dbv_ds is not None:
        dbv_ds = dbv_ds.drop(['ctp', 'doy'])
        dbv_ds.to_netcdf(f'{opts.tmppath}daily_basis_variables/'
                         f'DBV_lat{lat}_{opts.param_str}_{opts.region}_{opts.period}'
                         f'_{opts.dataset}_{opts.start}to{opts.end}.nc')
        ds.to_netcdf(f'{opts.tmppath}ctp_indicator_variables/'
                     f'CTP_lat{lat}_{opts.param_str}_{opts.region}_{opts.period}'
                     f'_{opts.dataset}_{opts.start}to{opts.end}.nc')
        ds_suppl.to_netcdf(f'{opts.tmppath}ctp_indicator_variables/supplementary/'
                           f'CTPsuppl_lat{lat}_{opts.param_str}_{opts.region}'
                           f'_{opts.period}_{opts.dataset}_{opts.start}to{opts.end}.nc')

        # remove individual cell files
        cell_files_ctp = sorted(glob.glob(f'{opts.tmppath}ctp_indicator_variables/'
                                          f'CTP_lat{lat}_lon*_{opts.param_str}_{opts.region}'
                                          f'_{opts.period}_{opts.dataset}'
                                          f'_{opts.start}to{opts.end}.nc'))
        cell_files_suppl = sorted(glob.glob(f'{opts.tmppath}ctp_indicator_variables/supplementary/'
                                            f'CTPsuppl_lat{lat}_lon*_{opts.param_str}_{opts.region}'
                                            f'_{opts.period}_{opts.dataset}'
                                            f'_{opts.start}to{opts.end}.nc'))
        cell_files_dbv = sorted(glob.glob(f'{opts.tmppath}daily_basis_variables/'
                                          f'DBV_lat{lat}_lon*_{opts.param_str}_{opts.region}'
                                          f'_{opts.period}_{opts.dataset}'
                                          f'_{opts.start}to{opts.end}.nc'))

        for ifile, file in enumerate(cell_files_ctp):
            os.system(f'rm {file}')
            os.system(f'rm {cell_files_suppl[ifile]}')
            os.system(f'rm {cell_files_dbv[ifile]}')


def filter_filenames(filenames, lat_range):
    filtered_filenames = []
    for file in filenames:
        file_lat = float(file.split('lat')[1][:4])
        if lat_range[0] <= file_lat <= lat_range[1]:
            filtered_filenames.append(file)

    return filtered_filenames


def area_grid(opts, da):
    if opts.dataset == 'ERA5':
        delta_fac = 4  # to get 0.25° resolution
    else:
        delta_fac = 10  # to get 0.1°

    lat = da.lat.values
    r_mean = 6371
    u_mean = 2 * np.pi * r_mean

    # calculate earth radius at different latitudes
    r_lat = np.cos(np.deg2rad(lat)) * r_mean

    # calculate earth circumference at latitude
    u_lat = 2 * np.pi * r_lat

    # calculate length of 0.25°/0.1° in m for x and y dimension
    x_len = (u_lat / 360) / delta_fac
    y_len = (u_mean / 360) / delta_fac

    # calculate size of cells in areals
    x_len_da = xr.DataArray(data=x_len, coords={'lat': (['lat'], lat)})
    agrid = xr.DataArray(data=np.ones((len(da.lat), len(da.lon))),
                         coords={'lat': (['lat'], da.lat.values), 'lon': (['lon'], da.lon.values)})
    agrid = (agrid * y_len * x_len_da) / 100

    return agrid


def combine_to_eur(opts, lat_lims, mask):
    """
    combines files of latitudes to one EUR file
    Args:
        opts: CLI parameter
        lat_lims: min and max latitude of the region
        mask: GR mask on output grid

    Returns:

    """
    # Filter file names
    bv_files = sorted(glob.glob(f'{opts.tmppath}daily_basis_variables/'
                                f'DBV_lat*_{opts.param_str}_{opts.region}_{opts.period}'
                                f'_{opts.dataset}_{opts.start}to{opts.end}.nc'))
    bv_files = filter_filenames(bv_files, lat_range=lat_lims)

    ctp_files = sorted(glob.glob(f'{opts.tmppath}ctp_indicator_variables/'
                                 f'CTP_lat*_{opts.param_str}_{opts.region}_{opts.period}'
                                 f'_{opts.dataset}_{opts.start}to{opts.end}.nc'))
    ctp_files = filter_filenames(ctp_files, lat_range=lat_lims)

    ctp_suppl_files = sorted(glob.glob(f'{opts.tmppath}ctp_indicator_variables/supplementary/'
                                       f'CTPsuppl_lat*_{opts.param_str}_{opts.region}_{opts.period}'
                                       f'_{opts.dataset}_{opts.start}to{opts.end}.nc'))
    ctp_suppl_files = filter_filenames(ctp_suppl_files, lat_range=lat_lims)

    files = {'DBV': bv_files, 'CTP': ctp_files, 'CTPsuppl': ctp_suppl_files}

    outpaths = {'DBV': f'{opts.outpath}daily_basis_variables/'
                       f'DBV_{opts.param_str}_{opts.region}_{opts.period}_{opts.dataset}'
                       f'_{opts.start}to{opts.end}.nc',
                'CTP': f'{opts.outpath}ctp_indicator_variables/'
                       f'CTP_{opts.param_str}_{opts.region}_{opts.period}_{opts.dataset}'
                       f'_{opts.start}to{opts.end}.nc',
                'CTPsuppl': f'{opts.outpath}ctp_indicator_variables/supplementary/'
                            f'CTPsuppl_{opts.param_str}_{opts.region}_{opts.period}_{opts.dataset}'
                            f'_{opts.start}to{opts.end}.nc'}

    # list of variables for which a GR version should be calculated
    gr_vars = ['DTEC', 'DTEEC', 'DTEM', 'DTEM', 'DTEA', 'EF', 'ED', 'EDavg', 'EM', 'EMavg',
               'EM_Md', 'EMavg_Md', 'EM_Max', 'EMavg_Max', 'ESavg', 'TEX', 'delta_y',
               'doy_first', 'doy_last']

    # load files
    for vvars in files.keys():
        ds = xr.open_mfdataset(files[vvars], concat_dim='lat', combine='nested')
        ds.attrs['info'] = (f'Grid values correspond to GR values of subcells. '
                            f'GR values for whole region {opts.region} can be calculated after '
                            f'decadal-mean indicator calculation with aggregate_to_AGR.py.')
        ds.to_netcdf(outpaths[vvars])
        ds.close()

        # remove tmp_files
        for tfile in files[vvars]:
            os.system(f'rm {tfile}')


def check_tmp_dirs(opts):
    """
    check if tmp directories are empty and ask if files should be deleted
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

    tmp_dirs = [f'{opts.outpath}daily_basis_variables/tmp/',
                f'{opts.tmppath}daily_basis_variables/',
                f'{opts.tmppath}ctp_indicator_variables/supplementary/']

    # Check each directory and interact with the user if necessary
    non_empty = 0
    for ddir in tmp_dirs:
        if not is_directory_empty(ddir):
            non_empty += 1
            break

    if non_empty > 0:
        logging.info(f'At least one tmp directory is not empty. tmp files will be deleted first.')
        for ddir in tmp_dirs:
            delete_files_in_directory(ddir)
        # ctp dir is always non-empty because of the sub-dir supplementary --> not checked before
        # delete files now
        delete_files_in_directory(directory=f'{opts.tmppath}ctp_indicator_variables/')


def calc_tea_large_gr(opts, data, masks, static):
    logging.info(f'Switching to calc_TEA_largeGR because GR > 100 areals.')

    # check if tmp directories are empty
    check_tmp_dirs(opts)

    # preselect region to reduce computation time
    min_lat = data.lat[np.where(masks['lt1500_mask'] > 0)[0][-1]].values
    max_lat = data.lat[np.where(masks['lt1500_mask'] > 0)[0][0]].values
    if min_lat < 35:
        min_lat = 35

    # define latitudes with 0.5° resolution for output
    lats = np.arange(math.floor(min_lat), math.ceil(max_lat) + 0.5, 0.5)

    with get_context('spawn').Pool(processes=5) as pool:
        pool.starmap(calc_tea_lat, zip(repeat(opts), repeat(data), repeat(static), repeat(masks),
                                       lats))
    pool.close()
    pool.join()

    # for testing with only one latitude or debugging
    # calc_tea_lat(opts=opts, data=data, static=static, masks=masks, lat=lats[3])
    # for llat in lats:
    #     calc_tea_lat(opts=opts, data=data, static=static, masks=masks, lat=llat)

    # create region mask on new grid
    if opts.dataset == 'ERA5':
        lons = np.arange(-12, 40.5, 0.5)
    else:
        lons = np.arange(9, 18, 0.5)
    ngrid_mask = masks['lt1500_mask'].sel(lat=slice(max_lat, min_lat))
    ngrid_mask = ngrid_mask.interp(lat=lats, lon=lons)

    logging.info(f'Combining individual latitudes to single file.')
    combine_to_eur(opts=opts, lat_lims=[min_lat, max_lat], mask=ngrid_mask)
