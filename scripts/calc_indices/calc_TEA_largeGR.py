import glob
import math
import numpy as np
import os
from pathlib import Path
import sys
import warnings
import xarray as xr

warnings.filterwarnings(action='ignore', message='All-NaN slice encountered')
warnings.filterwarnings(action='ignore', message='divide by zero encountered in divide')

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from scripts.general_stuff.general_functions import create_history
from scripts.general_stuff.var_attrs import get_attrs
from scripts.general_stuff.TEA_logger import logger
from scripts.calc_indices.calc_daily_basis_vars import calc_daily_basis_vars, calculate_event_count
from scripts.calc_indices.calc_TEA import (assign_ctp_coords, calc_event_frequency,
                                           calc_supplementary_event_vars,
                                           calc_event_duration, calc_exceedance_magnitude,
                                           calc_exceedance_area_tex_sev)


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
    small_col_w = abs(col_width_w) * (fac * len(cell_data.lat))
    frac_w = small_col_w / orig_col

    # calc fraction by which most left column is smaller in area than orig grid
    col_width_e = fac - ((lon + lon_off) - cell_data.lon[-1])
    small_col_e = abs(col_width_e) * (fac * len(cell_data.lat))
    frac_e = small_col_e / orig_col

    frac_da = xr.DataArray(data=np.ones((len(cell_lsm.lat), len(cell_lsm.lon))),
                           coords={'lat': (['lat'], cell_lsm.lat.values),
                                   'lon': (['lon'], cell_lsm.lon.values)},
                           dims={'lat': (['lon'], cell_lsm.lat.values),
                                 'lon': (['lon'], cell_lsm.lon.values)})

    frac_da[:, 0] = frac_da[:, 0] * frac_w.values
    frac_da[:, -1] = frac_da[:, -1] * frac_e.values

    # apply weights to LSM
    cell_lsm = cell_lsm * frac_da

    # calculate fraction covered by valid cells (land below 1500 m)
    land_frac = cell_lsm.sum() / np.size(cell_lsm)

    # select static data for cell
    cell_static = static.sel(lat=slice(lat + lat_off, lat - lat_off),
                             lon=slice(lon_min, lon_max))

    for vvar in ['area_grid', 'threshold']:
        cell_static[vvar] = cell_static[vvar] * frac_da

    cell_static['GR_size'] = cell_static['area_grid'].sum().values

    # apply weights to cell_data
    cell_data = cell_data * frac_da

    return cell_data, land_frac, cell_static


def dbv_to_new_grid_new(dbv):
    """
    average variables to new grid
    Args:
        dbv: daily basis variables

    Returns:

    """

    # GR vars of 0.5° sub-cells are used as new grid values
    # --> drop grid values on native 0.25° grid
    dvars = [vvar for vvar in dbv.data_vars if 'GR' not in vvar]
    dbv = dbv.drop_vars(dvars)

    # rename vars (remove 'GR' from var name and attrs)
    rename_dict = {}
    for vvar in dbv.data_vars:
        new_name = vvar.split('_GR')[0]
        rename_dict[vvar] = new_name
        dbv[vvar].attrs['long_name'] = dbv[vvar].attrs['long_name'].split(' (GR)')[0]
    dbv = dbv.rename(rename_dict)

    return dbv


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
    dbv = dbv.drop_vars(dvars)

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
    dbv_ds = dbv_ds.drop_vars('time')

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

    # set all vars to 0 if EF is 0
    for vvar in ds_out.data_vars:
        if 'GR' in vvar:
            ds_out[vvar] = ds_out[vvar].where(ef.EF_GR != 0, 0)
        else:
            ds_out[vvar] = ds_out[vvar].where(ef.EF != 0, 0)
    for vvar in ds_out_suppl.data_vars:
        if 'GR' in vvar:
            ds_out_suppl[vvar] = ds_out_suppl[vvar].where(ef.EF_GR != 0, 0)
        else:
            ds_out_suppl[vvar] = ds_out_suppl[vvar].where(ef.EF != 0, 0)

    ds_out['ctp'] = ds_out['ctp'].assign_attrs(get_attrs(opts=opts, vname='ctp'))
    ds_out_suppl['ctp'] = ds_out_suppl['ctp'].assign_attrs(get_attrs(opts=opts, vname='ctp'))

    # GR vars of 0.5° sub-cells are used as new grid values
    # --> drop grid values on native 0.25° grid
    dvars = [vvar for vvar in ds_out.data_vars if 'GR' not in vvar]
    dvars_suppl = [vvar for vvar in ds_out_suppl.data_vars if 'GR' not in vvar]
    ds_out = ds_out.drop_vars(dvars)
    ds_out_suppl = ds_out_suppl.drop_vars(dvars_suppl)

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
    ds = ds.drop_vars('time')
    ds_suppl = ds_suppl.drop_vars('time')

    ds = create_history(cli_params=sys.argv, ds=ds)
    ds_suppl = create_history(cli_params=sys.argv, ds=ds_suppl)

    return ds, ds_suppl


def calc_tea_lat(opts, data, static, masks, lat):
    logger.info(f'Processing lat {lat}')

    if opts.dataset == 'ERA5':
        lons = np.arange(-12, 40.5, 0.5)
    else:
        lons = np.arange(9, 18, 0.5)

    ds = None
    ds_suppl = None
    dbv_ds = None

    # step through all longitudes
    for ilon, lon in enumerate(lons):

        # this comment is necessary to suppress an unnecessary PyCharm warning for lon
        # noinspection PyTypeChecker
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
        dbv_ds = dbv_ds.drop_vars(['ctp', 'doy'])
        dbv_ds.to_netcdf(f'{opts.tmppath}daily_basis_variables/'
                         f'DBV_lat{lat}_{opts.param_str}_{opts.region}_{opts.period}'
                         f'_{opts.dataset}_{opts.start}to{opts.end}.nc')
        ds.to_netcdf(f'{opts.tmppath}ctp_indicator_variables/'
                     f'CTP_lat{lat}_{opts.param_str}_{opts.region}_{opts.period}'
                     f'_{opts.dataset}_{opts.start}to{opts.end}.nc')
        ds_suppl.to_netcdf(f'{opts.tmppath}ctp_indicator_variables/supplementary/'
                           f'CTPsuppl_lat{lat}_{opts.param_str}_{opts.region}'
                           f'_{opts.period}_{opts.dataset}_{opts.start}to{opts.end}.nc')

        # remove individual dbv cell files
        cell_files_dbv = sorted(glob.glob(f'{opts.tmppath}daily_basis_variables/'
                                          f'DBV_lat{lat}_lon*_{opts.param_str}_{opts.region}'
                                          f'_{opts.period}_{opts.dataset}'
                                          f'_{opts.start}to{opts.end}.nc'))

        for ifile, file in enumerate(cell_files_dbv):
            os.system(f'rm {cell_files_dbv[ifile]}')

        ds.close()
        ds_suppl.close()
        dbv_ds.close()


def filter_filenames(filenames, lat_range):
    filtered_filenames = []
    for file in filenames:
        file_lat = float(file.split('lat')[1][:4])
        if lat_range[0] <= file_lat <= lat_range[1]:
            filtered_filenames.append(file)

    return filtered_filenames


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

    # load files and combine them to GR on 0.5 grid
    for vvars in files.keys():
        ds = xr.open_mfdataset(files[vvars], concat_dim='lat', combine='nested')
        ds.attrs['info'] = (f'Grid values correspond to GR values of subcells. '
                            f'GR values for whole region {opts.region} can be calculated after '
                            f'decadal-mean indicator calculation with calc_AGR_vars.py.')
        # apply EUR mask on 0.5° grid
        ds = ds.where(mask > 0)

        # clear history of combined dataset and create new one (otherwise, history is added 10 times)
        ds.attrs['history'] = ''
        ds = create_history(cli_params=sys.argv, ds=ds)

        # save ds to netcdf
        ds.to_netcdf(outpaths[vvars])
        ds.close()

        # remove tmp_files
        for tfile in files[vvars]:
            os.system(f'rm {tfile}')

    # save 0.5° mask
    mask = create_history(cli_params=sys.argv, ds=mask)
    mask.to_netcdf(f'{opts.maskpath}{opts.region}_mask_0p5_{opts.dataset}.nc')


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
        logger.info(f'At least one tmp directory is not empty. tmp files will be deleted first.')
        for ddir in tmp_dirs:
            delete_files_in_directory(ddir)
        # ctp dir is always non-empty because of the sub-dir supplementary --> not checked before
        # delete files now
        delete_files_in_directory(directory=f'{opts.tmppath}ctp_indicator_variables/')


def create_0p5_mask(opts, mask_0p25, area_0p25, lats):
    """
    create mask of valid grid points (land < 1500 m) on output grid (Methods p.34, 3rd paragraph)
    Args:
        opts: CLI parameter
        mask_0p25: mask on 0.25° grid
        area_0p25: area on 0.25° grid
        lats: latitude coords of output grid

    Returns:
        mask_0p5: mask on 0.5° output grid

    """
    # flip lats to be consistent with other datasets
    lats = lats[::-1]

    if opts.dataset == 'ERA5':
        lons = np.arange(-12, 40.5, 0.5)
    else:
        lons = np.arange(9, 18, 0.5)

    mask_0p5 = xr.DataArray(data=np.ones((len(lats), len(lons))) * np.nan,
                            coords={'lat': (['lat'], lats), 'lon': (['lon'], lons)},
                            dims={'lat': (['lat'], lats), 'lon': (['lon'], lons)})
    mask_0p5 = mask_0p5.rename('mask_lt1500')

    area_0p5 = xr.DataArray(data=np.ones((len(lats), len(lons))) * np.nan,
                            coords={'lat': (['lat'], lats), 'lon': (['lon'], lons)},
                            dims={'lat': (['lat'], lats), 'lon': (['lon'], lons)})
    area_0p5 = area_0p5.rename('area_grid')

    for llat in mask_0p5.lat:
        for llon in mask_0p5.lon:
            cell_0p25 = mask_0p25.sel(lat=slice(llat + 0.25, llat - 0.25),
                                      lon=slice(llon - 0.25, llon + 0.25))
            cell_area = area_0p25.sel(lat=slice(llat + 0.25, llat - 0.25),
                                      lon=slice(llon - 0.25, llon + 0.25))
            valid_cells = cell_0p25.sum()
            if valid_cells == 0:
                continue
            vcell_frac = valid_cells / cell_0p25.size
            mask_0p5.loc[llat, llon] = vcell_frac.values
            area_0p5.loc[llat, llon] = cell_area.sum().values

    area_0p5 = create_history(cli_params=sys.argv, ds=area_0p5)
    area_0p5.to_netcdf(f'{opts.statpath}area_grid_0p5_{opts.region}_{opts.dataset}.nc')

    return mask_0p5


def calc_tea_large_gr(opts, data, masks, static):
    logger.info(f'Switching to calc_TEA_largeGR because GR > 100 areals.')

    # check if tmp directories are empty
    check_tmp_dirs(opts)

    # preselect region to reduce computation time
    min_lat = data.lat[np.where(masks['lt1500_mask'] > 0)[0][-1]].values
    max_lat = data.lat[np.where(masks['lt1500_mask'] > 0)[0][0]].values
    if min_lat < 35:
        min_lat = 35

    # define latitudes with 0.5° resolution for output
    lats = np.arange(math.floor(min_lat), math.ceil(max_lat) + 0.5, 0.5)

    # for testing with only one latitude or debugging
    # calc_tea_lat(opts=opts, data=data, static=static, masks=masks, lat=lats[3])
    for llat in lats:
        calc_tea_lat(opts=opts, data=data, static=static, masks=masks, lat=llat)

    # create region mask on new grid
    ngrid_mask = create_0p5_mask(opts=opts,
                                 mask_0p25=masks['lt1500_mask'].sel(lat=slice(max_lat, min_lat)),
                                 area_0p25=static['area_grid'].sel(lat=slice(max_lat, min_lat)),
                                 lats=lats)

    # logger.info(f'Combining individual latitudes to single file.')
    combine_to_eur(opts=opts, lat_lims=[min_lat, max_lat], mask=ngrid_mask)
