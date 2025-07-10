"""
scripts for general stuff (e.g. nc-history)
"""

import argparse
import datetime as dt
import yaml
import numpy as np
import glob
import os
import xarray as xr
import pandas as pd

from .check_CFG import check_config
from .TEA_logger import logger
from .cfg_paramter import show_parameters


def load_opts(fname, config_file='./config/TEA_CFG.yaml'):
    """
    load parameters from CFG file and put them into a Namespace object
    Args:
        fname: name of executed script
        config_file: path to CFG file

    Returns:
        opts: CFG parameter

    """

    fname = fname.split('/')[-1].split('.py')[0]
    with open(config_file, 'r') as stream:
        opts = yaml.safe_load(stream)
        opts = opts[fname]
        opts = argparse.Namespace(**opts)

    # add name of script
    opts.script = f'{fname}.py'
    
    opts = _get_default_opts(fname, opts)
    check_config(opts_dict=vars(opts))
    
    if opts.gui:
        # show set parameters
        show_parameters(opts)
        check_config(opts_dict=vars(opts))
        opts = argparse.Namespace(**opts)
        
    # add strings that are often needed to parameters
    if fname not in ['create_region_masks']:
        pstr = opts.parameter
        if opts.parameter != 'Tx':
            pstr = f'{opts.parameter}_'

        param_str = f'{pstr}{opts.threshold:.1f}p'
        if opts.threshold_type == 'abs':
            param_str = f'{pstr}{opts.threshold:.1f}{opts.unit}'

        opts.param_str = param_str

    # convert str to int
    if 'ref_period' in opts:
        ref_period = opts.ref_period.split('-')
        opts.ref_period = (int(ref_period[0]), int(ref_period[1]))
        cc_period = opts.cc_period.split('-')
        opts.cc_period = (int(cc_period[0]), int(cc_period[1]))

    return opts


def _get_default_opts(fname, opts):
    """
    set default options for parameters if not set in CFG file
    Args:
        fname: script name to check options for
        opts: options dictionary

    Returns:
        opts: options with default values set

    """
    if 'precip' not in opts:
        opts.precip = False
    
    # GeoRegion options
    if 'region' not in opts:
        opts.region = 'AUT'
    if 'agr' in opts:
        if 'agr_cell_size' not in opts:
            if opts.precip:
                opts.agr_cell_size = 1
            else:
                opts.agr_cell_size = 2
        if 'grg_grid_spacing' not in opts:
            opts.grg_grid_spacing = 0.5
        if 'land_frac_min' not in opts:
            opts.land_frac_min = 0.5
    
    # Parameter options
    if 'parameter' not in opts:
        opts.parameter = 'Tx'
    if 'threshold_type' not in opts:
        opts.threshold_type = 'perc'
    if 'threshold' not in opts:
        if opts.precip:
            opts.threshold = 95
        else:
            opts.threshold = 99
    if 'smoothing_radius' not in opts:
        if 'smoothing' in opts:
            opts.smoothing_radius = opts.smoothing
        else:
            opts.smoothing_radius = 0
    if 'unit' not in opts:
        if opts.precip:
            opts.unit = 'mm'
        else:
            opts.unit = 'degC'
    if 'low_extreme' not in opts:
        opts.low_extreme = False
        
    # time_params options
    if 'start' not in opts:
        opts.start = 1961
    if 'end' not in opts:
        opts.end = 2024
    if 'period' not in opts:
        opts.period = 'annual'
    if 'ref_period' not in opts:
        opts.ref_period = '1961-1990'
    if 'cc_period' not in opts:
        opts.cc_period = '2010-2024'
    
    # paths
    if 'statpath' in opts and 'maskpath' not in opts:
        opts.maskpath = opts.statpath
    if 'mask_sub' not in opts:
        opts.mask_sub = '/masks'
    if 'input_data_path' not in opts:
        if 'data_path' in opts:
            opts.input_data_path = opts.data_path
    
    # general options
    if 'gui' not in opts:
        opts.gui = False
    
    # calc_TEA.py options
    if 'recalc_threshold' not in opts:
        opts.recalc_threshold = False
    if 'recalc_daily' not in opts:
        opts.recalc_daily = True
    if 'decadal' not in opts:
        opts.decadal = True
    if 'decadal_only' not in opts:
        opts.decadal_only = False
    if 'recalc_decadal' not in opts:
        opts.recalc_decadal = True
    if 'hourly' not in opts:
        opts.hourly = False
    if 'compare_to_ref' not in opts:
        opts.compare_to_ref = False
    if 'spreads' not in opts:
        opts.spreads = False
    if 'use_dask' not in opts:
        opts.use_dask = False
    
    # create_region_masks.py options
    if 'gr_type' not in opts:
        opts.gr_type = 'polygon'
    if 'subreg' not in opts or opts.subreg == opts.region:
        opts.subreg = ''
    if 'target_sys' not in opts and 'natural_variability' not in fname:
        if opts.dataset == 'SPARTACUS':
            opts.target_sys = 3416
        elif 'ERA' in opts.dataset:
            opts.target_sys = 4326
        elif opts.dataset == 'HistAlp' or opts.dataset == 'TAWES':
            opts.target_sys = None
        else:
            raise ValueError(f'Unknown dataset {opts.dataset}. Please set target_sys manually in options.')
    if 'xy_name' not in opts and 'natural_variability' not in fname:
        if opts.dataset == 'SPARTACUS':
            opts.xy_name = 'x,y'
        elif 'ERA' in opts.dataset:
            opts.xy_name = 'lon,lat'
        elif 'station' in opts:
            opts.xy_name = None
        else:
            raise ValueError(f'Unknown dataset {opts.dataset}. Please set xy_name manually in options.')
        
    # regrid_SPARTACUS_to_WEGNext.py options
    if 'orography' not in opts:
        opts.orography = False
        
    return opts


def create_history(cli_params, ds):
    """
    add history to dataset
    :param cli_params: CLI parameter
    :param ds: dataset
    :return: ds with history in attrs
    """

    script = cli_params[0].split('/')[-1]
    cli_params = cli_params[1:]

    if 'history' in ds.attrs:
        new_hist = f'{ds.history}; {dt.datetime.now():%FT%H:%M:%S} {script} {" ".join(cli_params)}'

    else:
        new_hist = f'{dt.datetime.now():%FT%H:%M:%S} {script} {" ".join(cli_params)}'

    ds.attrs['history'] = new_hist

    return ds


def create_history_from_cfg(cfg_params, ds):
    """
    add history to dataset
    :param cfg_params: CFG parameter
    :param ds: dataset
    :return: ds with history in attrs
    """

    parts = []
    for key, value in vars(cfg_params).items():
        if key != 'script':
            part = f"--{key} {value}"
            parts.append(part)
    params = ' '.join(parts)

    script = cfg_params.script.split('/')[-1]

    if 'history' in ds.attrs:
        new_hist = f'{ds.history}; {dt.datetime.now():%FT%H:%M:%S} {script} {params}'
    else:
        new_hist = f'{dt.datetime.now():%FT%H:%M:%S} {script} {params}'

    ds.attrs['history'] = new_hist

    return ds


def create_tea_history(cfg_params, tea, result_type):
    """
    add history to dataset
    :param cfg_params: yaml config parameters
    :param tea: TEA object
    :param result_type: result type (e.g. 'CTP')
    """
    ds = getattr(tea, f'{result_type}_results')

    parts = []
    for key, value in vars(cfg_params).items():
        if key != 'script':
            part = f"--{key} {value}"
            parts.append(part)
    params = ' '.join(parts)

    script = cfg_params.script.split('/')[-1]

    if 'history' in ds.attrs:
        new_hist = f'{ds.history}; {dt.datetime.now():%FT%H:%M:%S} {script} {params}'
    else:
        new_hist = f'{dt.datetime.now():%FT%H:%M:%S} {script} {params}'

    tea.create_history(new_hist, result_type)


def create_natvar_history(cfg_params, nv):
    """
    add history to dataset
    :param cfg_params: yaml config parameters
    :param nv: NatVar object
    """
    ds = getattr(nv, f'nv')

    parts = []
    for key, value in vars(cfg_params).items():
        if key != 'script':
            part = f"--{key} {value}"
            parts.append(part)
    params = ' '.join(parts)

    script = cfg_params.script.split('/')[-1]

    if 'history' in ds.attrs:
        new_hist = f'{ds.history}; {dt.datetime.now():%FT%H:%M:%S} {script} {params}'
    else:
        new_hist = f'{dt.datetime.now():%FT%H:%M:%S} {script} {params}'

    nv.create_history(new_hist)


def ref_cc_params():
    params = {'REF': {'start': '1961-01-01', 'end': '1990-12-31',
                      'start_cy': '1966-01-01', 'end_cy': '1986-12-31',
                      'ref_str': 'REF1961-1990'},
              'CC': {'start': '2010-01-01', 'end': '2024-12-31',
                     'start_cy': '2015-01-01', 'end_cy': '2020-12-31',
                     'cc_str': 'CC2010-2024'}}

    return params


def compare_to_ref(tea_result, tea_ref, relative=False):
    for vvar in tea_result.data_vars:
        if vvar in tea_ref.data_vars:
            if relative:
                diff = (tea_result[vvar] - tea_ref[vvar]) / tea_ref[vvar]
                diff = diff.where(np.isfinite(diff), 0)
                threshold = .05
            else:
                diff = tea_result[vvar] - tea_ref[vvar]
                threshold = 5e-5
            max_diff = diff.max(skipna=True).values
            if max_diff > threshold:
                print(f'Maximum difference in {vvar} is {max_diff}')
        else:
            print(f'{vvar} not found in reference file.')


def get_input_filenames(start, end, inpath, param_str, period='annual', hourly=False):
    """
    get input filenames

    :param start: start year
    :type start: int
    :param end: end year
    :type end: int
    :param inpath: input path
    :param param_str: parameter string
    :param period: period of interest. Default is 'annual'
    :type period: str
    :param hourly: if True, return hourly data filenames
    :type hourly: bool

    :return: list of filenames
    """
    # check if inpath is file
    if os.path.isfile(inpath):
        return inpath
    elif '*' in inpath and glob.glob(inpath):
        return sorted(glob.glob(inpath))

    if hourly:
        inpath = f'{inpath}/hourly/'
        h_string = 'hourly_'
    else:
        h_string = ''

    # select only files of interest, if chosen period is 'seasonal' append one year in the
    # beginning to have the first winter fully included
    filenames = []
    if period == 'seasonal' and start != '1961':
        yrs = np.arange(start - 1, end + 1)
    else:
        yrs = np.arange(start, end + 1)
    for yr in yrs:
        year_files = sorted(glob.glob(
            f'{inpath}*{param_str}_{h_string}{yr}*.nc'))
        filenames.extend(year_files)
    return filenames


def extract_period(ds, period, start_year=None, end_year=None):
    """
    select only times of interest

    Args:
        ds: Xarray DataArray or Pandas DataFrame
        period: period of interest (annual, seasonal, ESS, WAS, JJA)
        start_year: start year (in case of seasonal: start year of first winter season (optional)
        end_year: end year (optional)

    Returns:
        ds: Data with selected time period

    """
    if period == 'seasonal':
        first_year = ds.time[0].dt.year
        if start_year is not None and start_year > first_year:
            start = f'{start_year - 1}-12-01 00:00'
            end = f'{end_year}-11-30 23:59'
        else:
            # if first year is first year of record, exclude first winter (data of Dec 1960 missing)
            start = f'{first_year}-03-01 00:00'
            end = f'{last_year}-11-30 23:59'
        ds = ds.loc[start:end]
    elif start_year is not None and end_year is not None:
        start = f'{start_year}-01-01 00:00'
        end = f'{end_year}-12-31 23:59'
        ds = ds.loc[start:end]
    if period in ['ESS', 'WAS', 'JJA']:
        months = {'ESS': np.arange(5, 10), 'WAS': np.arange(4, 11), 'JJA': np.arange(6, 9)}
        if isinstance(ds, xr.DataArray):
            season = ds['time.month'].isin(months[period])
            ds = ds.sel(time=season)
        elif isinstance(ds, pd.DataFrame):
            season = ds.index.month.isin(months[period])
            ds = ds.loc[season]
        else:
            raise ValueError('ds must be either xarray DataArray or pandas DataFrame')
    return ds


def get_gridded_data(start, end, opts, period='annual', hourly=False):
    """
    loads data for parameter and period
    :param start: start year
    :ptype start: int
    :param end: end year
    :ptype end: int
    :param opts: options
    :param period: period to load (annual, seasonal, ESS, WAS, JJA); default: annual
    :ptype period: str
    :param hourly: if True, load hourly data
    :ptype hourly: bool

    :return: dataset of given parameter
    """

    param_str = ''
    parameter = opts.parameter
    if hourly:
        # use correct parameter for hourly data
        if opts.parameter == 'Tx':
            parameter = 'T'

    if opts.dataset == 'SPARTACUS' and not opts.precip:
        param_str = f'{parameter}'
    elif opts.dataset == 'SPARTACUS' and opts.precip:
        param_str = 'RR'

    filenames = get_input_filenames(period=period, start=start, end=end, inpath=opts.input_data_path, param_str=param_str,
                                    hourly=hourly)

    # load relevant years
    logger.info(f'Loading data from {filenames}...')
    try:
        ds = xr.open_mfdataset(filenames, combine='by_coords')
    except ValueError as e:
        logger.warning(f'Error loading data: {e} Trying again with combine="nested"')
        ds = xr.open_mfdataset(filenames, combine='nested')

    # select variable
    if opts.dataset == 'SPARTACUS' and parameter == 'P24h_7to7':
        ds = ds.rename({'RR': parameter})
    data = ds[parameter]

    # get only values from selected period
    data = extract_period(ds=data, period=period, start_year=start, end_year=end)

    if opts.dataset == 'SPARTACUS':
        data = data.drop('lambert_conformal_conic')

    if not opts.use_dask:
        # load data into memory
        data.load()

    return data


def get_csv_data(opts):
    """
    load station data
    Args:
        opts: Config parameters as defined in CFG-PARAMS-doc.md and TEA_CFG_DEFAULT.yaml

    Returns:
        data: interpolated station data

    """

    if opts.parameter == 'Tx':
        pstr = 'Tmax'
        rename_dict = {'tmax': opts.parameter}
    else:
        pstr = 'RR'
        rename_dict = {'nied': opts.parameter}

    # read csv file of station data and set time as index of df
    filenames = f'{opts.input_data_path}{pstr}_{opts.station}*18770101*.csv'
    file = glob.glob(filenames)
    if len(file) == 0:
        filenames = f'{opts.input_data_path}{pstr}_{opts.station}*.csv'
        file = glob.glob(filenames)
    data = pd.read_csv(file[0])
    data['time'] = pd.to_datetime(data['time'])
    data = data.set_index('time')

    # remove timezone information
    data = data.tz_localize(None)

    # rename columns
    data = data.rename(columns=rename_dict)

    # interpolate missing data
    data = interpolate_gaps(opts=opts, data=data)

    # convert to xarray DataArray
    data = data.to_xarray()[opts.parameter]

    # extract only timestamps of interest
    data = extract_period(ds=data, period=opts.period, start_year=opts.start, end_year=opts.end)

    return data


def interpolate_gaps(opts, data):
    """
    interpolates data gaps with average of missing day from other years
    Args:
        opts: CLI parameter
        data: station data

    Returns:
        data: interpolated data
    """

    non_nan = data.loc[data[opts.parameter].notnull(), :]
    start_yr = non_nan.index[0]

    gaps = data[data[opts.parameter].isnull()]
    for igap in gaps.index:
        if igap < start_yr:
            continue
        # select all values from that day of year
        day_data = data[data.index.month == igap.month]
        day_data = day_data[day_data.index.day == igap.day]
        # calculate mean
        fill_val = day_data[opts.parameter].mean(skipna=True)
        # fill gap with fill value
        data.at[igap, opts.parameter] = fill_val

    return data


def area_grid(opts, masks):
    """
    creates grid where each grid cell gets assigned the size of each grid cell in
    areals (1 areal = 100 km^2)
    Args:
        opts: CLI parameter
        masks: masks

    Returns:
        agrid: area grid
    """

    if opts.dataset == 'SPARTACUS':
        # creates area grid in areals
        agrid = masks['nw_mask'] / 100

    else:
        if opts.dataset == 'ERA5':
            delta_fac = 4  # to get 0.25° resolution
        else:
            delta_fac = 10  # to get 0.1°

        lat = masks.lat.values
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
        mask_template = masks['nw_mask']
        agrid = mask_template * y_len * x_len_da
        agrid = agrid / 100

    # apply GR mask
    agrid = agrid.where(masks['lt1500_mask'] == 1)
    agrid = agrid * masks['mask']
    agrid = agrid.rename('area_grid')
    agrid.attrs = {'units': 'areals'}

    # calculate GR size
    gr_size = agrid.sum()
    gr_size = gr_size.rename('GR_size')
    gr_size.attrs = {'units': 'areals'}

    return agrid, gr_size


def calc_percentiles(opts, threshold_min=None, data=None):
    """
    calculate percentile of reference period for each grid point
    Args:
        opts: CLI parameter
        threshold_min: minimum data value for threshold calculation (e.g. 0.99 for precip); optional
        data: data to calculate percentiles for; if not provided, data will be loaded

    Returns:
        thresh: threshold (percentile) grid

    """

    # load data if not provided
    if data is None:
        data = get_gridded_data(start=opts.ref_period[0], end=opts.ref_period[1], opts=opts, period=opts.period)
    else:
        data = extract_period(ds=data, period=opts.period, start_year=opts.ref_period[0], end_year=opts.ref_period[1])

    if threshold_min is not None:
        data = data.where(data > threshold_min)

    # calc the chosen percentile for each grid point as threshold
    percent = data.chunk(dict(time=-1)).quantile(q=opts.threshold / 100, dim='time')

    # smooth SPARTACUS precip percentiles (for each grid point calculate the average of all grid
    # points within the given radius)
    if 'smoothing' in opts:
        radius = opts.smoothing_radius
    else:
        radius = 0

    if radius > 0:
        percent_smooth = smooth_data(percent, radius)
        percent = percent_smooth

    percent = percent.drop_vars('quantile')

    return percent


def smooth_data(data, radius):
    y_size, x_size = data.shape
    percent_smooth_arr = np.full_like(data.values, np.nan)
    percent_tmp = np.zeros((y_size + 2 * radius, x_size + 2 * radius),
                           dtype='float32') * np.nan
    percent_tmp[radius:radius + y_size, radius:radius + x_size] = data
    rad_circ = radius + 0.5
    x_vec = np.arange(0, x_size + 2 * radius)
    y_vec = np.arange(0, y_size + 2 * radius)
    iy_new = 0
    for iy in range(radius, y_size):
        ix_new = 0
        for ix in range(radius, x_size):
            circ_mask = (x_vec[np.newaxis, :] - ix) ** 2 + (y_vec[:, np.newaxis] - iy) ** 2 \
                        < rad_circ ** 2
            percent_smooth_arr[iy_new, ix_new] = np.nanmean(percent_tmp[circ_mask])
            ix_new += 1
        iy_new += 1
    percent_smooth = xr.full_like(data, np.nan)
    percent_smooth[:, :] = percent_smooth_arr
    return percent_smooth


def create_threshold_grid(opts, data=None):
    """
    create threshold grid for the given parameter and reference period
    Args:
        opts: options as defined in CFG-PARAMS-doc.md and TEA_CFG_DEFAULT.yaml
        data: data to calculate threshold grid for; if not provided, data will be loaded

    Returns:
        thr_grid: threshold grid
    """
    if opts.precip:
        threshold_min = 0.99
    else:
        threshold_min = None
    thr_grid = calc_percentiles(opts=opts, threshold_min=threshold_min, data=data)
    thr_grid = thr_grid.rename('threshold')

    if opts.precip:
        ref_str = 'WetDays > 1 mm Ref'
    else:
        ref_str = 'Ref'
    vname = f'{opts.parameter}-p{opts.threshold}{opts.period} {ref_str}{opts.ref_period[0]}-{opts.ref_period[1]}'
    thr_grid.attrs = {'units': opts.unit, 'methods_variable_name': vname, 'percentile': f'{opts.threshold}'}
    return thr_grid
