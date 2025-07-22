"""
script to check CFG parameter
"""
import os
import re
import glob

import argparse
import pandas as pd
import numpy as np
import yaml
import cfunits

from .cfg_paramter_gui import show_parameters
import teametrics


def is_dir_path(path):
    if os.path.isdir(path) or os.path.exists(path) or glob.glob(path):
        return True
    else:
        raise argparse.ArgumentTypeError(f'{path} not found or is not a valid directory or file path. ')


def is_file(entry):
    if os.path.isfile(entry):
        return True
    else:
        raise argparse.ArgumentTypeError(f'{entry} is not a valid file')


def float_1pcd(value):
    value = str(value)
    if not re.match(r'^\d+(\.\d)?$', value):
        raise argparse.ArgumentTypeError('Threshold value must have at most one digit after '
                                         'the decimal point.')
    return float(value)


def bools(param, val):
    if not isinstance(val, bool):
        raise argparse.ArgumentTypeError(f'{val} is not a valid value for {param}. '
                                         f'Please pass a boolean (true/false).')


def strs(param, val):
    if not isinstance(param, str):
        raise argparse.ArgumentTypeError(f'{val} is not a valid value for {param}. '
                                         f'Please pass a string.')


def choices(param, val, poss_vals):
    if val not in poss_vals:
        raise argparse.ArgumentTypeError(f'{val} is not a valid value for {param}. '
                                         f'Please choose one of hte following: {poss_vals}.')


def ints(param, val):
    if not isinstance(val, int):
        raise argparse.ArgumentTypeError(f'{val} is not a valid value for {param}. '
                                         f'Please pass an integer.')
    if param in ['start', 'end']:
        if val > pd.to_datetime('today').year:
            raise argparse.ArgumentTypeError(f'{val} is not a valid value for {param}. '
                                             f'Please pass a year before the current year or the '
                                             f'current year.')


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
    # TODO: add minimum exceedance area option as defined in gki RCF

    # time_params options
    if 'start' not in opts:
        opts.start = 1961
    if 'end' not in opts:
        opts.end = 2024
    if 'period' not in opts:
        opts.period = 'annual'
    if 'ref_period' not in opts:
        opts.ref_period = '1961-1990'
    # TODO: add separate option for percentile period as defined in gki RCF
    # TODO: add option for percentile estimation period as defined in gki RCF
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
    if fname == 'calc_TEA':
        if 'recalc_threshold' not in opts:
            opts.recalc_threshold = False
        if 'recalc_daily' not in opts:
            opts.recalc_daily = True
        if 'decadal' not in opts:
            opts.decadal = True
        # TODO: add decadal window options as defined in gki RCF
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
    if fname == 'create_region_masks':
        if 'gr_type' not in opts:
            opts.gr_type = 'polygon'
        if 'subreg' not in opts or opts.subreg == opts.region:
            opts.subreg = False
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
    if fname == 'regrid_SPARTACUS_to_WEGNext':
        if 'orography' not in opts:
            opts.orography = False
    
    return opts


def check_type(key, value):
    """
    Check if the value is of the expected type.
    """
    types = {
        # input data
        'dataset': str,
        
        # GeoRegion
        'region': str,
        'station': str,
        'agr': str,
        'agr_cell_size': float,
        'grg_grid_spacing': float,
        'land_frac_min': float,
        
        # Parameters
        'parameter': str,
        'precip': bool,
        'threshold_type': str,
        'threshold': float,
        'smoothing_radius': float,
        'unit': str,
        'low_extreme': bool,
        
        # time parameters
        'start': int,
        'end': int,
        'period': str,
        
        # general
        'gui': bool,
        
        # calc_TEA.py
        'recalc_threshold': bool,
        'hourly': bool,
        'recalc_daily': bool,
        'decadal': bool,
        'recalc_decadal': bool,
        'decadal_only': bool,
        'spreads': bool,
        'use_dask': bool,
        'compare_to_ref': bool,
        
        # create_region_masks.py
        'gr_type': str,
        'we_len': float,
        'ns_len': float,
        'subreg': bool,
        'target_sys': int,
        'xy_name': str,
        'shpfile': 'path',
        'orofile': 'path',
        'lsmfile': 'path',
        
        # regrid_SPARTACUS_to_WEGNext.py
        'orography': bool,
        'wegnfile': 'path',
    }
    expected_type = types.get(key, str)
    if value == None or 'file' in key:
        return
    if expected_type == float:
        try:
            value = float(value)
        except ValueError:
            raise argparse.ArgumentTypeError(f'Expected a float for {key}, but got {value} instead.')
    if not isinstance(value, expected_type):
        raise argparse.ArgumentTypeError(f'Expected type {expected_type} for {key}, '
                                         f'but got {value} of type {type(value)} instead.')
    # check for correct unit
    if key == 'unit':
        unit = cfunits.Units(value)
        if not unit.isvalid:
            raise argparse.ArgumentTypeError(f'{unit.reason_notvalid}. '
                                             f'Please use a valid unit from udunits.')


def set_variables(opts_dict):
    """
    replace variables in opts_dict with their values
    Args:
        opts_dict: dictionary with configuration parameters

    Returns:

    """
    for param in opts_dict.keys():
        if not isinstance(opts_dict[param], str):
            continue
        if '$script_path' in opts_dict[param]:
            # replace 'script_path' with the path of the script
            script_path = os.path.dirname(os.path.abspath(teametrics.__file__))
            opts_dict[param] = opts_dict[param].replace('$script_path', script_path)
        elif '$' in opts_dict[param]:
            raise ValueError(f'Unknown variable name in {param}: {opts_dict[param]}. ')


def check_config(opts_dict):
    """
    check configuration parameters for validity
    Args:
        opts_dict: dictionary with configuration parameters

    Returns:
        opts_dict: dictionary with validated configuration parameters

    """
    choice_vals = {
        'station': ['Graz', 'Innsbruck', 'Wien', 'Salzburg', 'Kremsmuenster',
                    'BadGleichenberg', 'Deutschlandsberg'],
        'threshold_type': ['abs', 'perc'],
        'period': ['monthly', 'seasonal', 'annual', 'WAS', 'ESS', 'MAM', 'JJA', 'SON', 'DJF'],
        'gr_type': ['polygon', 'corners', 'center'],
    }

    for param in opts_dict.keys():
        if 'path' in param or 'file' in param:
            is_dir_path(opts_dict[param])
        else:
            check_type(param, opts_dict[param])
        if 'file' in param:
            is_file(opts_dict[param])
        if param in ['precip', 'low_extreme', 'decadal', 'spreads', 'decadal_only',
                     'recalc_daily', 'orography', 'recalc_decadal', 'gui']:
            bools(param=param, val=opts_dict[param])
        if param in ['region', 'parameter', 'unit', 'subreg', 'dataset', 'xy_name']:
            strs(param=param, val=opts_dict[param])
        if param == 'threshold':
            float_1pcd(opts_dict[param])
        if param in choice_vals.keys():
            choices(param=param, val=opts_dict[param], poss_vals=choice_vals[param])
        if param in ['start', 'end', 'target_sys', 'smoothing']:
            ints(param=param, val=opts_dict[param])
    
    if 'input_data_path' not in opts_dict:
        raise ValueError('input_data_path not set in options. Please set it in the CFG file.')


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
        if 'plot' in fname:
            opts = opts['calc_TEA']
        else:
            opts = opts[fname]
        opts = argparse.Namespace(**opts)

    # add name of script
    opts.script = f'{fname}.py'
    
    opts = _get_default_opts(fname, opts)
    set_variables(opts_dict=vars(opts))
    check_config(opts_dict=vars(opts))
    
    if opts.gui:
        # show set parameters
        show_parameters(opts)
        set_variables(opts_dict=vars(opts))
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
