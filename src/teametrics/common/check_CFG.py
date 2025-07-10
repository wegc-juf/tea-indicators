"""
script to check CFG parameter
"""
import argparse
import os
import pandas as pd
import re
import yaml
import glob


def is_dir_path(path):
    if os.path.isdir(path) or os.path.exists(path) or glob.glob(path):
        return True
    else:
        raise argparse.ArgumentTypeError(f'{path} is not a valid path.')


def is_file(entry):
    if os.path.isfile(entry):
        return True
    else:
        raise argparse.ArgumentTypeError(f'{entry} is not a valid file')


def float_1pcd(value):
    value = str(value)
    if not re.match(r'^\d+(\.\d{1})?$', value):
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


file_path = os.path.dirname(__file__)


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
        'subreg': str,
        'target_sys': int,
        'xy_name': str,
        'shpfile': 'path',
        'orofile': 'path',
        'lsmfile': 'path',
        
        # regrid_SPARTACUS_to_WEGNext.py
        'orography': bool,
        'orofile': 'path',
        'wegnfile': 'path',
    }
    expected_type = types.get(key, str)
    if expected_type == float:
        try:
            value = float(value)
        except ValueError:
            raise argparse.ArgumentTypeError(f'Expected a float for {key}, but got {value} instead.')
    if not isinstance(value, expected_type):
        raise argparse.ArgumentTypeError(f'Expected type {expected_type} for {key}, '
                                         f'but got {value} of type {type(value)} instead.')


def check_config(opts_dict):
    """
    check configuration parameters for validity
    Args:
        opts_dict: dictionary with configuration parameters

    Returns:
        opts_dict: dictionary with validated configuration parameters

    """
    choice_vals = {
        'threshold_type': ['abs', 'perc'],
        'period': ['monthly', 'seasonal', 'annual', 'WAS', 'ESS', 'JJA'],
        'gr_type': ['polygon', 'corners', 'center'],
        'station': ['Graz', 'Innsbruck', 'Wien', 'Salzburg', 'Kremsmuenster',
                    'BadGleichenberg', 'Deutschlandsberg']}

    for param in opts_dict.keys():
        if 'path' in param:
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
    