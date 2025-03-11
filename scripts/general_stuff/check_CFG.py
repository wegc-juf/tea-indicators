"""
script to check CFG parameter
"""
import argparse
import os
import pandas as pd
import re
import yaml


def dir_path(path):
    if os.path.isdir(path):
        return path
    else:
        raise argparse.ArgumentTypeError(f'{path} is not a valid path.')


def file_path(entry):
    if os.path.isfile(entry):
        return entry
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


def check_config(opts_dict):
    with open('../TEA_CFG_DEFAULTS.yaml', 'r') as stream:
        defaults = yaml.safe_load(stream)
        defaults = defaults['calc_TEA']

    choice_vals = {'threshold_type': ['abs', 'perc'],
                   'period': ['monthly', 'seasonal', 'annual', 'WAS', 'ESS', 'JJA'],
                   'gr_type': ['polygon', 'corners', 'center'],
                   'station': ['Graz', 'Innsbruck', 'Wien', 'Salzburg', 'Kremsmuenster',
                               'BadGleichenberg', 'Deutschlandsberg']}

    for param in opts_dict.keys():
        # set default value if None was passed
        if opts_dict[param] is None:
            opts_dict[param] = defaults[param]
            continue
        else:
            if 'path' in param:
                dir_path(opts_dict[param])
            if 'file' in param:
                file_path(opts_dict[param])
            if param in ['precip', 'low_extreme', 'decadal', 'spreads', 'decadal_only',
                         'recalc_daily', 'orography',
                         'recalc_decadal', 'compare_to_ref', 'save_old']:
                bools(param=param, val=opts_dict[param])
            if param in ['region', 'parameter', 'unit', 'subreg', 'target_ds', 'dataset',
                         'xy_name']:
                strs(param=param, val=opts_dict[param])
            if param == 'threshold':
                float_1pcd(opts_dict[param])
            if param in ['threshold_type', 'period', 'gr_type']:
                choices(param=param, val=opts_dict[param], poss_vals=choice_vals[param])
            if param in ['start', 'end', 'target_sys', 'smoothing']:
                ints(param=param, val=opts_dict[param])

    return opts_dict
