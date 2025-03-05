"""
scripts for general stuff (e.g. nc-history)
"""

import argparse
import datetime as dt
import yaml
import numpy as np

from scripts.general_stuff.check_CFG import check_config


def load_opts(fname):
    """
    load parameters from CFG file and put them into a Namespace object
    Args:
        fname: name of executed script

    Returns:
        opts: CFG parameter

    """

    fname = fname.split('/')[-1].split('.py')[0]
    with open('../TEA_CFG.yaml', 'r') as stream:
        opts = yaml.safe_load(stream)
        opts = opts[fname]
        opts = check_config(opts_dict=opts)
        opts = argparse.Namespace(**opts)

    # add name of script
    opts.script = f'{fname}.py'

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
    ref_period = opts.ref_period.split('-')
    opts.ref_period = (int(ref_period[0]), int(ref_period[1]))
    cc_period = opts.cc_period.split('-')
    opts.cc_period = (int(cc_period[0]), int(cc_period[1]))

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


def ref_cc_params():
    params = {'REF': {'start': '1961-01-01', 'end': '1990-12-31',
                      'start_cy': '1966-01-01', 'end_cy': '1986-12-31',
                      'ref_str': 'REF1961-1990'},
              'CC': {'start': '2010-01-01', 'end': '2024-12-31',
                     'start_cy': '2015-01-01', 'end_cy': '2020-12-31',
                     'cc_str': 'CC2010-2024'}}

    return params


def extend_tea_opts(opts):
    """
    add strings that are often needed to opts
    Args:
        opts: CLI parameter

    Returns:

    """

    pstr = opts.parameter
    if opts.parameter != 'Tx':
        pstr = f'{opts.parameter}_'

    param_str = f'{pstr}{opts.threshold:.1f}p'
    if opts.threshold_type == 'abs':
        param_str = f'{pstr}{opts.threshold:.1f}{opts.unit}'

    opts.param_str = param_str

    return opts


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


