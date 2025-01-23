"""
scripts for general stuff (e.g. nc-history)
"""

import argparse
import datetime as dt
import yaml

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
    pstr = opts.parameter
    if opts.parameter != 'Tx':
        pstr = f'{opts.parameter}_'

    param_str = f'{pstr}{opts.threshold:.1f}p'
    if opts.threshold_type == 'abs':
        param_str = f'{pstr}{opts.threshold:.1f}{opts.unit}'

    opts.param_str = param_str

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


def ref_cc_params():
    params = {'REF': {'start': '1961-01-01', 'end': '1990-12-31',
                      'start_cy': '1966-01-01', 'end_cy': '1986-12-31',
                      'ref_str': 'REF1961-1990'},
              'CC': {'start': '2010-01-01', 'end': '2024-12-31',
                     'start_cy': '2015-01-01', 'end_cy': '2020-12-31',
                     'cc_str': 'CC2010-2024'}}
    # params = {'REF': {'start': '1961-01-01', 'end': '1990-12-31',
    #                   'start_cy': '1966-01-01', 'end_cy': '1986-12-31',
    #                   'ref_str': 'REF1961-1990'},
    #           'CC': {'start': '2008-01-01', 'end': '2022-12-31',
    #                  'start_cy': '2013-01-01', 'end_cy': '2018-12-31',
    #                  'cc_str': 'CC2008-2022'}}
    return params


def extend_tea_opts(opts):
    """
    add strings that are often needed to parameters (opts)
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