"""
scripts for general stuff (e.g. nc-history)
"""

import datetime as dt


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


def create_tea_history(cli_params, tea, result_type):
    """
    add history to dataset
    :param cli_params: CLI parameter
    :param tea: TEA object
    """
    
    script = cli_params[0].split('/')[-1]
    cli_params = cli_params[1:]
    
    new_hist = f'{dt.datetime.now():%FT%H:%M:%S} {script} {" ".join(cli_params)}'
    tea.create_history(new_hist, result_type)


def ref_cc_params():
    params = {'REF': {'start': '1961-01-01', 'end': '1990-12-31',
                      'start_cy': '1966-01-01', 'end_cy': '1986-12-31'},
              'CC': {'start': '2008-01-01', 'end': '2022-12-31',
                     'start_cy': '2013-01-01', 'end_cy': '2018-12-31'}}
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


def compare_to_ref(tea_result, tea_ref):
    for vvar in tea_result.data_vars:
        if vvar in tea_ref.data_vars:
            diff = tea_result[vvar] - tea_ref[vvar]
            max_diff = diff.max(skipna=True).values
            if max_diff > 5e-5:
                logger.warning(f'Maximum difference in {vvar} is {max_diff}')
        else:
            logger.warning(f'{vvar} not found in reference file.')


