#!/usr/bin/env python3

import argparse
import glob
import os
import pandas as pd
import xarray as xr
from glob import glob

from NatVar import NaturalVariability
from teametrics.common.general_functions import create_natvar_history
from teametrics.common.config import load_opts


def _getopts():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--config-file', '-cf',
                        dest='config_file',
                        type=str,
                        default='../TEA_CFG.yaml',
                        help='TEA configuration file (default: TEA_CFG.yaml)')
    
    myopts = parser.parse_args()
    
    return myopts


def run():
    opts = _getopts()
    opts = load_opts('calc_TEA', opts.config_file)
    testing = False
    
    if not testing:
        file_mask = (f'{opts.outpath}/ctp_indicator_variables/CTP_{opts.param_str}'
                     f'_{opts.region}_{opts.period}'
                     f'_SPARTACUS_*.nc')
    else:
        file_mask = (f'{opts.outpath}/ctp_indicator_variables/CTP_{opts.param_str}'
                     f'_{opts.region}_{opts.period}'
                     f'_SPARTACUS_1961to1962.nc')
    filenames = sorted(glob(file_mask))
    if not filenames:
        raise FileNotFoundError(f'No files found for pattern: {file_mask}')
    print(f'Loading data from files: {filenames}')
    tea_ds = xr.open_mfdataset(filenames, combine='by_coords')
    hdd_data = tea_ds.EM.mean(dim=['x', 'y'])
    # Save to CSV
    output_csv = os.path.join(opts.outpath, f'HDD_{opts.region}_{opts.period}_SPARTACUS_{opts.start}to'
                                            f'{opts.end}.csv')
    print(f'Saving HDD data to {output_csv}')
    hdd_data.to_dataframe(name='HDD').to_csv(output_csv, float_format='%.1f')

    
if __name__ == '__main__':
    run()
