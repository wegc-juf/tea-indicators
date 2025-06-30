import argparse
import glob
import os
import pandas as pd
import xarray as xr

from scripts.additional_stuff.NatVar import NaturalVariability
from scripts.general_stuff.general_functions import load_opts, create_natvar_history


def _getopts():
    parser = argparse.ArgumentParser()

    parser.add_argument('--config-file', '-cf',
                        dest='config_file',
                        type=str,
                        default='../TEA_CFG.yaml',
                        help='TEA configuration file (default: TEA_CFG.yaml)')

    myopts = parser.parse_args()

    return myopts


def load_station_data(opts):
    """
    load TEA Indicators for the given region and stations
    Args:
        opts: CLI parameter
        af: set to True to load amplification factors data

    Returns:
        ds: station data

    """
    if opts.region == 'AUT':
        stations = ['Graz', 'Wien', 'Innsbruck', 'Salzburg', 'Kremsmuenster']
    elif opts.region == 'SEA':
        stations = ['Graz', 'BadGleichenberg', 'Deutschlandsberg']
    else:
        stations = opts.stations.split(',')

    dm_var, em_var = 'DM_avg', 'EM_avg'
    if opts.parameter != 'Tx':
        dm_var, em_var = 'DM_Md', 'EM_avg_Md'

    files = sorted(glob.glob(f'{opts.tea_path}station/dec_indicator_variables/amplification/'
                             f'*{opts.param_str}*.nc'))

    # only select files for the given stations
    files = [f for f in files if any(station in f for station in stations)]

    def extract_station_name(filename):
        for station in stations:
            if station in filename:
                return station
        return None

    datasets = []
    for ifile, file in enumerate(files):
        ds_station = xr.open_dataset(file)

        # Add a new 'station' coordinate (single value)
        station_name = extract_station_name(os.path.basename(file))
        if station_name == 'Kremsmuenster':
            ds_station = ds_station.where(ds_station['time'] > pd.Timestamp('1920-01-01'))
        ds_station = ds_station.expand_dims({'station': [station_name]})

        datasets.append(ds_station)

    ds = xr.concat(datasets, dim='station')

    keeps = [v for v in ds.data_vars if any(
        vvar in v for vvar in ['EF', 'ED_avg', em_var, dm_var, 'tEX'])]
    drops = [vvar for vvar in ds.data_vars if vvar not in keeps or '_s' in vvar or 'CC' in vvar]

    ds = ds.drop(drops)

    # add DM variables
    ds[f'{dm_var}_AF'] = ds['ED_avg_AF'] * ds[f'{em_var}_AF']

    return ds


def load_spartacus_data(opts):

    ds = xr.open_dataset(f'{opts.tea_path}/dec_indicator_variables/amplification/'
                         f'AF_{opts.param_str}_{opts.region}_{opts.period}_SPARTACUS'
                         f'_{opts.start}to{opts.end}.nc')

    dm_var, em_var = 'DM_avg', 'EM_avg'
    if opts.parameter != 'Tx':
        dm_var, em_var = 'DM_Md', 'EM_avg_Md'

    keeps = [v for v in ds.data_vars if
             any(vvar in v for vvar in ['EF', 'ED_avg', em_var, 'EA_avg', dm_var, 'tEX'])]
    drops = [vvar for vvar in ds.data_vars if vvar not in keeps or '_s' in vvar or 'GR' not in vvar
             or 'CC' in vvar]
    drops.extend(['x', 'y'])

    ds = ds.drop(drops)

    # rename variables to match station data names (i.e., remove '_GR')
    rename_dict = {}
    for dvar in ds.data_vars:
        parts = dvar.split('_GR')
        new_name = ''
        for ipart in parts:
            new_name += ipart
        rename_dict[dvar] = new_name

    ds = ds.rename(rename_dict)

    if opts.parameter == 'Tx':
        ds = ds.drop(['EM_avg_Md_AF'])

    ds = ds.drop('EM_avg_Max_AF')

    # add DM variables
    ds[f'{dm_var}_AF'] = ds['ED_avg_AF'] * ds[f'{em_var}_AF']

    return ds


def run():
    # get command line parameters
    cmd_opts = _getopts()

    # load CFG parameters
    opts = load_opts(fname=__file__, config_file=cmd_opts.config_file)

    # load data
    station_af = load_station_data(opts=opts)
    spcus_af = load_spartacus_data(opts=opts)

    # initialize NaturalVariability class
    natvar = NaturalVariability(station_data_af=station_af, spcus_data_af=spcus_af,
                                param=opts.parameter, ref_period=opts.ref_period)

    # calculate standard deviation during REF period
    natvar.calc_ref_std()

    # calculate scaling factors, natural variability, and natural variability of combined data
    natvar.calc_factors()
    natvar.calc_natvar()
    natvar.calc_combined()

    # add history and save results
    create_natvar_history(cfg_params=opts, nv=natvar)
    natvar.save_results(outname=f'{opts.outpath}NV_AF_{opts.param_str}_{opts.region}.nc')


if __name__ == '__main__':
    run()
