import argparse
import glob
import os
import xarray as xr

from scripts.additional_stuff.NatVar import NaturalVariability
from scripts.general_stuff.general_functions import (load_opts, create_history_from_cfg)


def _getopts():
    parser = argparse.ArgumentParser()

    parser.add_argument('--config-file', '-cf',
                        dest='config_file',
                        type=str,
                        default='../TEA_CFG.yaml',
                        help='TEA configuration file (default: TEA_CFG.yaml)')

    myopts = parser.parse_args()

    return myopts


def load_station_data(opts, af=False):
    """
    load TEA Indicatos for the given region and stations
    Args:
        opts: CLI parameter

    Returns:
        ds: station data

    """
    if opts.region == 'AUT':
        stations = ['Graz', 'Wien', 'Innsbruck', 'Salzburg', 'Kremsmuenster']
    elif opts.region == 'SEA':
        stations = ['Graz', 'BadGleichenberg', 'Deutschlandsberg']
    else:
        stations = opts.stations.split(',')

    sdir = ''
    if af:
        sdir = 'amplification/'

    files = sorted(glob.glob(f'{opts.tea_path}station/dec_indicator_variables/{sdir}'
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
        ds_station = ds_station.expand_dims({'station': [station_name]})

        datasets.append(ds_station)

    ds = xr.concat(datasets, dim='station')

    keeps = [v for v in ds.data_vars if any(
        vvar in v for vvar in ['EF', 'ED_avg', 'EM_avg', 'EM_Md', 'DM_avg', 'DM_Md', 'tEX'])]
    drops = [vvar for vvar in ds.data_vars if vvar not in keeps or '_s' in vvar or 'CC' in vvar]

    ds = ds.drop(drops)

    # add DM variables
    if af:
        ds['DM_avg_AF'] = ds['ED_avg_AF'] * ds['EM_avg_AF']
        ds['DM_Md_AF'] = ds['ED_avg_AF'] * ds['EM_Md_AF']
    else:
        ds['DM_avg'] = ds['ED_avg'] * ds['EM_avg']
        ds['DM_Md'] = ds['ED_avg'] * ds['EM_Md']

    return ds


def load_spartacus_data(opts, af=False):

    sdir = ''
    abbr = 'DEC'
    if af:
        sdir = 'amplification/'
        abbr = 'AF'
    ds = xr.open_dataset(f'{opts.tea_path}/dec_indicator_variables/{sdir}'
                         f'{abbr}_{opts.param_str}_{opts.region}_{opts.period}_SPARTACUS'
                         f'_{opts.start}to{opts.end}.nc')

    keeps = [v for v in ds.data_vars if
             any(vvar in v for vvar in ['EF', 'ED_avg', 'EM_avg', 'EM_Md', 'DM_avg', 'DM_Md', 'tEX'])]
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

    # add DM variables
    if af:
        ds = ds.rename({'EM_avg_Md_AF': 'EM_Md_AF'})
        ds['DM_avg_AF'] = ds['ED_avg_AF'] * ds['EM_avg_AF']
        ds['DM_Md_AF'] = ds['ED_avg_AF'] * ds['EM_Md_AF']
    else:
        ds = ds.rename({'EM_avg_Md': 'EM_Md'})
        ds['DM_avg'] = ds['ED_avg'] * ds['EM_avg']
        ds['DM_Md'] = ds['ED_avg'] * ds['EM_Md']

    return ds


def run():
    # get command line parameters
    cmd_opts = _getopts()

    # load CFG parameters
    opts = load_opts(fname=__file__, config_file=cmd_opts.config_file)

    # load data
    station_data = load_station_data(opts=opts)
    station_data_af = load_station_data(opts=opts, af=True)
    spcus_data = load_spartacus_data(opts=opts)
    spcus_data_af = load_spartacus_data(opts=opts, af=True)

    # initialize NaturalVariability class
    natvar = NaturalVariability(station_data=station_data, spcus_data=spcus_data,
                                station_data_af=station_data_af, spcus_data_af=spcus_data_af)

    natvar.calc_ref_std(ref_period=opts.ref_period)
    natvar.calc_factors()
    natvar.calc_natvar(ref_eyr=opts.ref_period[1])

    # TODO: save to output nc
    # TODO: crosscheck with old results


if __name__ == '__main__':
    run()
