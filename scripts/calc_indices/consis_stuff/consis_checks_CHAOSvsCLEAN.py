import argparse
import cftime as cft
import glob
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
import os
import xarray as xr


def get_opts():
    """
    get CLI parameter
    Returns:
        opts: CLI parameter
    """

    parser = argparse.ArgumentParser()

    parser.add_argument('--parameter',
                        default='Tx',
                        choices=['Tx', 'P24h_7to7'],
                        type=str,
                        help='Parameter for which the TEA indices should be calculated'
                             '[default: Tx].')

    parser.add_argument('--threshold',
                        default=99,
                        type=float,
                        help='Threshold in degrees Celsius, mm, or as percentile [default: 99].')

    parser.add_argument('--region',
                        default='AUT',
                        type=str,
                        help='Geo region [options: AUT (default), Austrian state name, '
                             'or ISO2 code of european country].')

    parser.add_argument('--level',
                        dest='level',
                        default='static',
                        choices=['static', 'basis', 'ctp', 'dec'],
                        type=str,
                        help='Calculation level to compare.')

    parser.add_argument('--dataset',
                        dest='dataset',
                        default='SPARTACUS',
                        choices=['SPARTACUS', 'ERA5', 'ERA5Land'],
                        type=str,
                        help='Input dataset [default: SPARTACUS].')

    myopts = parser.parse_args()

    return myopts


def check_static(opts):
    # new static files
    dash = ''
    if opts.parameter == 'P24h_7to7':
        dash = '_'
    nstats = xr.open_dataset(f'/data/arsclisys/normal/clim-hydro/TEA-Indicators/static/'
                             f'static_{opts.parameter}{dash}{opts.threshold:.1f}p_{opts.region}'
                             f'_{opts.dataset}.nc')

    ods = opts.dataset
    if opts.dataset == 'SPARTACUS':
        ods = 'SPARTACUSv1.5reg'

    pvar = 'T'
    if opts.parameter == 'P24h_7to7':
        pvar = 'Precip24Hsum_7to7'

    # old static files
    ostat_path = '/data/arsclisys/backup/clim-hydro/gcci_ewm/actea_input/static/'
    oarea = np.load(f'{ostat_path}area_grid_{ods}.npy')
    othr = np.load(
        f'{ostat_path}thresholds/threshold_grid_{opts.region}_{pvar}_{opts.threshold:.0f}percentile'
        f'_{ods}.npy')
    ogr = np.load(f'{ostat_path}gr_sizes/{opts.region}_total_size_{ods}.npy')

    # create plots
    # area
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    oavals = axs[0].contourf(oarea, levels=np.arange(0, 8, 0.5))
    divider1 = make_axes_locatable(axs[0])
    cax1 = divider1.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(oavals, cax=cax1, orientation='vertical')
    axs[0].set_title('OLD AREA')

    navals = axs[1].contourf(nstats.area_grid, levels=np.arange(0, 8, 0.5))
    divider2 = make_axes_locatable(axs[1])
    cax2 = divider2.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(navals, cax=cax2, orientation='vertical')
    axs[1].set_title('NEW AREA')

    # thresholds
    fig2, axs2 = plt.subplots(1, 2, figsize=(10, 5))
    otvals = axs2[0].contourf(othr, levels=np.arange(0, 35, 1))
    divider3 = make_axes_locatable(axs2[0])
    cax3 = divider3.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(otvals, cax=cax3, orientation='vertical')
    axs2[0].set_title('OLD THRESHS')

    ntvals = axs2[1].contourf(nstats.threshold, levels=np.arange(0, 35, 1))
    divider4 = make_axes_locatable(axs2[1])
    cax4 = divider4.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(ntvals, cax=cax4, orientation='vertical')
    axs2[1].set_title('NEW THRESHS')

    # print stats
    print('AREA')
    print('old (min|max|median)')
    print(np.nanmin(oarea), '|', np.nanmax(oarea), '|', np.nanmedian(oarea))
    print('new (min|max|median)')
    print(nstats.area_grid.min().values, '|', nstats.area_grid.max().values, '|',
          nstats.area_grid.median().values)

    print('THRESH')
    print('old (min|max|median)')
    print(np.nanmin(othr), '|', np.nanmax(othr), '|', np.nanmedian(othr))
    print('new (min|max|median)')
    print(nstats.threshold.min().values, '|', nstats.threshold.max().values, '|',
          nstats.threshold.median().values)

    print('GR size')
    print('old')
    print(ogr)
    print('new')
    print(nstats.GR_size.values)

    plt.show()


def check_basis(opts):
    nbv = xr.open_dataset(f'/data/users/hst/TEA-clean/TEA/daily_basis_variables/'
                          f'DBV_P24h_7to7_95.0p_{opts.region}_{opts.dataset}_1961to1970.nc')
    obv = xr.open_dataset(f'/home/hst/tmp_data/TEAclean/basic_vars_P24h.nc')

    return


def preprocess(ds_in):
    ds = ds_in.copy()
    vkeep = ['TEF_GR', 'EF_GR', 'ED_GR', 'EDavg_GR', 'EM_GR', 'EMavg_GR', 'EA_GR', 'EAavg_GR']
    vdrop = [vvar for vvar in ds.data_vars if vvar not in vkeep]
    ds = ds.drop_vars(vdrop)

    return ds


def check_ctp_tea(opts):
    nfiles = sorted(glob.glob(f'/data/users/hst/TEA-clean/TEA/ctp_indicator_variables/'
                              f'CTP_{opts.parameter}{opts.threshold:.1f}p_{opts.region}'
                              f'_WAS_{opts.dataset}_*.nc'))
    new = xr.open_mfdataset(nfiles, preprocess=preprocess, data_vars='minimal')

    pstr = ''
    if opts.parameter == 'Tx':
        pstr = 'T'
    elif opts.parameter == 'P24h_7to7':
        pstr = 'RR'

    ofiles = sorted(glob.glob(f'/data/users/hst/cdrDPS/ACTEM_indices/ACTEM/Temperature/'
                              f'ACTEM_indices_{opts.region}_{pstr}'
                              f'_{opts.threshold}percentile_WAS_*'
                              f'_{opts.dataset}.nc'))
    old = xr.open_mfdataset(ofiles, preprocess=preprocess, data_vars='minimal')
    old['periods'] = cft.num2pydate(old['periods'], units='days since 1961-01-01')

    # plot basis GR vars
    fig, axs = plt.subplots(2, 2, figsize=(15, 10))
    axs[0, 0].plot(old['periods'], old['TEF_GR'], marker='o', color='tab:grey')
    axs[0, 0].plot(new['ctp'], new['EF_GR'], marker='o', linestyle='--', color='tab:blue')
    axs[0, 0].set_title('Frequency')

    axs[0, 1].plot(old['periods'], old['ED_GR'], marker='o', color='tab:grey')
    axs[0, 1].plot(new['ctp'], new['EDavg_GR'], marker='o', linestyle='--', color='tab:purple')
    axs[0, 1].set_title('Duration')

    axs[1, 0].plot(old['periods'], old['EM_GR'], marker='o', color='tab:grey')
    axs[1, 0].plot(new['ctp'], new['EMavg_GR'], marker='o', linestyle='--', color='tab:orange')
    axs[1, 0].set_title('Magnitude')

    axs[1, 1].plot(old['periods'], old['EA_GR'], marker='o', color='tab:grey')
    axs[1, 1].plot(new['ctp'], new['EAavg_GR'], marker='o', linestyle='--', color='tab:red')
    axs[1, 1].set_title('Area')

    plt.show()


def check_dec_tea(opts):
    agr_str, vend = '', 'GR'
    if 'ERA5' in opts.dataset and opts.region != 'FBR':
        agr_str, vend = 'AGR-', 'AGR'
    new = xr.open_dataset(f'/data/users/hst/TEA-clean/TEA/dec_indicator_variables/'
                          f'DEC_T99.0p_{agr_str}{opts.region}_WAS_{opts.dataset}_1961to2022.nc')
    old = xr.open_dataset(f'/data/users/hst/cdrDPS/ACTEM_indices/ACTEM/decadal/'
                          f'TEA_indicators_{opts.region}_T_99percentile_decadal'
                          f'_WAS_1961_to_2022_{opts.dataset}.nc')

    old['periods'] = cft.num2pydate(old['periods'], units='days since 1961-01-01')

    if 'ERA5' in opts.dataset and opts.region != 'FBR':
        # load EA data
        files = sorted(glob.glob(f'/data/users/hst/cdrDPS/ACTEM_indices/EA_EUR_grid/'
                                 f'EA_EUR_grid_Temperature_AUT_*_{opts.dataset}.nc'))

        ds = xr.open_mfdataset(files, data_vars='minimal')
        ea = ds.AV_EA_GR
        ea = ea.transpose('periods', 'lat', 'lon')
        ea = ea.rename('EA_GR')
        # calculate decadal means of EA
        weights = xr.DataArray([1, 1, 1, 1, 1, 1, 1, 1, 1, 1], dims=['window']) / 10
        ea = ea.rolling(periods=10, center=True).construct('window').dot(weights)
        ea = ea.mean(dim=('lat', 'lon'))
        old['EA_GR'] = ea

    # plot basis GR vars
    fig, axs = plt.subplots(2, 2, figsize=(15, 10))
    axs[0, 0].plot(old['periods'], old['TEF_GR'], marker='o', color='tab:grey')
    axs[0, 0].plot(new['ctp'], new[f'EF_{vend}'], marker='o', linestyle='--', color='tab:blue')
    axs[0, 0].set_title('Frequency')

    axs[0, 1].plot(old['periods'], old['ED_GR'], marker='o', color='tab:grey')
    axs[0, 1].plot(new['ctp'], new[f'EDavg_{vend}'], marker='o', linestyle='--', color='tab:purple')
    axs[0, 1].set_title('Duration')

    axs[1, 0].plot(old['periods'], old['EM_GR'], marker='o', color='tab:grey')
    axs[1, 0].plot(new['ctp'], new[f'EMavg_{vend}'], marker='o', linestyle='--', color='tab:orange')
    axs[1, 0].set_title('Magnitude')

    axs[1, 1].plot(old['periods'], old['EA_GR'], marker='o', color='tab:grey')
    axs[1, 1].plot(new['ctp'], new[f'EAavg_{vend}'], marker='o', linestyle='--', color='tab:red')
    axs[1, 1].set_title('Area')

    plt.show()


def run():
    opts = get_opts()

    if opts.level == 'static':
        check_static(opts=opts)
    elif opts.level == 'basis':
        check_basis(opts=opts)
    elif opts.level == 'ctp':
        check_ctp_tea(opts=opts)
    else:
        check_dec_tea(opts=opts)


if __name__ == '__main__':
    run()
