import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import seaborn as sns
import xarray as xr


def run():
    regs = ['S-EUR', 'C-EUR', 'N-EUR'] #'EUR',
    cols = {'EUR': 'tab:grey',
            'S-EUR': sns.color_palette('Spectral', n_colors=10)[2],
            'C-EUR': sns.color_palette('RdYlGn', n_colors=10)[7],
            'N-EUR': sns.color_palette('Spectral', n_colors=10)[8]}

    fig, axs = plt.subplots(1, 1, figsize=(7, 3.5))
    xticks = np.arange(1960, 2025)

    for ireg, reg in enumerate(regs):
        data = xr.open_dataset(f'/data/users/hst/TEA-clean/TEA/paper_data/dec_indicator_variables/'
                               f'amplification/AF_Tx99.0p_AGR-{reg}_WAS_ERA5_1961to2024.nc')
        data = data['TEX_AGR_AF']

        axs.plot(xticks[1:], data, color=cols[reg], linewidth=2, markersize=3,
                 label=reg)

    axs.set_title('Total Events Extremity (TEX) amplification | EUR & Europe regions',
                  fontsize=12)
    axs.set_xlabel('Time (core year of decadal-mean value)', fontsize=10)
    axs.set_ylabel(r'TEX amplification ($\mathcal{A}_\mathrm{CC}^\mathrm{T}$)',
                   fontsize=10)
    axs.minorticks_on()
    axs.grid(color='gray', which='major', linestyle=':')
    axs.set_xlim(1960, 2025)
    # axs.set_ylim(0, 25)
    axs.xaxis.set_minor_locator(mticker.FixedLocator(np.arange(1960, 2025)))
    axs.legend()

    plt.show()


if __name__ == '__main__':
    run()