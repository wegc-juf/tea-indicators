import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

def load_data():
    """
    Load rainfall and temperature data for the SEA region from GSA stations.
    Returns:
        df_rr (pd.DataFrame): DataFrame containing daily rainfall data.
        df_tm (pd.DataFrame): DataFrame containing daily temperature data.

    """
    stations = pd.read_csv(
                f'/data/users/hst/additional_stuff/heavyrainfall_Haslinger2025/data/raw/'
                f'TEAprep/SEA_stations.csv', index_col=0)

    df_rr = pd.read_csv(f'/data/users/hst/additional_stuff/heavyrainfall_Haslinger2025/'
                         f'data/raw/TEAprep/ts_hourly_gsa.csv', index_col=0)
    df_tm = pd.read_csv(f'/data/users/hst/additional_stuff/heavyrainfall_Haslinger2025/'
                         f'data/raw/TEAprep/t2m_hourly_gsa.csv', index_col=0)
    station_idx_gsa = [idx[4:] for idx in stations['id'] if 'gsa' in idx]
    df_rr = df_rr.loc[:, station_idx_gsa]
    df_tm = df_tm.loc[:, station_idx_gsa]

    # drop first index which is nan
    df_rr = df_rr.iloc[1:, :]
    df_tm = df_tm.iloc[1:, :]

    # convert index to datetime
    df_rr.index = pd.to_datetime(df_rr.index)
    df_tm.index = pd.to_datetime(df_tm.index)

    # only select summer months (JJA)
    df_rr = df_rr[(df_rr.index.month >= 6) & (df_rr.index.month <= 8)]
    df_tm = df_tm[(df_tm.index.month >= 6) & (df_tm.index.month <= 8)]

    # resample to daily data
    df_rr = df_rr.resample('D').max()
    df_tm = df_tm.resample('D').max()

    # only keep data with daily RR > 90th %ile
    # calculate percentile based on 1961-1990 data
    ref_data = df_rr[(df_rr.index >= '1961-01-01') & (df_rr.index <= '1990-12-31')]
    p90 = ref_data.quantile(0.90, axis=0)
    df_rr = df_rr[df_rr > p90]
    df_tm = df_tm[df_rr > p90]

    # only select data until end of 2023
    df_rr = df_rr[df_rr.index <= '2023-12-31']

    return df_rr, df_tm


def plot_scaling(fig, axs, rr, t2m):
    """
    Plot the apparent scaling of rainfall data against temperature data.

    Parameters:
        fig (plt.Figure): The figure object to plot on.
        axs (plt.Axes): The axes object to plot on.
        rr (pd.DataFrame): DataFrame containing daily rainfall data.
        t2m (pd.DataFrame): DataFrame containing daily temperature data.
    """
    dec_colors = ['#264653', '#2a9d8f', '#e9c46a', '#f4a261', '#e76f51']

    for idy, dy in enumerate(rr.index):
        if dy.year < 1991:
            col = dec_colors[0]
        elif 1991 <= dy.year < 2001:
            col = dec_colors[1]
        elif 2001 <= dy.year < 2011:
            col = dec_colors[2]
        elif 2011 <= dy.year < 2021:
            col = dec_colors[3]
        else:
            col = dec_colors[4]

        axs.plot(t2m.loc[dy, :], rr.loc[dy, :], 'o',
                 color=col)

    axs.set_xlim(5, 35)
    axs.set_ylim(0, 60)
    axs.set_xlabel('Tx [°C]', fontsize=12)
    axs.set_ylabel('RRx [mm]', fontsize=12)

    # create dummy points for legend
    labels = ['<= 1990', '1991-2000', '2001-2010', '2011-2020', '>= 2021']
    for i in range(len(dec_colors)):
        axs.plot([-10], [-10], 'o', color=dec_colors[i], label=labels[i])
    axs.legend()
    plt.show()

def run():
    rr, t2m = load_data()

    fig, axs = plt.subplots(1, 1, figsize=(5, 5))
    plot_scaling(fig=fig, axs=axs, rr=rr, t2m=t2m)


if __name__ == '__main__':
    run()
