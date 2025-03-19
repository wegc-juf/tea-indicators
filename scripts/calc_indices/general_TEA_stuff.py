from datetime import timedelta
import numpy as np
import pandas as pd
import numpy as np


def assign_ctp_coords(opts, data):
    """
    create dictionary of all start & end dates, the chosen frequency and period
    Args:
        opts: CLI parameter
        data: data array

    Returns:

    """
    
    pd_major, pd_minor = pd.__version__.split('.')[:2]
    if int(pd_major) >= 2 or int(pd_minor) >= 2:
        freqs = {'annual': 'YS', 'seasonal': '3MS', 'WAS': 'YS-APR', 'ESS': 'YS-MAY', 'JJA': 'YS-JUN',
                 'monthly': 'MS', 'EWS': 'YS-NOV'}
    else:
        freqs = {'annual': 'AS', 'seasonal': '3MS', 'WAS': 'AS-APR', 'ESS': 'AS-MAY', 'JJA': 'AS-JUN',
                 'monthly': 'MS', 'EWS': 'AS-NOV'}

    freq = freqs[opts.period]

    pstarts = pd.date_range(data.time[0].values, data.time[-1].values,
                            freq=freq).to_series()
    if opts.period == 'WAS':
        pends = pd.date_range(data.time[0].values, data.time[-1].values,
                              freq='A-OCT').to_series()
    elif opts.period == 'ESS':
        pends = pd.date_range(data.time[0].values, data.time[-1].values,
                              freq='A-SEP').to_series()
    else:
        pends = pstarts - timedelta(days=1)
        pends[0:-1] = pends[1:]
        pends.iloc[-1] = data.time[-1].values

    # add ctp as coordinates to enable using groupby later
    # map the 'time' coordinate to 'ctp'
    def map_to_ctp(dy, starts, ends):
        for start, end, ctp in zip(starts, ends, starts):
            if start <= dy <= end:
                return ctp
        return np.nan

    days_to_ctp = []
    for day in data.time.values:
        ctp_dy = map_to_ctp(dy=day, starts=pstarts, ends=pends)
        days_to_ctp.append(ctp_dy)

    data.coords['ctp'] = ('time', days_to_ctp)

    # group into CTPs
    data_per = data.groupby('ctp')

    return data, data_per