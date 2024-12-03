from datetime import timedelta
import pandas as pd

def validate_period(opts):
    valid_dec_periods = ['annual', 'WAS', 'ESS', 'JJA']
    if opts.decadal and opts.period not in valid_dec_periods:
        raise AttributeError(f'For decadal output, please select from {valid_dec_periods} as '
                             f'period! {opts.period} was passed instead.')

    if opts.decadal or opts.decadal_only:
        if opts.end - opts.start < 9:
            raise AttributeError(f'For decadal output, please pass more at least 10 years! '
                                 f'{(opts.end - opts.start) + 1} years were passed instead.')


def assign_ctp_coords(opts, data):
    """
    create dictionary of all start & end dates, the chosen frequency and period
    Args:
        opts: CLI parameter
        data: data array

    Returns:

    """

    freqs = {'annual': 'AS', 'seasonal': '3MS', 'WAS': 'AS-APR', 'ESS': 'AS-MAY', 'JJA': 'AS-JUN',
             'monthly': 'MS'}
    freq = freqs[opts.period]

    pstarts = pd.date_range(data.days[0].values, data.days[-1].values,
                            freq=freq).to_series()
    if opts.period == 'WAS':
        pends = pd.date_range(data.days[0].values, data.days[-1].values,
                              freq='A-OCT').to_series()
    elif opts.period == 'ESS':
        pends = pd.date_range(data.days[0].values, data.days[-1].values,
                              freq='A-SEP').to_series()
    else:
        pends = pstarts - timedelta(days=1)
        pends[0:-1] = pends[1:]
        pends.iloc[-1] = data.days[-1].values

    # add ctp as coordinates to enable using groupby later
    # map the 'days' coordinate to 'ctp'
    def map_to_ctp(dy, starts, ends):
        for start, end, ctp in zip(starts, ends, starts):
            if start <= dy <= end:
                return ctp
        return np.nan

    days_to_ctp = []
    for day in data.days.values:
        ctp_dy = map_to_ctp(dy=day, starts=pstarts, ends=pends)
        days_to_ctp.append(ctp_dy)

    data.coords['ctp'] = ('days', days_to_ctp)

    # group into CTPs
    data_per = data.groupby('ctp')

    return data, data_per