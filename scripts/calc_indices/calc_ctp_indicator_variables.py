import pandas as pd

from combine_to_ds import *

def calc_event_frequency(pdata):
    """
    calculate event frequency (Eq. 11 & 12)
    Args:
        pdata: daily basis variables grouped into CTPs

    Returns:
        ef: event frequency
    """

    ef = pdata.sum('days').DTEEC
    ef_gr = pdata.sum('days').DTEEC_GR

    # combine to ds
    ef_ds = create_ef_ds(ef=ef, ef_gr=ef_gr)

    return ef_ds


def calc_supplementary_event_vars(data):
    """
    calculate supplementary event variables (Eq. 13)
    Args:
        data: daily basis variables grouped into CTPs

    Returns:
        svars: supplementary variables

    """

    doy = [pd.Timestamp(dy.values).day_of_year for dy in data.days]
    data.coords['doy'] = ('days', doy)

    # calculate dEfirst(_GR), dElast(_GR)
    doy_events_gr = data['doy'].where(data['DTEEC_GR'].notnull())
    doy_events = data['doy'].where(data['DTEEC'].notnull())
    doy_first, doy_first_gr = doy_events.groupby('ctp').min(), doy_events_gr.groupby('ctp').min()
    doy_last, doy_last_gr = doy_events.groupby('ctp').max(), doy_events_gr.groupby('ctp').max()

    # calculate annual exposure period
    delta_y = (doy_last - doy_first + 1) / 30.5
    delta_y_gr = (doy_last_gr - doy_first_gr + 1) / 30.5

    # combine to ds
    svars = create_svars_ds(doy_first=doy_first, doy_last=doy_last, doy_first_gr=doy_first_gr,
                            doy_last_gr=doy_last_gr, delta_y=delta_y, delta_y_gr=delta_y_gr)

    return svars


def calc_event_duration(pdata, ef):
    """
    calculate event duration (Eq. 14 & 15)
    Args:
        pdata: daily basis variables grouped into CTPs
        ef: event frequency

    Returns:
        ed: event duration
    """

    # calc cumulative events duration (Eq. 14_2 and 15_2)
    pdata_sum = pdata.sum('days')
    ed, ed_gr = pdata_sum['DTEC'], pdata_sum['DTEC_GR']

    # calc average event duration (Eq. 14_1 and 15_1)
    ed_avg = ed / ef['EF']
    ed_avg_gr = ed_gr / ef['EF_GR']

    # combine to ds
    ed_ds = create_ed_ds(ed=ed, ed_gr=ed_gr, ed_avg=ed_avg, ed_avg_gr=ed_avg_gr)

    return  ed_ds


def calc_exceedance_magnitude(opts, pdata, ed):
    """
    calculate event duration (Eq. 17 & 18)
    Args:
        opts: CLI parameter
        pdata: daily basis variables grouped into CTPs
        ed: event duration

    Returns:
        ed: event duration
    """

    # calc cumulative events magnitude (Eq. 17_2 and 18_2)
    pdata_sum = pdata.sum('days')
    em, em_gr = pdata_sum['DTEM'], pdata_sum['DTEM_GR']

    # calc average exceedance magnitude (Eq. 17_1 and 18_1)
    em_avg = em / ed['ED']
    em_avg_gr = em_gr / ed['ED_GR']

    # calc median exceedance magnitude (Eq. 19)
    pdata_med = pdata.median('days')
    em_avg_med, em_avg_gr_med = pdata_med['DTEM'], pdata_med['DTEM_GR']
    em_med = ed['ED'] * em_avg_med
    em_gr_med = ed['ED_GR'] * em_avg_gr_med

    # calc maximum exceedance magnitude (Eq. 20)
    pdata_max = pdata.max('days')
    em_gr_max = pdata_max['DTEM_Max']
    em_gr_avg_max = em_gr_max / ed['ED_GR']

    # combine to ds
    em_ds, em_suppl = create_em_ds(opts=opts, em=em, em_gr=em_gr, em_avg=em_avg, em_avg_gr=em_avg_gr,
                         em_avg_med=em_avg_med, em_avg_gr_med=em_avg_gr_med, em_med=em_med,
                         em_gr_med=em_gr_med, em_gr_max=em_gr_max, em_gr_avg_max=em_gr_avg_max)

    return  em_ds, em_suppl


def calc_exceedance_area_tex_sev(opts, data, ed, em):
    """
    calculate event area, event severity and TEX (Eq. 21)
    Args:
        opts: CLI parameter
        data: daily basis variables
        ed: event duration
        em: event magnitude

    Returns:

    """

    if opts.parameter == 'T':
        em_var, em_avg_var = 'EM_GR', 'EMavg_GR'
    else:
        em_var, em_avg_var = 'EM_GR_med', 'EMavg_GR_med'

    tex = (data['DTEM_GR'] * data['DTEA_GR']).groupby('ctp').sum('days')
    ea_gr = tex / em[em_var]

    es_gr = ed['EDavg_GR'] * em[em_avg_var] * ea_gr

    # combine to ds
    ea_ds = create_ea_ds(opts=opts, ea_gr=ea_gr, tex=tex, es_gr=es_gr)

    return ea_ds
