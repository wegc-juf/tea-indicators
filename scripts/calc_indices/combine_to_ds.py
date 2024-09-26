import xarray as xr


def create_ef_ds(ef, ef_gr):
    """
    combine EF variables to ds
    Args:
        ef: event frequency da
        ef_gr: event frequency (GR) da

    Returns:
        ef_ds: EF ds
    """
    ef = ef.rename('EF')
    ef = ef.assign_attrs({'long_name': 'event frequency', 'units': '1'})

    ef_gr = ef_gr.rename('EF_GR')
    ef_gr.attrs = {'long_name': 'event frequency (GR)', 'units': '1'}

    ef_ds = xr.merge([ef, ef_gr])

    return ef_ds


def create_svars_ds(doy_first, doy_first_gr, doy_last, doy_last_gr, delta_y, delta_y_gr):
    """
    combine supplementary variables to ds
    Args:
        doy_first: DOY of first event occurrence da
        doy_first_gr: DOY of first event occurrence (GR) da
        doy_last: DOY of last event occurrence da
        doy_last_gr: DOY of last event occurrence (GR) da
        delta_y: annual exposure period
        delta_y_gr: annual exposure period (GR)

    Returns:
        svars_ds: ds with supplementary variables
    """
    doy_first = doy_first.rename(f'doy_first')
    doy_first = doy_first.assign_attrs({'long_name': 'day of first event occurrence', 'units': '1'})
    doy_first_gr = doy_first_gr.rename(f'doy_first_GR')
    doy_first_gr = doy_first_gr.assign_attrs({'long_name': 'day of first event occurrence (GR)',
                                              'units': '1'})

    doy_last = doy_last.rename(f'doy_last')
    doy_last = doy_last.assign_attrs({'long_name': 'day of last event occurrence', 'units': '1'})
    doy_last_gr = doy_last_gr.rename(f'doy_last_GR')
    doy_last_gr = doy_last_gr.assign_attrs({'long_name': 'day of last event occurrence (GR)',
                                            'units': '1'})

    delta_y = delta_y.rename(f'delta_y')
    delta_y = delta_y.assign_attrs({'long_name': 'annual exposure period', 'units': 'dys'})
    delta_y_gr = delta_y_gr.rename(f'delta_y_GR')
    delta_y_gr = delta_y_gr.assign_attrs({'long_name': 'annual exposure period (GR)',
                                          'units': 'dys'})

    svars_ds = xr.merge([doy_first, doy_last, delta_y, doy_first_gr, doy_last_gr, delta_y_gr])

    return svars_ds


def create_ed_ds(ed, ed_gr, ed_avg, ed_avg_gr):
    """
    combine ED variables to ds
    Args:
        ed: cumulative event duration da
        ed_gr: cumulative event duration (GR) da
        ed_avg: average event duration da
        ed_avg_gr: average event duration (GR) da

    Returns:
        ed_ds: ED ds

    """

    ed = ed.rename('ED')
    ed = ed.assign_attrs({'long_name': 'cumulative events duration', 'units': 'dys'})
    ed_gr = ed_gr.rename('ED_GR')
    ed_gr.attrs = {'long_name': 'cumulative events duration (GR)', 'units': 'dys'}
    ed_avg = ed_avg.rename('EDavg')
    ed_avg.attrs = {'long_name': 'average events duration', 'units': 'dys'}
    ed_avg_gr = ed_avg_gr.rename('EDavg_GR')
    ed_avg_gr.attrs = {'long_name': 'average events duration (GR)', 'units': 'dys'}

    ed_ds = xr.merge([ed, ed_gr, ed_avg, ed_avg_gr])

    return ed_ds


def create_em_ds(opts, em, em_gr, em_avg, em_avg_gr, em_avg_med, em_avg_gr_med, em_med, em_gr_med,
                 em_gr_max, em_gr_avg_max):
    """
    combine EM variables to ds
    Args:
        opts: CLI parameter
        em: cumulative exceedance magnitude da
        em_gr: cumulative exceedance magnitude (GR) da
        em_avg: average exceedance magnitude da
        em_avg_gr: average exceedance magnitude (GR) da
        em_avg_med: average daily-median exceedance magnitude da
        em_avg_gr_med: average daily-median exceedance magnitude (GR) da
        em_med: daily-median exceedance magnitude da
        em_gr_med: daily-median exceedance magnitude (GR) da
        em_gr_max: cumulative maximum grid cell exceedance magnitude (GR) da
        em_gr_avg_max: average maximum grid cell exceedance magnitude (GR) da

    Returns:
        em_ds: EM ds
        em_ds_suppl: supplementary EM vars ds

    """

    em = em.rename('EM')
    em.attrs = {'long_name': 'cumulative exceedance magnitude', 'units': opts.unit,
                'description': 'expresses the temporal events extremity (tEX)'}

    em_gr = em_gr.rename('EM_GR')
    em_gr.attrs = {'long_name': 'cumulative exceedance magnitude (GR)', 'units': opts.unit,
                   'description': 'expresses the temporal events extremity (tEX_GR)'}

    em_avg = em_avg.rename('EMavg')
    em_avg.attrs = {'long_name': 'average exceedance magnitude', 'units': opts.unit}

    em_avg_gr = em_avg_gr.rename('EMavg_GR')
    em_avg_gr.attrs = {'long_name': 'average exceedance magnitude (GR)', 'units': opts.unit}

    em_avg_med = em_avg_med.rename('EMavg_Md')
    em_avg_med.attrs = {'long_name': 'average daily-median  exceedance magnitude',
                        'units': opts.unit}

    em_avg_gr_med = em_avg_gr_med.rename('EMavg_Md_GR')
    em_avg_gr_med.attrs = {'long_name': 'average daily-median exceedance magnitude (GR)',
                           'units': opts.unit}

    em_med = em_med.rename('EM_Md')
    em_med.attrs = {'long_name': 'cumulative daily-median exceedance magnitude', 'units': opts.unit}

    em_gr_med = em_gr_med.rename('EM_Md_GR')
    em_gr_med.attrs = {'long_name': 'cumulative daily-median exceedance magnitude (GR)',
                       'units': opts.unit}

    em_gr_max = em_gr_max.rename('EM_Max_GR')
    em_gr_max.attrs = {'long_name': 'cumulative maximum exceedance magnitude (GR)',
                       'units': opts.unit}

    em_gr_avg_max = em_gr_avg_max.rename('EMavg_Max_GR')
    em_gr_avg_max.attrs = {'long_name': 'average maximum exceedance magnitude (GR)',
                           'units': opts.unit}


    if opts.parameter == 'T':
        em_ds = xr.merge([em, em_gr, em_avg, em_avg_gr])
        em_ds_suppl = xr.merge([em_avg_med, em_avg_gr_med, em_med, em_gr_med, em_gr_max,
                                em_gr_avg_max])
    else:
        em_ds = xr.merge([em_med, em_gr_med, em_avg_med, em_avg_gr_med])
        em_ds_suppl = xr.merge([em, em_gr, em_avg, em_avg_gr, em_gr_max, em_gr_avg_max])

    return em_ds, em_ds_suppl


def create_ea_ds(opts, ea_gr, tex, es_gr):
    """
    combine EA variables
    Args:
        opts: CLI parameter
        ea_gr: exceedance area GR da
        tex: TEX da
        es_gr: event severity da

    Returns:
        ea_ds: ds
    """

    if opts.parameter:
        unit = 'areal °C dys'
    else:
        unit = 'areal mm dys'

    ea_gr = ea_gr.rename('EAavg_GR')
    ea_gr.attrs = {'long_name': 'average exceedance area (GR)', 'units': 'areals'}

    tex = tex.rename('TEX_GR')
    tex.attrs = {'long_name': 'total events extremity (GR)', 'units': unit}

    es_gr = es_gr.rename('ESavg_GR')
    es_gr.attrs = {'long_name': 'average event severity (GR)', 'units': unit}

    ea_ds = xr.merge([ea_gr, tex, es_gr])

    return ea_ds