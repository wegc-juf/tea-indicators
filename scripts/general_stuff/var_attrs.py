"""
script for adding attributes to TEA variables
"""

equal_vars = {'EM': 'tEX'}


def get_attrs(opts=None, vname=None, dec=False, spread=None, period='', data_unit=''):
    """
    get attributes for TEA variables
    Args:
        opts: command line options
        vname: variable name
        dec: decadal mean added to variable name
        spread: None, 'upper' or 'lower'. Set if spread estimator
        period: climatic time period
        data_unit: data unit (e.g. 'degC', 'mm')

    Returns:
        attributes: dict with attributes
    """
    if opts is not None:
        data_unit = opts.unit
        period = opts.period

    attrs = {'ctp': {'long_name': f'climatic time period ({period})'},
             'CTP': {'long_name': f'start date of climatic time period {period}', 'standard_name': 'ctp_time'},
             'CTP_global_attrs': {'title': f'TEA indicators for annual climatic time period: {period}'},
             'decadal': {'long_name': f'center year of decadal indicators for climatic time period {period}',},
             'decadal_global_attrs': {'title': f'TEA decadal-mean indicator variables for climatic time period: '
                                               f'{period}'},
             'amplification': {'long_name': 'center year of decadal amplification factors for climatic time period'
                                            f' {period}'},
             'amplification_global_attrs': {'title': f'TEA decadal-mean amplification factors for climatic time period:'
                                                     f' {period}'},
             'DTEC': {'long_name': 'daily threshold exceedance count', 'units': '1'},
             'DTEM': {'long_name': 'daily threshold exceedance magnitude', 'units': data_unit},
             'DTEA': {'long_name': 'daily threshold exceedance area', 'units': '100 km^2'},
             'DTEM_Max': {'long_name': 'daily maximum grid cell exceedance magnitude',
                          'units': data_unit},
             f'DTEEC': {'long_name': f'daily threshold exceedance event count', 'units': '1'},
             'EF': {'long_name': 'event frequency', 'units': '1', 'metric_type': 'basic'},
             'doy_first': {'long_name': 'day of first event occurrence', 'units': '1', 'metric_type': 'basic'},
             'doy_last': {'long_name': 'day of last event occurrence', 'units': '1', 'metric_type': 'basic'},
             'AEP': {'long_name': 'annual exposure period', 'units': 'dys', 'metric_type': 'basic'},
             'ED': {'long_name': 'cumulative events duration', 'units': f'{data_unit} dys', 'metric_type': 'compound'},
             'ED_avg': {'long_name': 'average events duration', 'units': 'dys', 'metric_type': 'basic'},
             'EM': {'long_name': 'cumulative exceedance magnitude', 'units': data_unit,
                    'description': 'expresses the temporal events extremity (tEX)', 'metric_type': 'compound'},
             'EM_avg': {'long_name': 'average exceedance magnitude', 'units': data_unit, 'metric_type': 'basic'},
             'EM_avg_Md': {'long_name': 'average daily-median exceedance magnitude',
                           'units': data_unit, 'metric_type': 'basic'},
             'EM_Md': {'long_name': 'cumulative daily-median exceedance magnitude',
                       'units': data_unit, 'metric_type': 'compound'},
             'EM_Max': {'long_name': 'cumulative maximum exceedance magnitude',
                        'units': data_unit, 'metric_type': 'compound'},
             'EM_avg_Max': {'long_name': 'average maximum exceedance magnitude',
                            'units': data_unit, 'metric_type': 'basic'},
             'EA_avg': {'long_name': 'average exceedance area', 'units': 'areals', 'metric_type': 'basic'},
             'DM': {'long_name': 'duration-magnitude indicator', 'units': f'{data_unit} dys', 'metric_type':
                    'compound'},
             'TEX': {'long_name': 'total events extremity', 'units': f'areal {data_unit} dys', 'metric_type':
                     'compound'},
             'ES_avg': {'long_name': 'average event severity',
                        'units': f'areal {data_unit} dys', 'metric_type': 'compound'},
             'tEX': {'long_name': 'temporal events extremity', 'units': f'{data_unit} dys', 'metric_type':
                     'compound'},
             'H_AEHC': {'long_name': 'cumulative atmospheric boundary layer exceedance '
                                     'heat content', 'units': 'PJ', 'metric_type': 'compound'}
             }

    # add (A)GR indicators if necessary
    if '_GR' in vname:
        vname_dict = vname.split('_GR')[0]
        vattrs = attrs[vname_dict]
        vattrs['long_name'] = f'{vattrs["long_name"]} (GR)'
    elif '_AGR' in vname:
        vname_dict = vname.split('_AGR')[0]
        vattrs = attrs[vname_dict]
        vattrs['long_name'] = f'{vattrs["long_name"]} (AGR)'
    else:
        if 'AF' in vname:
            vname_dict = vname.split('_AF')[0]
            vattrs = attrs[vname_dict]
        else:
            vattrs = attrs[vname]

    # add (CC) amplification if amplification factors are passed and set units to unity
    if 'AF' in vname and 'CC' not in vname:
        vattrs['long_name'] = f'{vattrs["long_name"]} amplification'
        vattrs['units'] = '1'
    elif 'AF_CC' in vname:
        vattrs['long_name'] = f'{vattrs["long_name"]} CC amplification'
        vattrs['units'] = '1'

    # add decadal-mean for decadal variables
    if dec:
        vattrs['long_name'] = f'decadal-mean {vattrs["long_name"]}'

    # add upper/lower spread estimator for spreads
    if spread:
        vattrs['long_name'] = f'{vattrs["long_name"]} {spread} spread estimator'

    return vattrs
