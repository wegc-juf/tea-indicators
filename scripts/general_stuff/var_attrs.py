"""
script for adding attributes to TEA variables
"""


def get_attrs(opts=None, vname=None, dec=False, spread=None):
    if opts is None:
        data_unit = ''
        period = ''
    else:
        data_unit = opts.unit
        period = opts.period

    attrs = {'ctp': {'long_name': f'climatic time period ({period})'},
             'DTEM': {'long_name': 'daily threshold exceedance magnitude', 'units': data_unit},
             'DTEM_Max': {'long_name': 'daily maximum grid cell exceedance magnitude',
                          'units': data_unit},
             f'DTEEC': {'long_name': f'daily threshold exceedance event count', 'units': '1'},
             'EF': {'long_name': 'event frequency', 'units': '1'},
             'doy_first': {'long_name': 'day of first event occurrence', 'units': '1'},
             'doy_last': {'long_name': 'day of last event occurrence', 'units': '1'},
             'delta_y': {'long_name': 'annual exposure period', 'units': 'dys'},
             'ED': {'long_name': 'cumulative events duration', 'units': 'dys'},
             'DM': {'long_name': 'duration-magnitude indicator', 'units': f'{data_unit} dys'},
             'EDavg': {'long_name': 'average events duration', 'units': 'dys'},
             'EM': {'long_name': 'cumulative exceedance magnitude', 'units': data_unit,
                    'description': 'expresses the temporal events extremity (tEX)'},
             'EMavg': {'long_name': 'average exceedance magnitude', 'units': data_unit},
             'EMavg_Md': {'long_name': 'average daily-median exceedance magnitude',
                          'units': data_unit},
             'EM_Md': {'long_name': 'cumulative daily-median exceedance magnitude',
                       'units': data_unit},
             'EM_Max': {'long_name': 'cumulative maximum exceedance magnitude',
                        'units': data_unit},
             'EMavg_Max': {'long_name': 'average maximum exceedance magnitude',
                              'units': data_unit},
             'EAavg': {'long_name': 'average exceedance area', 'units': 'areals'},
             'TEX': {'long_name': 'total events extremity', 'units': f'areal {data_unit} dys'},
             'ESavg': {'long_name': 'average event severity',
                       'units': f'areal {data_unit} dys'},
             'tEX': {'long_name': 'temporal events extremity', 'units': f'areal {data_unit} dys'},
             'H_AEHC': {'long_name': 'cumulative atmospheric boundary layer exceedance '
                                     'heat content', 'units': 'PJ'}
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
