import cdsapi

if __name__ == "__main__":
    # YEARS = ["%d" % (y) for y in range(1979, 2022)]
    YEARS = ['2023', '2024']
    VARS = ['10m_u_component_of_wind',
            '10m_v_component_of_wind',
            '2m_dewpoint_temperature',
            '2m_temperature',
            'geopotential',
            'mean_sea_level_pressure',
            'surface_pressure',
            'total_precipitation']

    for yr in YEARS:
        for var in VARS:
            request = {
                'product_type': 'reanalysis',
                'variable': [var],
                'year': f'{yr}',
                'month': ["{:02d}".format(dd) for dd in range(1, 13)],
                'day': ["{:02d}".format(dd) for dd in range(1, 32)],
                'time': [
                    '00:00', '01:00', '02:00',
                    '03:00', '04:00', '05:00',
                    '06:00', '07:00', '08:00',
                    '09:00', '10:00', '11:00',
                    '12:00', '13:00', '14:00',
                    '15:00', '16:00', '17:00',
                    '18:00', '19:00', '20:00',
                    '21:00', '22:00', '23:00',
                ],
                'area': [
                    72, -30, 30,
                    45,
                ],
                'format': 'netcdf',
            }

            client = cdsapi.Client()
            client.retrieve('reanalysis-era5-single-levels', request,
                            f'ERA5_{yr}_{var}.nc')
