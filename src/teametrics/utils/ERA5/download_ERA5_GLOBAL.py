import cdsapi

YEARS = ["%d" % y for y in range(1969, 2025)]

dataset = "derived-era5-single-levels-daily-statistics"
for yr in YEARS:
    request = {
        "product_type": "reanalysis",
        "variable": ["2m_temperature"],
        "year": yr,
        "month": [
            "01", "02", "03",
            "04", "05", "06",
            "07", "08", "09",
            "10", "11", "12"
        ],
        "day": [
            "01", "02", "03",
            "04", "05", "06",
            "07", "08", "09",
            "10", "11", "12",
            "13", "14", "15",
            "16", "17", "18",
            "19", "20", "21",
            "22", "23", "24",
            "25", "26", "27",
            "28", "29", "30",
            "31"
        ],
        "daily_statistic": "daily_maximum",
        "time_zone": "utc+00:00",
        "frequency": "1_hourly"
    }

    client = cdsapi.Client()
    client.retrieve(dataset, request,
                    f'/data/arsclisys/normal/clim-hydro/TEA-Indicators/ERA5_GLO/'
                    f'ERA5_{yr}.nc')
