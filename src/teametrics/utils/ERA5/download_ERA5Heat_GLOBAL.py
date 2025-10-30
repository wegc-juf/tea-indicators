"""
API request to download ERA5 Heat data
"""
import cdsapi

if __name__ == "__main__":
    YEARS = ["%d" % y for y in range(2003, 2025)]

    for yr in YEARS:
        request = {
            "variable": ["universal_thermal_climate_index"],
            "version": "1_1",
            "product_type": "consolidated_dataset",
            "year": [yr],
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
            ]
        }

        client = cdsapi.Client()
        client.retrieve("derived-utci-historical", request,
                        f'/data/arsclisys/normal/clim-hydro/TEA-Indicators/'
                        f'ERA5Heat_GLO/raw/ERA5Heat_{yr}.zip')