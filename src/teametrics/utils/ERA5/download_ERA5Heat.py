"""
API request to download ERA5 Heat data
"""
import cdsapi

if __name__ == "__main__":
    YEARS = ["%d" % y for y in range(1967, 2025)]

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
            ],
            "area": [72, -30, 30, 45]
        }

        client = cdsapi.Client()
        client.retrieve("derived-utci-historical", request,
                        f'ERA5Heat_{yr}.nc')

# original request from CDS ---------------------------------------------------
# import cdsapi
#
# dataset = "derived-utci-historical"
# request = {
#     "variable": ["universal_thermal_climate_index"],
#     "version": "1_1",
#     "product_type": "consolidated_dataset",
#     "year": ["1961"],
#     "month": [
#         "01", "02", "03",
#         "04", "05", "06",
#         "07", "08", "09",
#         "10", "11", "12"
#     ],
#     "day": [
#         "01", "02", "03",
#         "04", "05", "06",
#         "07", "08", "09",
#         "10", "11", "12",
#         "13", "14", "15",
#         "16", "17", "18",
#         "19", "20", "21",
#         "22", "23", "24",
#         "25", "26", "27",
#         "28", "29", "30",
#         "31"
#     ],
#     "area": [72, -30, 30, 45]
# }
#
# client = cdsapi.Client()
# client.retrieve(dataset, request).download()