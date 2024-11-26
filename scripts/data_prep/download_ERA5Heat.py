"""
API request to download ERA5 Heat data -- NOT WORKING AT THE MOMENT!!!
"""

import cdsapi
from concurrent.futures import ThreadPoolExecutor, as_completed

def retrieve(client, request, year):
    print(f"requesting year: {year} /n")
    request.update({"year": year})
    client.retrieve(
        "derived-utci-historical", request, f"ERA5Heat_{year}.nc"
    ).download()
    return f"retrieved year: {year}"

def main(_request):
    """concurrent request using 10 threads"""
    client = cdsapi.Client()
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = [
            executor.submit(retrieve, client, _request.copy(), year) for year in YEARS
        ]
        for f in as_completed(futures):
            try:
                print(f.result())
            except:
                print("could not retrieve")

if __name__ == "__main__":
    YEARS = ["%d" % (y) for y in range(1961, 2024)]

    dataset = "derived-utci-historical"
    request = {
        "variable": [
            "universal_thermal_climate_index",
            "mean_radiant_temperature"
        ],
        "version": "1_1",
        "product_type": "consolidated_dataset",
        "year": "",
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

    main(request)
