# teametrics

A python package to calculate threshold-exceedance-amount (TEA) indicators 
    as described in https://doi.org/10.48550/arXiv.2504.18964.

## Installation
Start by creating a new `virtualenv` for your project
```bash
mkvirtualenv <project_name>
```

Then, install the package using `pip`:
```bash
pip install teametrics-{{VERSION}}-py3-none-any.whl
```

## Changelog

See [Tags](https://wegcgitlab.uni-graz.at/hst/tea-indicators/-/tags) for a detailed changelog.

## Usage

### 1) Download of input datasets
To calculate TEA indicators, you need to download and prepare your input data first.

Download input datasets (ERA5, ERA5-Land, or SPARTACUS) if necessary. For ERA5 data, you can use the following scripts
as a starting point:
- [`download_ERA5.py`](https://wegcgitlab.uni-graz.at/hst/tea-indicators/-/blob/main/src/teametrics/utils/ERA5/download_ERA5.py) - for downloading ERA5 
  data from the Copernicus Climate Data Store (CDS).
- [`download_ERA5-Land.py`](https://wegcgitlab.uni-graz.at/hst/tea-indicators/-/blob/main/src/teametrics/utils/ERA5/download_ERA5-Land.py) - for downloading 
  ERA5-Land data from the Copernicus Climate Data Store (CDS).

### 2) Preparation of input datasets
For calculation of daily TEA indicators, the input data must be aggregated to daily data.
To prepare the input datasets (ERA5, ERA5-Land, or SPARTACUS) run one of the following data prep scripts:

- `prep_ERA5 --inpath INPATH --outpath OUTPATH` -- for preparing ERA5 data (aggregates hourly data to
  daily data).
- `prep_ERA5Land --inpath INPATH --outpath OUTPATH --orog-file PATH_TO_OROG_FILE` -- for preparing 
  ERA5-Land data
  (aggregates hourly data to daily data).
- `regrid_SPARTACUS_to_WEGNext --config-file CONFIG_FILE` -- only needed for SPARTACUS data for regridding 
  SPARTACUS to a regular 1 km x 1 km
  grid which is congruent with the 1 km x 1 km WEGN grid within FBR. Attention: run twice, once for regular data
  and once for orography data.

### 3) Creation of mask file (optional)
In case you want do define your own GeoRegion (GR) mask, you can create a mask file using the script
`create_region_masks --config-file CONFIG_FILE`\
This script allows you to create a mask file for your GR based on a shapefile or coordinates.
The configuration options for the script are documented in [`CFG-PARAMS-doc.md`](https://wegcgitlab.uni-graz.at/hst/tea-indicators/-/blob/main/docs/CFG-PARAMS-doc.md) \
(For WEGC users: input data filepaths are listed in [`create_region_masks.md`](https://wegcgitlab.uni-graz.at/hst/tea-indicators/-/blob/main/docs/create_region_masks.md))

### 4) Calculation of TEA Indicators
After preparing all the necessary input and mask data, run `calc_tea --config-file CONFIG_FILE`.

A minimal example config can be found in [`TEA_CFG_minimal.yaml`](https://wegcgitlab.uni-graz.at/hst/tea-indicators/-/blob/main/src/teametrics/config/TEA_CFG_minimal.yaml).
Template config files are [`TEA_CFG_template.yaml`](https://wegcgitlab.uni-graz.at/hst/tea-indicators/-/blob/main/src/teametrics/config/TEA_CFG_template.yaml) for
gridded data and [`TEA_CFG_template_station.yaml`](https://wegcgitlab.uni-graz.at/hst/tea-indicators/-/blob/main/src/teametrics/config/TEA_CFG_template_station.yaml) for station data. \
The configuration options for the script are documented in [`CFG-PARAMS-doc.md`](https://wegcgitlab.uni-graz.at/hst/tea-indicators/-/blob/main/docs/CFG-PARAMS-doc.md).

### 5) Using the TEA Indicator classes TEAIndicators and TEAAgr (optional)
In case you want a more fine-grained control over the TEA Indicator calculations, you can use the classes
`teametrics.TEA.TEAIndicators` for normal GeoRegions, and \
`teametrics.TEA_AGR.TEAAgr` for Aggregated GeoRegions.

A simple use example can be found in the script [`tea_example`](https://wegcgitlab.uni-graz.at/hst/tea-indicators/-/blob/main/src/teametrics/TEA_example.py). \
Source code documentation for the classes can be found in TODO.

## Support
Just open an issue on the [GitHub repository](https://wegcgitlab.uni-graz.at/hst/tea-indicators/) or contact the authors directly.

## Authors 
- **Stephanie Haas** — Developer, Maintainer\
  Wegener Center for Climate and Global Change, University of Graz, Graz, Austria
  (stephanie.haas@uni-graz.at)
- **Jürgen Fuchsberger** — Developer, Maintainer\
  Wegener Center for Climate and Global Change, University of Graz, Graz, Austria
  (juergen.fuchsberger@uni-graz.at)
- **Gottfried Kirchengast** — Project Lead, main concept\
  Wegener Center for Climate and Global Change, University of Graz, Graz, Austria
  (gottfried.kirchengast@uni-graz.at)

## License
Gnu General Public License v3.0 (GPL-3.0)

## Contributing
Always welcome - just get in touch with the project developers.

