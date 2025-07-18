# HOW TO: TEA-Indicators

## 1) Download of input datasets
Download input datasets (ERA5, ERA5-Land, or SPARTACUS) if necessary. For ERA5 data, you can use the following scripts 
as a starting point:
- `download_ERA5.py` - for downloading ERA5 data from the Copernicus Climate Data Store (CDS).
- `download_ERA5-Land.py` - for downloading ERA5-Land data from the Copernicus Climate Data Store (CDS).

## 2) Preparation of input datasets
To prepare the input datasets (ERA5, ERA5-Land, or SPARTACUS) run the following data prep scripts  
(order doesn't matter).

- `prep_ERA5 --inpath <input_path> --outpath <output_path>` -- for preparing ERA5 data (aggregates hourly data to 
  daily data).
- `prep_ERA5Land --inpath INPATH --outpath OUTPATH --orog-file OROG` -- for preparing ERA5-Land data 
  (aggregates hourly data to daily data).
- `regrid_SPARTACUS_to_WEGNext` -- only needed for SPARTACUS data for regridding SPARTACUS to a regular 1 km x 1 km 
  grid which is congruent with the 1 km x 1 km WEGN grid within FBR. Attention: run twice, once for regular data
  and once for orography data.

## 3) Creation of mask file (optional)
In case you want do define your own GeoRegion (GR) mask, you can create a mask file using the script
`create_region_masks --config-file CONFIG_FILE`\
This script allows you to create a mask file for your GR based on a shapefile or coordinates. 
The configuration options for the script are documented in `CFG-PARAMS-doc.md` \
   (For WEGC users: input data filepaths are listed in `create_region_masks.md`) 

## 4) Calculation of TEA Indicators
After preparing all the necessary input and mask data, run `calc_TEA.py --config-file CONFIG_FILE`. \
A minimal example config can be found in `TEA_CFG_minimal.yaml`. Template config files are `TEA_CFG_template.yaml` for 
gridded data and `TEA_CFG_template_station.yaml` for station data. \
The configuration options for the script are documented in `CFG-PARAMS-doc.md`

## 5) Using the TEA Indicator classes TEAIndicators and TEAAgr (optional)
In case you want a more fine-grained control over the TEA Indicator calculations, you can use the classes 
`TEAIndicators` for normal GeoRegions and `TEAAgr` for Aggregated GeoRegions. A simple use example can be found in 
the script `TEA_example.py`. \
Source code documentation for the classes can be found in TODO.
