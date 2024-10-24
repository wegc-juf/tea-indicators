# HOW TO: TEA-Indicators

## 1) Preparation of input datasets
To prepare the input datasets (SPARTACUS, ERA5, and ERA5-Land) run the following data prep scripts  
(order doesn't matter).

- `regrid_SPARTACUS_to_WEGNext.py` - for regridding SPARTACUS to a regular 1 km x 1 km grid which is 
congruent with the 1 km x 1 km WEGN grid within FBR. Attention: run twice, once for regular data 
and once for orography data.
- `prep_ERA5.py` - for preparing ERA5 data (aggregates hourly data to daily data).
- `prep_ERA5Land.py` - for preparing ERA5-Land data (aggregates hourly data to daily data).

## 2) Creation of background files
To create the necessary background files run the following scripts:
1. `create_region_masks.py`\
   (For WEGC users: input data filepaths are listed in `create_region_masks.md`) 
2. `create_static_files.py`

## 3) TEA Indicators
After preparing all the necessary input and background data, run `calc_TEA.py`.