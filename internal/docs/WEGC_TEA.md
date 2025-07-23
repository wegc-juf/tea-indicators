# WEGC specific information about TEA indicators

## General information

All TEA related input data stored in the `/data/arsclisys/normal/clim-hydro/TEA-Indicators/`
directory. Henceforth, refered to as `<TEA-path>`.

## CFG files

The CFG files used for the TEA publication (Kirchengast et al. 2025) and some exemplary CFG files
for regions in Africa are stored in `<TEA-path>/CFG-files/TEA-paper-data/` and
`<TEA-path>/CFG-files/Africa/`.

## Input data (SPARTACUS, ERA5, ERA5-Land)

Preprocessed SPARTACUSv2 (and v1.5), ERA5, and ERA5-Land data are stored in the following directories:

- `<TEA-path>/SPARTACUS/` (output of `regrid_SPARTACUS_to_WEGNext.py`)
- `<TEA-path>/ERA5/` (output of `prep_ERA5.py`)
- `<TEA-path>/ERA5Land/` (output of `prep_ERA5Land.py`)

**Note** that the SPARTACUS data is regridded to the WEGNext grid (EPSG:32633).

Data of selected GeoSphere stations (used for the calculation of the natural variability) are stored
in `<TEA-path>/station_data/`.

## Static input data

Static input data (masks, threshold grids, etc.) produced by `calc_TEA.py` are stored in
`<TEA-path>/static/`.

## Shapefiles

Shapefiles are stored in `<TEA-path>/shapefiles/`. The following shapefiles are used:

- `GLOBAL.shp`: Global shapefile containing all countries.
- `AUSTRIA.shp`: Shapefile of Austria.
- `AUSTRIA_Bundeslaender.shp`: Shapefile of Austrian Bundeslaender.
- `AUSTRIA_Gemeinden.shp`: Shapefile of Austrian Gemeinden.
- `SAR.shp`: Shapefile of Southeast Austria Region (SAR) region.
- `FBR.shp`: Shapefile of the Feldbach (FBR) region.

## Orography files and Land Sea Masks

Orography files and Land Sea Masks for SPARTACUS, ERA5, and ERA5-Land are stored in the
corresponding dataset's directory.

## EPSG IDs (relevant for the creation of region masks)

SPARTACUS: 3416 \
ERA5: 4326 \
ERA5-Land: 4326 \
WEGN: 32633 \
EOBS: 4326 \
INCA: 31287 

**Note** that to for regridded SPARTACUS data, the same coordinate system as the WEGN data 
(EPSG:32633) has to be passed in `create_region_masks.py`.