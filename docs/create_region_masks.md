# create_region_masks.py
Information about input data for `create_region_masks.py`

### shp file paths
EUR: `/data/arsclisys/backup/clim-hydro/TEA-Indicators/shapefiles/shapes_europe/CNTR_RG_01M_2020_4326.shp.zip` \
AUT: `/data/reloclim/backup/GEO/shapefiles/OEKS15/good/AUSTRIA.shp` \
AUT-Bundeslaender: `/data/reloclim/backup/GEO/shapefiles/OEKS15/good/LAND_AT_.shp` \
AUT-Gemeinden: `/data/reloclim/backup/GEO/shapefiles/OEKS15/good/GEM_AT_.shp` \
SAR: `/data/arsclisys/backup/clim-hydro/TEA-Indicators/shapefiles/SARext_shape.shp` \
FBR: `/data/arsclisys/backup/clim-hydro/TEA-Indicators/shapefiles/FBR_polygon.shp`

### EPSG IDs
SPARTACUS: 3416 \
INCA: 31287 \
WEGN*: 32633 \
ERA5: 4326 \
ERA5-Land: 4326 \
EOBS: 4326

*Note: to regrid to the extended WEGN grid, pass the same coordinate system as the 
WEGN data (EPSG:32633).

### Orography files
SPARTACUS: `/data/arsclisys/normal/clim-hydro/TEA-Indicators/SPARTACUS/SPARTACUSreg_orography.nc` 
(created with `regrid_SPARTACUS_toWEGNext.py`) \
ERA5: `/data/arsclisys/normal/clim-hydro/TEA-Indicators/hydro/ERA5/ERA5_orography.nc` (created with `prep_ERA5.py`) \
ERA5Land: `/data/arsclisys/normal/clim-hydro/TEA-Indicators/ERA5Land/ERA5Land_orography.nc` (created with 
`prep_ERA5Land.py`) \

### Land Sea Mask
ERA5: `/data/arsclisys/backup/clim-hydro/TEA-Indicators/ERA5/ERA5_LSM.nc`
