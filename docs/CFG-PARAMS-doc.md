# Documentation of TEA CFG parameter

## Common parameter (used in all scripts)
| NAME             | DESCRIPTION                                                                                                                      | TYPE  | DEFAULT                                                   |
|------------------|----------------------------------------------------------------------------------------------------------------------------------|-------|-----------------------------------------------------------|
|                  |                                                                                                                                  |       |                                                           |
| *region*         | Name of GeoRegion; AUT, SAR, SEA, FBR, name of Austrian state, EUR, or ISO2 country code.                                        | str   | AUT                                                       |
| *gr_type*        | Method to define GR; polygon, corners, or center.                                                                                | str   | polygon                                                   |
| *sw_corner*      | Only if *gr_type* corners, southwest corner of GR; lon,lat or x,y separated by ",".                                              | x,y   | null                                                      |
| *ne_corner*      | Only if *gr_type* corners, northeast corner of GR; lon,lat or x,y separated by ",".                                              | x,y   | null                                                      |
| *center*         | Only if *gr_type* center, center of GR; lon,lat or x,y separated by ",".                                                         | x,y   | null                                                      |
| *we_len*         | Only if *gr_type* center,length of GR west to east.                                                                              | float | null                                                      |
| *ns_len*         | Only: if *gr_type* center,length of GR north to south.                                                                           | float | null                                                      |
|                  |                                                                                                                                  |       |                                                           |
| *parameter*      | Name of parameter for TEA calculation .                                                                                          | str   | Tx                                                        |
| *precip*         | Marks if precipitation data is used; set if input is precipitation data.                                                         | bool  | false                                                     |
| *threshold*      | Threshold value; if percentiles are used as thresholds, *theshold* defines the percentile, otherwise it is the absolute value    | float | 99                                                        |
| *threshold_type* | Type of threshold; abs for absolute thresholds, perc for percentiles.                                                            | str   | perc                                                      |
| *unit*           | Physical unit of chosen parameter.                                                                                               | str   | degC                                                      |
| *low_extreme*    | Marks if a low extreme is investigated; set if low extreme is investigated (values lower than threshold are considered extreme). | bool  | false                                                     |
|                  |                                                                                                                                  |       |                                                           |
| *start*          | Start year.                                                                                                                      | int   | 1961                                                      |
| *end*            | End year.                                                                                                                        | int   | current year                                              |
| *period*         | Climatic time period of interest; monthly, seasonal, WAS, ESS, JJA, or annual.                                                   | str   | WAS                                                       |
| *dataset*        | Name of dataset; SPARTACUS, ERA5, or ERA5Land.                                                                                   | str   | SPARTACUS                                                 |
|                  |                                                                                                                                  |       |                                                           |
| *maskpath*       | Path of mask directory.                                                                                                          | path  | /data/arsclisys/normal/clim-hydro/TEA-Indicators/masks/   |
| *statpath*       | Path of static file directory.                                                                                                   | path  | /data/arsclisys/normal/clim-hydro/TEA-Indicators/static/  |
| *tmppath*        | Path of temporary directory. Only relevant if large GR (> 100 areals) are processed with ERA5(-Land) data.                       | path  | /home/hst/tmp_data/TEAclean/largeGR/                      |

## regrid_SPARTACUS_to_WEGNext
| NAME        | DESCRIPTION                             | TYPE | DEFAULT                                                                    |
|-------------|-----------------------------------------|------|----------------------------------------------------------------------------|
| *inpath*    | Path of input data.                     | path | /data/arsclisys/normal/clim-hydro/TEA-Indicators/SPARTACUS_raw/v2024_v1.5/ |
| *orography* | Marks if orography should be regridded. | bool | false                                                                      |
| *orofile*   | Path of orography file.                 | path | /data/reloclim/backup/ZAMG_INCA/data/original/INCA_orog_corrected_y_dim.nc |
| *wegnfile*  | Dummy WEGN file to extract grid.        | path | /data/users/hst/cdrDPS/wegnet/WN_L2_DD_v7_UTM_TF1_UTC_2020-08.nc           |
| *outpath*   | Path of output directory.               | path | /data/arsclisys/normal/clim-hydro/TEA-Indicators/SPARTACUS/                |


## create_region_masks
| NAME         | DESCRIPTION                                                                                                                                                                                                                    | TYPE | DEFAULT                                                                                |
|--------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|------|----------------------------------------------------------------------------------------|
| *subreg*     | Only necessary if selected region is not the entire region in the shp file (Austrian states, european countries etc.). In case of Austrian states, give name of state. In case of european country, give ISO2 code of country. | str  | null                                                                                   |
| *target_sys* | ID of wanted coordinate System (https://epsg.io) which should be used for mask.                                                                                                                                                | int  | 3416                                                                                   |
| *target_ds*  | Dataset for which mask should be created.                                                                                                                                                                                      | str  | SPARTACUS                                                                              |
| *xy_name*    | Names of x and y coordinates in testfile, separated by ",".                                                                                                                                                                    | path | x,y                                                                                    |
| *shpfile*    | Shape file of region.                                                                                                                                                                                                          | path | /data/reloclim/backup/GEO/shapefiles/OEKS15/good/AUSTRIA.shp                           |
| *testfile*   | File with coordinate information of target grid.                                                                                                                                                                               | path | /data/arsclisys/normal/clim-hydro/TEA-Indicators/SPARTACUS/SPARTACUS-DAILY_Tx_1961.nc  |
| *orofile*    | File with orography information of target grid.                                                                                                                                                                                | path | /data/arsclisys/normal/clim-hydro/TEA-Indicators/SPARTACUS/ SPARTACUSreg_orography.nc  |
| *lsmfile*    | Only necessary if mask for EUR should be created. File with land-sea-mask of target grid.                                                                                                                                      | path | /data/users/hst/cdrDPS/ERA5/ERA5_LSM.nc                                                |
| *outpath*    | Path of output directory.                                                                                                                                                                                                      | path | /data/arsclisys/normal/clim-hydro/TEA-Indicators/masks/                                |


## create_static_files
| NAME            | DESCRIPTION                                                                                                 | TYPE | DEFAULT                                                     |
|-----------------|-------------------------------------------------------------------------------------------------------------|------|-------------------------------------------------------------|
| *season_length* | Number of days in season for threshold calculation. For whole year use 366, for WAS (Apr-Oct) 214.          | int  | 366                                                         |
| *smoothing*     | Radius for spatial smoothing of threshold grid in km. Used for precipitation parameter from SPARTACUS data. | int  | 0                                                           |
| *inpath*        | Path of input directory.                                                                                    | path | /data/arsclisys/normal/clim-hydro/TEA-Indicators/SPARTACUS/ |
| *outpath*       | Path of output directory.                                                                                   | path | /data/arsclisys/normal/clim-hydro/TEA-Indicators/static/    |


## calc_TEA
| NAME             | DESCRIPTION                                                                                                | TYPE  | DEFAULT                                                     |
|------------------|------------------------------------------------------------------------------------------------------------|-------|-------------------------------------------------------------|
| *inpath*         | Path of input directory.                                                                                   | path  | /data/arsclisys/normal/clim-hydro/TEA-Indicators/SPARTACUS/ |
| *outpath*        | Path of output directory.                                                                                  | path  | /data/users/hst/TEA-clean/TEA/                              |
| *recalc_daily*   | Set if daily basis variables (DBVs) should be recalculated or loaded from memory.                          | bool  | false                                                       |
| *decadal*        | Set if decadal TEA indicators should also be calculated. Only possible if end - start >= 10                | bool  | false                                                       |
| *spreads*        | Set if spread estimators of decadal TEA indicators should also be calculated.                              | bool  | false                                                       |
| *decadal_only*   | Set if ONLY decadal TEA indicators should be calculated. Only possible if CTP vars already calculated.     | bool  | false                                                       |
| *recalc_daily*   | Set if decadal indicator variables (DECs) should be recalculated or loaded from memory.                    | bool  | false                                                       |
| *compare_to_ref* | Set if comparison to reference should be done as well.                                                     | bool  | false                                                       |
| *save_old*       | Set if old and new versions should be stored.                                                              | bool  | false                                                       |


## calc_station_TEA
| NAME      | DESCRIPTION                                                                                            | TYPE | DEFAULT                               |
|-----------|--------------------------------------------------------------------------------------------------------|------|---------------------------------------|
| *inpath*  | Path of input directory.                                                                               | path | /data/users/hst/cdrDPS/station_data/  |
| *outpath* | Path of output directory.                                                                              | path | /data/users/hst/TEA-clean/TEA/        |
| *station* | Name of station; Graz, Innsbruck, Wien, Salzburg, BadGleichenberg, Kremsmuenster, or Deutschlandsberg. | str  | Graz                                  |


## calc_amplification_factors
| NAME             | DESCRIPTION               | TYPE  | DEFAULT                                                 |
|------------------|---------------------------|-------|---------------------------------------------------------|
| *inpath*         | Path of input directory.  | path  | /data/users/hst/TEA-clean/TEA/dec_indicator_variables/  |
| *outpath*        | Path of output directory. | path  | /data/users/hst/TEA-clean/TEA/                          |


## calc_AGR_vars
| NAME       | DESCRIPTION                                                                          | TYPE | DEFAULT                                                  |
|------------|--------------------------------------------------------------------------------------|------|----------------------------------------------------------|
| *inpath*   | Path of input directory.                                                             | path | /data/users/hst/TEA-clean/TEA/dec_indicator_variables/   |
| *outpath*  | Path of output directory.                                                            | path | /data/users/hst/TEA-clean/TEA/                           |
| *agr*      | Name of aggregate GeoRegion; EUR, S-EUR, C-EUR, N-EUR, AUT, or ISO2-code of country. | str  | EUR                                                      |
| *spreads*  | Set if spread estimators of decadal TEA indicators should also be calculated.        | bool | false                                                    |
