# TEA metrics Run Control File (RCF) specifications

## [Project-Id and TEArun-Id]
```markdown
TEAmetrics_Version            = TEAmetrics v1.0      ;(vN.n)
RCF_Creation_Date_and_Time    = 2025-09-03T15:01:16Z ;(ISO string*20)
Project_Id                    = TEAmTestProject1     ;(string*25)
TEArun_Id                     = Test1_Europe2AGR_1a  ;(string*25)
```

## [1. Input Datsets Definition] {#input-datasets-def}
```markdown
KeyVarData_DatasetType = ERA5 ;(string*25)
*elem{ERA5, ERA5-Land, ERA5-HEAT, E-OBS, SPARTACUS, CMIP6, …}*

KeyVarData_GenericFilename = ../InpData/ERA5/ERA5-EUR1961-2024v1_*.nc
*(files need to supply the selected dataset type and to contain all relevant*
*data to properly feed the Key Var, GRs, AGRs, and Time Domain Defs)*

ThreshData_DatasetType = ERA5-Land ;(string*25)
*elem{NoDataset, ERA5, ERA5-Land, ERA5-HEAT, E-OBS, SPARTACUS, CMIP6, …}*
*(if NoDataset the ThreshData_GenericFilename is N/A and the ThresholdMap_Type*
*within the Threshold Map and Exceedance Defs must be set to Constant)*

ThreshData_GenericFilename = ../InpData/ERA5L/ERA5L-EUR1961-1990v1_*.nc
*(files need to supply the selected dataset type and to contain all relevant*
*data to properly feed the Key Var, GRs, AGRs, and Threshold Map and Exc Defs)*

NatVarData_DatasetType = StnData_AUT ;(string*25)
*elem{NoDataset, StnData_AUT, ModData_MPI-GE, ModData_CMIP6, …}*
*(if NoDataset the NatVarData_GenericFilename is N/A and no NatVar estimates are*
*computed along with the amplification factor variable timeseries)*

NatVarData_GenericFilename = ../InpData/AUT/GeoSph-HistDailyStDv2_*.nc
*(files need to supply the selected dataset type and to contain all relevant*
*data to properly feed the NatVar Estimation Defs)*
```

## [2. Key Variable Definitions] {#key-variable-def}
```markdown
KeyVariable = TMax ;(string*6)
*elem{TMax, TMin, Tm1H, TmaxUTCI, TmUTCI1H, TmaxWBGT, TmWBGT1H, P24H, P1H, …}*

KeyVariable_LongName = Daily Max Temperature ;(string*50)
*elem{Daily Max Temperature, Daily Min Temperature, Hourly-mean Temperature,*
*Daily Max Universal Thermal Comfort Index, Hourly-mean Universal Thermal Comfort*
*Index, Daily Max Wet-Bulb Globe Temperature, Hourly-mean Wet-Bulb Globe*
*Temperature, Daily Precipitation Sum, Hourly Precipitation Sum, …}*

KeyVariable_GeographicDomain = Land Surface ;(string*25)
*elem{Land Surface, Sea Surface, Earth Surface, <N> hPa Pressure level, <N> km*
*Altitude level, <N> m Ocean depth level, …}*

KeyVariable_MSLAltitudeDomain[2]   = -10, 1500 ;[bottom m, top m]
*(range -10000 to 10000 m, formally from deep oceans to beyond Mt.Everest top;*
*if no “Surface” geographic domain is chosen, this key variable is N/A)*
```

## [3. GeoRegions Definition] {#geo-regions-def}
```markdown
GR_Computation_Type = GridofGRs ;(string*25)
*elem{OneGR, GridofGRs}*

GR_Geolocation_Type = LatLonGrid ;(string*25)
*if OneGR elem{PolygonRegion, CenterLatLon, CenterUTMXY}*
*if GridofGRs elem{LatLonGrid, UTMXYGrid}*

GR_CellShape_Type = LatLonArea ;(string*25)
*if PolygonRegion elem{Polygon)*
*if any other Geolocation Type elem{LatLonArea, UTMXYArea}*

---
/if Types PolygonRegion and Polygon selected:
GR_PolygonRegion = Austria ;(string*25)
GR_PolygonShapefile = ../RegShapefiles/AustriaBorders1.sh
*(from list of polygon-region names, e.g., of countries, states of the world;*
*and the shapefile from those available for the selected GR_PolygonRegion)*

---
/if Type CenterLatLon selected:
GR_LatLonRegion = SAR ;(string*25)
GR_CenterLatLon[2]                     = 47.0000, 15.0000 ;[deg N, deg E]
*(range -90 to 90 deg N, -180 to 180 deg E)*

---
/if Type CenterUTMXY selected:
GR_UTMXYRegion = SAR ;(string*25)
GR_UTMZone = 33N ;(string*3)
GR_CenterUTMXY[2]                      = 5200.00, 550.00 ;[km N|S, km E]
*(range 0 to 10000 km N or S(+FalseN), 0 to 1000 km E; zones 01N|S-60N|S)*

---
/if Type LatLonGrid selected:
GR_LatLonGridRegion = EUR-GRsGrid ;(string*25)
GR_LatLonGrid_SWcorner[2]              = 35.0000, -11.0000 ;[deg N, deg E]
GR_LatLonGrid_NEcorner[2]              = 71.0000, 40.0000 ;[deg N, deg E]
GR_LatLonGrid_Spacing[2]               = 0.5000, 0.5000 ;[deg N, deg E]
*(basic range -90 to 90 deg N, -180 to 180 deg E; SW=SouthWest, NE=NorthEast)*

---
/if Type UTMXYGrid selected:
GR_UTMXYGridRegion = ATR-GRsGrid ;(string*25)
GR_UTMZone = 33N ;(string*3)
GR_UTMXXGrid_SWcorner[2]               = 5100.00, 100.00 ;[km N, km E]
GR_UTMXYGrid_NEcorner[2]               = 5450.00, 650.00 ;[km N, km E]
GR_UTMXYGrid_Spacing[2]                = 10.00, 10.00 ;[km N, km E]
*(range 0 to 10000 km N or S(x+FalseN), 0 to 1000 km E; zones 01N|S-60N|S)*

---
/if Type LatLonArea selected:
GR_LatLonArea_aroundCenterLatLon[2]    = 2.0000, 2.0000 ;[deg S-N, degEq W-E]
*(full width in deg of the GR cell’s LatLon area in South-North direction, and*
*in degEq (=deg at Equator) in West-East direction; vs cell center location)*

---
/if Type UTMXYArea selected:
GR_UTMXYArea_aroundCenterLatLon[2]     = 100.00, 100.00 ;[km S-N, km W-E]
*(full width in km of the GR cell’s UTMXY area in South-North and West-East*
*direction; relative to the cell center location)*
```

## [4. Aggregate GeoRegions Definition] {#aggregate-geo-regions-def}
```markdown
AGR_Computation_Type = SampleofAGRs ;(string*25)
*elem{NoAGR, OneAGR, SampleofAGRs, GridofAGRs};*
*if NoAGR all further AGR type variables are N/A (no AGR results computed)*

AGR_Geolocation_Type = PolygonRegionSample ;(string*25)
*if OneAGR elem{PolygonRegion, CenterLatLon, CenterUTMXY}*
*if SampleofAGRs elem{PolygonRegionSample}*
*if GridofAGRs elem{LatLonGrid, UTMXYGrid}*

AGR_CellShape_Type = LatLonArea ;(string*25)
*if PolygonRegion or PolygonRegionSample elem{Polygon}*
*if any other Geolocation Type elem{LatLonArea, UTMXYArea}*

---
/if Types PolygonRegion and Polygon selected:
AGR_PolygonRegion = Germany ;(string*25)
AGR_PolygonShapefile = ../RegShapefiles/DE_Borders1.sh
*(from list of polygon-region names, e.g., of countries & states of the world;*
*and the shapefile from those available for the selected AGR_PolygonRegion)*

---
/if Types PolygonRegionSample and Polygon selected:
AGR_PolygonRegionsList = ../RegListfiles/Eur2AGR_1a.rlist
AGR_PolygonShapefilesList = ../RegListfiles/Eur2AGR_1a.shlist
*(same as for individual PolygonRegion, but here listfiles of these inputs)*

---
/if Type CenterLatLon selected:
AGR_LatLonRegion = EUR ;(string*25)
AGR_CenterLatLon[2]                   = 53.50, 15.00 ;[deg N, deg E]
*(range -90 to 90 deg N, -180 to 180 deg E)*

---
/if Type CenterUTMXY selected:
AGR_UTMXYRegion = ATR ;(string*25)
AGR_UTMZone = 33N ;(string*3)
AGR_CenterUTMXY[2]                    = 5280., 380. ;[km N|S, km E]
*(range 0 to 10000 km N or S(+FalseN), 0 to 1000 km E; zones 01N|S-60N|S)*

---
/if Type LatLonGrid selected:
AGR_LatLonGridRegion = EUR-10x20degAGRs ;(string*25)
AGR_LatLonGrid_SWcorner[2]            = 40.00, 0.00 ;[deg N, deg E]
AGR_LatLonGrid_NEcorner[2]            = 65.00, 30.00 ;[deg N, deg E]
AGR_LatLonGrid_Spacing[2]             = 5.00, 10.00 ;[deg N, deg E]
*(basic range -90 to 90 deg N, -180 to 180 deg E; SW=SouthWest, NE=NorthEast)*

---
/if Type UTMXYGrid selected:
AGR_UTMXYGridRegion = ATR-100x100kmAGRs ;(string*25)
AGR_UTMZone = 33N ;(string*3)
AGR_UTMXXGrid_SWcorner[2]             = 5150.00, 150.00 ;[km N, km E]
AGR_UTMXYGrid_NEcorner[2]             = 5400.00, 600.00 ;[km N, km E]
AGR_UTMXYGrid_Spacing[2]              = 50.00, 50.00 ;[km N, km E]
*(range 0 to 10000 km N or S(x+FalseN), 0 to 1000 km E; zones 01N|S-60N|S)*

---
/if Type LatLonArea selected:
AGR_LatLonArea_aroundCenterLatLon[2]  = 10.00, 20.00 ;[deg S-N, degEq W-E]
*(full width in deg of the AGR cell’s LatLon area in South-North direction, and*
*in degEq (=deg at Equator) in West-East direction; vs cell center loc.)*

---
/if Type UTMXYArea selected:
AGR_UTMXYArea_aroundCenterLatLon[2]   = 100., 100. ;[km S-N, km W-E]
*(full width in km of the AGR cell’s UTMXY area in South-North and West-East*
*direction; relative to the cell center location)*
```

## [5. Time Domain Definitions] {#time-domain-def}
```markdown
TimePeriod_StartDate = 1961-01-01 ;(datestring*10)
TimePeriod_EndDate = 2024-12-31 ;(datestring*10)
*(range from 1 year up to the time period available from the input data)*

TimeResolution_KeyVarFields = Hourly ;(string*6)
*elem{Hourly, Daily}*

TimeSampling_TEAmetrics = Annual ;(string*6)
*elem{Annual}*

AnnualCTP_TEAmetrics = ANN ;(string*3)
*elem{ANN; WAS, ESS, EWS; MAM...DJF; Jan...Dec; warm season WAS: Apr-Oct,*
*extended summer season ESS: May-Sep, extended winter season EWS: Nov-Mar}*

AvgWindow_DecadalTEAmetrics[3]    = 10, -5, 4 ;[yrs, -deltayrs, +deltayrs]
*(range 1 to 30 yrs; deltayrs specify the positioning of the averaging window*
*about its core year, i.e., the degree of time asymmetry vs core year; the*
*selection must suitably fit into the overall time period chosen; setting the*
*window to [1, 0, 0] deactivates computation of the decadal TEA metrics)*

RefPeriod_TEAmetrics[2]           = 1961, 1990 ;[first yr, last yr]
CCPeriod_TEAmetrics[2]            = 2010, 2024 ;[first yr, last yr]
*(time periods at least as long as the AvgWindow_DecadalTEAmetrics choice;*
*setting first yr = last yr deactivates the Ref and CC metrics computation)*
```

## [6. Threshold Map and Exceedance Definitions] {#threshold-map-and-exceedance-def}
```markdown
ThresholdMap_Type = Percentile ;(string*25)
*elem{Constant, Percentile, SeasonalVar Percentile}*

ThresholdExceedance_Type = Exceed Thres Upward ;(string*25)
*elem{Exceed Thres Upward, Exceed Thres Downward, Confined in ThresRange}*

MinimumGRThresExceedanceArea = 1 ;[areals](=100km2)
*(recommended range: 1 to 10 areals; min. 0, max. GR eligible cell area)*

---
/if Types Constant and Exceed Thres Upward or Exceed Thres Downward selected:
Constant_Thres = 30.0 ;[degC]
/if Types Constant and Confined in ThresRange selected:
Constant_ThresRange[2]               = -1.0, 2.0 ;[degC, degC]
*(examples for Key Variable TMax here; reasonable values and units will depend*
*on the Key Var selected, and on which type of exceedance is chosen)*

---
/if Types Percentile or SeasonalVar Percentile selected:
Percentile_EstimationTimePeriod[2]   = 1961, 1990 ;[first yr, last yr]
*(recommended estimation time period is at least 30 years)*

Percentile_EstimationCTPperYear = ANN ;(string*3)
*elem{ANN; WAS, ESS, EWS; MAM...DJF; Jan...Dec; warm season WAS: Apr-Oct,*
*extended summer season ESS: May-Sep, extended winter season EWS: Nov-Mar}*

---
/if Types Percentile or SeasonalVar Percentile and Exceed Thres Upward or Exceed
Thres Downward selected:
Percentile_Thres = 99.0 ;[NN.N](=NN.Nth percentile)
/if Types Percentile or SeasonalVar Percentile and Confined in ThresRange
selected:
Percentile_ThresRange[2]             = 95.0, 99.0 ;[NN.N, NN.N]
*(recommended range for percentiles is within 00.2th to 99.8th percentile; if*
*the SeasonalVar Percentile type is set, the seasonal variation is accounted for*
*in the sense that specific threshold maps---or pairs of maps if Confined in*
*ThresRange is set---are computed for all individual months of the CTP per year,*
*enabling to analyze pctle-exceedances on top of the avg seasonal cycle)*
```

## [7. Natural Variability Estimation Definitions] {#natural-variability-estimation-def}
```markdown
NatVarPeriod_StartDate = 1880-01-01 ;(datestring*10)
NatVarPeriod_EndDate = 1990-12-31 ;(datestring*10)
*(range from 30 years, the minimum period considered formally useful (for a*
*station included) to contribute to NatVar estimation, up to the time period*
*covered by the NatVar data; time periods longer than the longest (station) data*
*record available in the NatVar input data file are not meaningful)*

TimeResolution_NatVarData = Daily ;(string*6)
*elem{Hourly, Daily}*

---
/if NatVarData_DatasetType StnData_AUT selected as part of the Input Datasets
Definition group:
NatVarData_StationsList = ../StnListfiles/AUT-GeoSphv2_1a.stnlist
*(includes the station data IDs and names that must be consistent with the*
*station data available within the files under the NatVarData_GenericFilename*
*selected as part of the Input Datasets Def)*
```

[EOF]