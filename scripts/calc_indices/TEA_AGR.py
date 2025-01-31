"""
Threshold Exceedance Amount (TEA) indicators Class implementation for aggregated georegions (AGR)
Based on:
TODO: add reference to the paper
Equation numbers refer to Supplementary Notes
"""
import warnings

import xarray as xr
import pandas as pd
import numpy as np
import datetime as dt
import time

from scripts.general_stuff.var_attrs import get_attrs, equal_vars
from scripts.general_stuff.TEA_logger import logger
from TEA import TEAIndicators


class TEAAgr(TEAIndicators):
    """
    Class for Threshold Exceedance Amount (TEA) indicators for aggregated georegions (AGR)
    """
    def __init__(self, input_data_grid=None, threshold_grid=None, area_grid=None, mask=None, min_area=0.0001,
                 agr_resolution=0.5, land_sea_mask=None, agr_mask=None, land_frac_min=0.5, cell_size_lat=2, ctp=None,
                 **kwargs):
        """
        initialize TEA object
        
        Args:
            input_data_grid: input data grid
            threshold_grid: threshold grid
            area_grid: area grid
            mask: mask grid
            min_area: minimum area for valid grid cells
            agr_resolution: resolution for aggregated GeoRegion (in degrees)
            land_sea_mask: land-sea mask
            land_frac_min: minimum fraction of land below 1500m
            cell_size_lat: size of AGR cell in latitudinal direction (in degrees)
            ctp: climatic time period. For definition see TEAIndicators.set_ctp()
        """
        super().__init__(input_data_grid=input_data_grid, threshold_grid=threshold_grid, area_grid=area_grid,
                         mask=mask, min_area=min_area, apply_mask=False, ctp=ctp, **kwargs)
        if self.area_grid is not None:
            self.lat_resolution = abs(self.area_grid.lat.values[0] - self.area_grid.lat.values[1])
        else:
            self.lat_resolution = None
        self.agr_resolution = agr_resolution
        self.agr_mask = None
        self.agr_area = None
        self.land_sea_mask = land_sea_mask
        self.land_frac_min = land_frac_min
        self.cell_size_lat = cell_size_lat
        
        if agr_mask is None and mask is not None and area_grid is not None:
            self._generate_agr_mask()
        else:
            self.agr_mask = agr_mask
        
        # daily basis variables for aggregated GeoRegion
        self.dbv_agr_results = None
    
    def calc_daily_basis_vars(self, grid=True, gr=False):
        """
        calculate all daily basis variables
        
        Args:
            grid: set if grid cells should be calculated (default: True)
            gr: set if GeoRegion values should be calculated (default: False)
        """
        super().calc_daily_basis_vars(grid=grid, gr=gr)
        
    @staticmethod
    def _my_rolling_test(data, axis, keep_attrs=True, step=1):
        """
        custom rolling function
        """
        return data

    def regrid_to_grs(self):
        """
        regrid daily basis variables to individual georegions
        """
        # apply custom rolling function
        self.daily_results.rolling(lat=4, lon=4, center=True).reduce(self._my_rolling_test, step=2).compute()

    def select_sub_gr(self, lat, lon):
        """
        select data of GeoRegion sub-cell and weight edges
        Args:
            lat: center latitude of cell (in degrees)
            lon: center longitude of cell (in degrees)

        Returns:
            cell_data: data of cell
            cell_static: static data of cell
        """
        
        lat_off = self.cell_size_lat / 2
        lon_off_exact = 1 / np.cos(np.deg2rad(lat)) * lat_off
        size_exact = lon_off_exact * lat_off
        
        round_coords = True
        if round_coords:
            lon_off = np.round(lon_off_exact * 4, 0) / 4.
        size_real = lon_off * lat_off
        area_frac = size_real / size_exact
        
        if self.land_frac_min > 0:
            # get land-sea mask
            cell_lsm = self.land_sea_mask.sel(lat=slice(lat + lat_off, lat - lat_off),
                                              lon=slice(lon - lon_off, lon + lon_off))
            
            # calculate fraction covered by valid cells (land below 1500 m)
            land_frac = cell_lsm.sum() / np.size(cell_lsm)
            if land_frac < self.land_frac_min:
                return None
        
        # select data for cell
        cell_data = self.daily_results.sel(lat=slice(lat + lat_off, lat - lat_off),
                                           lon=slice(lon - lon_off, lon + lon_off))
        # compensate rounding errors
        cell_data['DTEA'] = cell_data['DTEA'] / area_frac
        
        # select static data for cell
        cell_area_grid = self.area_grid.sel(lat=slice(lat + lat_off, lat - lat_off),
                                            lon=slice(lon - lon_off, lon + lon_off))
        cell_area_grid = cell_area_grid / area_frac
        
        if len(cell_area_grid.lat) == 0:
            raise ValueError('No valid cell found, check why this happens')
        
        # TODO: two options: either return data itself and stack to xarray then calculate TEA or return individual TEA
        # objects
        tea_sub_gr = TEAIndicators(area_grid=cell_area_grid, min_area=self.min_area, unit=self.unit, ctp=self.CTP)
        tea_sub_gr.set_daily_results(cell_data)
        return tea_sub_gr
    
    def set_dbv_results(self, lat, lon, dbv_results):
        """
        set daily basis variables for point
        Args:
            lat: latitude
            lon: longitude
            dbv_results: daily basis GR data for point
        """
        if self.dbv_agr_results is None:
            data_vars = [var for var in dbv_results.data_vars]
            var_dict = {}
            lats, lons = self._get_lats_lons()
            for var in data_vars:
                var_dict[var] = (['time', 'lat', 'lon'], np.nan * np.ones((len(dbv_results.time),
                                                                           len(lats), len(lons))))
            self.dbv_agr_results = xr.Dataset(coords=dict(time=dbv_results.time,
                                                          lon=lons,
                                                          lat=lats),
                                              data_vars=var_dict,
                                              attrs=dbv_results.attrs)
        
        self.dbv_agr_results.loc[dict(lat=lat, lon=lon)] = dbv_results

    def get_dbv_results(self, grid=True, gr=True):
        """
        get daily basis variable results for aggregated GeoRegion

        Args:
            grid: get gridded results. Default: True
            gr: get GR results. Default: True
        """
        gr_vars = [var for var in self.dbv_agr_results.data_vars if 'GR' in var]
        grid_vars = [var for var in self.dbv_agr_results.data_vars if 'GR' not in var]
        if not grid:
            return self.dbv_agr_results.drop_vars(grid_vars)
        if not gr:
            return self.dbv_agr_results.drop_vars(gr_vars)
        else:
            return self.dbv_agr_results
    
    def save_dbv_results(self, filepath):
        """
        save all daily basis variable results to filepath
        """
        with warnings.catch_warnings():
            # ignore warnings due to nan multiplication
            warnings.simplefilter("ignore")
            self.dbv_agr_results.to_netcdf(filepath)

    def set_ctp_results(self, lat, lon, ctp_results):
        """
        set CTP variables for point
        Args:
            lat: latitude
            lon: longitude
            ctp_results: CTP GR data for point
        """
        # remove GR from variable names
        ctp_results = ctp_results.rename({var: var.replace('_GR', '') for var in ctp_results.data_vars})
        
        if self.CTP_results is None or not len(self.CTP_results.data_vars):
            data_vars = [var for var in ctp_results.data_vars]
            var_dict = {}
            lats, lons = self._get_lats_lons()
            for var in data_vars:
                var_dict[var] = (['time', 'lat', 'lon'], np.nan * np.ones((len(ctp_results.time),
                                                                           len(lats),
                                                                           len(lons))))
            self.CTP_results = xr.Dataset(coords=dict(time=ctp_results.time,
                                                      lon=lons,
                                                      lat=lats),
                                          data_vars=var_dict,
                                          attrs=ctp_results.attrs)
            
        self.CTP_results.loc[dict(lat=lat, lon=lon)] = ctp_results
        
        # set attributes for variables
        for var in ctp_results.data_vars:
            if 'attrs' not in self.CTP_results[var]:
                attrs = ctp_results[var].attrs
                new_attrs = get_attrs(vname=var)
                attrs['long_name'] = new_attrs['long_name']
                self.CTP_results[var].attrs = attrs
        
    def get_ctp_results(self, grid=True, gr=True):
        """
        get CTP results for aggregated GeoRegion

        Args:
            grid: get gridded results. Default: True
            gr: get GR results. Default: True
        """
        gr_vars = [var for var in self.CTP_results.data_vars if 'GR' in var]
        grid_vars = [var for var in self.CTP_results.data_vars if 'GR' not in var]
        if not grid:
            return self.CTP_results.drop_vars(grid_vars)
        if not gr:
            return self.CTP_results.drop_vars(gr_vars)
        else:
            return self.CTP_results
    
    def save_ctp_results(self, filepath):
        """
        save all CTP results to filepath
        """
        with warnings.catch_warnings():
            # ignore warnings due to nan multiplication
            warnings.simplefilter("ignore")
            self.CTP_results.to_netcdf(filepath)
    
    def apply_mask(self):
        """
        apply AGR mask to daily basis variables and CTP results
        """
        if self.dbv_agr_results is not None:
            self.dbv_agr_results = self.dbv_agr_results.where(self.agr_mask > 0)
        if self.CTP_results is not None:
            self.CTP_results = self.CTP_results.where(self.agr_mask > 0)
    
    def calc_tea_agr(self, lats=None, lons=None):
        """
        calculate TEA indicators for all aggregated GeoRegions
        Args:
            lats: Latitudes (default: get automatically)
            lons: Longitudes (default: get automatically)

        Returns:

        """
        if lats is None:
            lats, lons = self._get_lats_lons()
        
        for lat in lats:
            self._calc_tea_lat(lat, lons=lons)
    
    def _get_lats_lons(self, margin=None):
        """
        get latitudes and longitudes for GeoRegion grid
        """
        if margin is None:
            margin = self.cell_size_lat

        lats = np.arange(self.input_data_grid.lat.max() - margin,
                         self.input_data_grid.lat.min() - self.agr_resolution + margin,
                         -self.agr_resolution)
        lons = np.arange(self.input_data_grid.lon.min() + margin,
                         self.input_data_grid.lon.max() + self.agr_resolution - margin,
                         self.agr_resolution)
        return lats, lons
    
    def _generate_agr_mask(self):
        """
        generate mask for aggregated GeoRegion
        """
        logger.info('Generating AGR mask')
        lats, lons = self._get_lats_lons(margin=0)
        mask_orig = self.mask
        area_orig = self.area_grid
        res_orig = self.lat_resolution
        
        mask_agr = xr.DataArray(data=np.ones((len(lats), len(lons))) * np.nan,
                                coords={'lat': (['lat'], lats), 'lon': (['lon'], lons)},
                                dims={'lat': (['lat'], lats), 'lon': (['lon'], lons)})
        mask_agr = mask_agr.rename('mask_lt1500')
        
        area_agr = xr.DataArray(data=np.ones((len(lats), len(lons))) * np.nan,
                                coords={'lat': (['lat'], lats), 'lon': (['lon'], lons)},
                                dims={'lat': (['lat'], lats), 'lon': (['lon'], lons)})
        area_agr = area_agr.rename('area_grid')
        
        for llat in mask_agr.lat:
            for llon in mask_agr.lon:
                cell_orig = mask_orig.sel(lat=slice(llat, llat - res_orig),
                                          lon=slice(llon, llon + res_orig))
                cell_area = area_orig.sel(lat=slice(llat, llat - res_orig),
                                          lon=slice(llon, llon + res_orig))
                valid_cells = cell_orig.sum()
                if valid_cells == 0:
                    continue
                vcell_frac = valid_cells / cell_orig.size
                mask_agr.loc[llat, llon] = vcell_frac.values
                area_agr.loc[llat, llon] = cell_area.sum().values
        
        self.agr_mask = mask_agr
        self.agr_area = area_agr

    def _calc_tea_lat(self, lat, lons=None):
        """
        calculate TEA indicators for all longitudes of a latitude
        Args:
            lat: Latitude
            lons: Longitudes (default: get automatically)

        Returns:

        """
        if lons is None:
            lats, lons = self._get_lats_lons()
        
        # step through all longitudes
        for ilon, lon in enumerate(lons):
            # this comment is necessary to suppress an unnecessary PyCharm warning for lon
            # noinspection PyTypeChecker
            logger.info(f'Processing lat {lat}, lon {lon}')
            start_time = time.time()
            tea_sub = self.select_sub_gr(lat=lat, lon=lon)
            if tea_sub is None:
                continue
            
            # calculate daily basis variables
            tea_sub.calc_daily_basis_vars(grid=False)
            # TODO check if this is necessary
            # dbv_results = tea_sub.get_daily_results(gr=True, grid=False).compute()
            # self.set_dbv_results(lat, lon, dbv_results)
            
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', message='invalid value encountered in multiply')
                # calculate CTP indicators
                tea_sub.calc_annual_CTP_indicators(drop_daily_results=True)
                
                # set agr_results for lat and lon
                ctp_results = tea_sub.get_CTP_results(gr=True, grid=False).compute()
            
            self.set_ctp_results(lat, lon, ctp_results)
            end_time = time.time()
            logger.debug(f'Lat {lat}, lon {lon} processed in {end_time - start_time} seconds')
        
        
