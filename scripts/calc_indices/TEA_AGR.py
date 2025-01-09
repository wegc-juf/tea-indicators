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

from scripts.general_stuff.var_attrs import get_attrs, equal_vars
from TEA import TEAIndicators


class TEAAgr(TEAIndicators):
    """
    Class for Threshold Exceedance Amount (TEA) indicators for aggregated georegions (AGR)
    """
    def __init__(self, input_data_grid=None, threshold_grid=None, area_grid=None, mask=None, min_area=0.0001,
                 agr_resolution=0.5):
        """
        initialize TEA object
        
        Args:
            input_data_grid: input data grid
            threshold_grid: threshold grid
            area_grid: area grid
            mask: mask grid
            min_area: minimum area for valid grid cells
            agr_resolution: resolution for aggregated GeoRegion (in degrees)
        """
        super().__init__(input_data_grid=input_data_grid, threshold_grid=threshold_grid, area_grid=area_grid,
                         mask=mask, min_area=min_area)
        if self.area_grid is not None:
            self.lat_resolution = self.area_grid.lat[1] - self.area_grid.lat[0]
        else:
            self.lat_resolution = None
        self.agr_resolution = agr_resolution
        
        self.ctp_results = None
        
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

    def select_sub_gr(self, lat, lon, cell_size_lat, land_frac_min=0):
        """
        select data of GeoRegion sub-cell and weight edges
        Args:
            lat: center latitude of cell (in degrees)
            lon: center longitude of cell (in degrees)
            cell_size_lat: size of cell in latitudinal direction (in degrees)
            land_frac_min: minimum fraction of land below 1500m

        Returns:
            cell_data: data of cell
            cell_static: static data of cell
        """
        
        lat_off = cell_size_lat / 2
        lon_off = 1 / np.cos(np.deg2rad(lat)) * lat_off
        
        if land_frac_min > 0:
            # get land-sea mask
            cell_lsm = self.mask.sel(lat=slice(lat + lat_off, lat - lat_off),
                                     lon=slice(lon - lon_off, lon + lon_off))
            
            # calculate fraction covered by valid cells (land below 1500 m)
            land_frac = cell_lsm.sum() / np.size(cell_lsm)
            if land_frac < land_frac_min:
                return None
        
        # select data for cell
        cell_data = self.daily_results.sel(lat=slice(lat + lat_off, lat - lat_off),
                                           lon=slice(lon - lon_off, lon + lon_off))
        
        # select static data for cell
        cell_area_grid = self.area_grid.sel(lat=slice(lat + lat_off, lat - lat_off),
                                            lon=slice(lon - lon_off, lon + lon_off))
        if len(cell_area_grid.lat) == 0:
            raise ValueError('No valid cell found, check why this happens')
        
        # TODO: two options: either return data itself and stack to xarray then calculate TEA or return individual TEA
        # objects
        tea_sub_gr = TEAIndicators(area_grid=cell_area_grid, min_area=0.0001, unit=self.unit)
        tea_sub_gr.set_daily_results(cell_data)
        return tea_sub_gr

    def set_ctp_results(self, lat, lon, ctp_results):
        """
        set CTP variables for point
        Args:
            lat: latitude
            lon: longitude
            ctp_results: CTP GR data for point
        """
        if self.ctp_results is None:
            data_vars = [var for var in ctp_results.data_vars]
            var_dict = {}
            lats = np.arange(self.input_data_grid.lat.max(),
                             self.input_data_grid.lat.min(),
                             -self.agr_resolution)
            lons = np.arange(self.input_data_grid.lon.min(),
                             self.input_data_grid.lon.max(),
                             self.agr_resolution)
            for var in data_vars:
                var_dict[var] = (['time', 'lat', 'lon'], np.nan * np.ones((len(ctp_results.time),
                                                                           len(lats),
                                                                           len(lons))))
            self.ctp_results = xr.Dataset(coords=dict(time=ctp_results.time,
                                                      lon=lons,
                                                      lat=lats),
                                          data_vars=var_dict,
                                          attrs=ctp_results.attrs)
            
        self.ctp_results.loc[dict(lat=lat, lon=lon)] = ctp_results
