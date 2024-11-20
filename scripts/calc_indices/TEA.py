"""
Threshold Exceedance Amount (TEA) indicators Class implementation
Based on:
TODO: add reference to the paper
Equation numbers refer to Supplementary Notes
"""
import xarray as xr
import numpy as np

from scripts.general_stuff.var_attrs import get_attrs


class TEAIndicators:
    """
    Class to calculate TEA indicators
    """
    
    def __init__(self, input_data_grid, threshold_grid, area_grid=None):
        """
        Initialize TEAIndicators object
        Args:
            input_data_grid: gridded input data (e.g. temperature, precipitation)
            threshold_grid: gridded threshold values
            area_grid: results containing the area of each results cell, if None, area is assumed to be 1 for each cell
                       nan values mask out the corresponding results cells
        """
        self.threshold_grid = threshold_grid
        if area_grid is None:
            area_grid = xr.ones_like(threshold_grid)
        self.area_grid = area_grid
        self.input_data_grid = input_data_grid
        self.results = xr.Dataset()
        self.gr_vars = None
        
        if input_data_grid.shape[-2:] != threshold_grid.shape:
            raise ValueError("Input data and threshold results must have the same area")
        if input_data_grid.shape[-2:] != area_grid.shape:
            raise ValueError("Input data and area results must have the same shape")
        
    def calc_DTEC(self):
        """
        calculate Daily Threshold Exceedance Count (equation 01)
        note that 0 values are stored as NaN for optimization
        """
        if self.results['DTEM'] is None:
            self.calc_DTEM()
        dtem = self.results.DTEM
        dtec = dtem.where(dtem.isnull(), 1)
        dtec.attrs = get_attrs(vname='DTEC')
        self.results['DTEC'] = dtec
    
    def calc_DTEC_GR(self, min_area=1):
        """
        calculate Daily Threshold Exceedance Count (GR) (equation 03)
        note that 0 values are stored as NaN for optimization

        @param min_area: minimum area for a timestep to be considered as exceedance (same unit as area_grid)
        """
        if self.results['DTEA_GR'] is None:
            self.calc_DTEA_GR()
        dtea_gr = self.results.DTEA_GR
        dtec_gr = xr.where(dtea_gr >= min_area, 1, np.nan)
        dtec_gr.attrs = get_attrs(vname='DTEC_GR')
        self.results['DTEC_GR'] = dtec_gr
    
    def calc_DTEA(self):
        """
        calculate Daily Threshold Exceedance Area (equation 02)
        note that 0 values are stored as NaN for optimization
        """
        if self.results['DTEC'] is None:
            self.calc_DTEC()
        dtec = self.results.DTEC
        # equation 02_1 not needed (cells with TEC == 0 are already nan)
        # equation 02_2
        dtea = dtec * self.area_grid
        dtea.attrs = get_attrs(vname='DTEA')
        self.results['DTEA'] = dtea
        
    def calc_DTEA_GR(self):
        """
        calculate Daily Threshold Exceedance Area (GR) (equation 06)
        """
        if self.results['DTEA'] is None:
            self.calc_DTEA()
        dtea = self.results.DTEA
        dtea_gr = dtea.sum(axis=(1, 2), skipna=True)
        dtea_gr = dtea_gr.rename('DTEA_GR')
        dtea_gr.attrs = get_attrs(vname='DTEA_GR')
        self.results['DTEA_GR'] = dtea_gr
    
    def calc_DTEM(self):
        """
        calculate Daily Threshold Exceedance Magnitude (equation 07)
        note that 0 values are stored as NaN for optimization
        """
        dtem = self.input_data_grid - self.threshold_grid
        dtem = dtem.where(dtem > 0).astype('float32')
        dtem.attrs = get_attrs(vname='DTEM')
        self.results['DTEM'] = dtem

