"""
Threshold Exceedance Amount (TEA) indicators Class implementation
Based on:
TODO: add reference to the paper
Equation numbers refer to Supplementary Notes
"""
import xarray as xr

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
            area_grid: grid containing the area of each grid cell, if None, area is assumed to be 1 for each cell
                       nan values mask out the corresponding grid cells
        """
        self.threshold_grid = threshold_grid
        if area_grid is None:
            area_grid = xr.ones_like(threshold_grid)
        self.area_grid = area_grid
        self.input_data_grid = input_data_grid
        self.grid = xr.Dataset()
        self.gr_vars = None
        
        if input_data_grid.shape[-2:] != threshold_grid.shape:
            raise ValueError("Input data and threshold grid must have the same area")
        if input_data_grid.shape[-2:] != area_grid.shape:
            raise ValueError("Input data and area grid must have the same shape")
        
    def calc_DTEC(self):
        """
        calculate Daily Threshold Exceedance Count (equation 01)
        """
        if self.grid['DTEM'] is None:
            self.calc_DTEM()
        dtem = self.grid.DTEM
        dtec = dtem.where(dtem.isnull(), 1)
        dtec.attrs = get_attrs(vname='DTEC')
        self.grid['DTEC'] = dtec
    
    def calc_DTEM(self):
        """
        calculate Daily Threshold Exceedance Magnitude (equation 07)
        note that 0 values are stored as NaN for optimization
        """
        dtem = self.input_data_grid - self.threshold_grid
        dtem = dtem.where(dtem > 0).astype('float32')
        dtem.attrs = get_attrs(vname='DTEM')
        self.grid['DTEM'] = dtem
    
    def calc_DTEA(self):
        """
        calculate Daily Threshold Exceedance Area (equation 02)
        """
        if self.grid['DTEC'] is None:
            self.calc_DTEC()
        dtec = self.grid.DTEC
        # equation 02_1 not needed (cells with TEC == 0 are already nan)
        # equation 02_2
        dtea = dtec * self.area_grid
        dtea.attrs = get_attrs(vname='DTEA')
        self.grid['DTEA'] = dtea
    


