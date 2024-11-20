import xarray as xr

from scripts.general_stuff.var_attrs import get_attrs


class TEAIndicators:
    """
    Class to calculate TEA indicators
    """
    
    def __init__(self, input_data_grid, threshold_grid):
        self.threshold_grid = threshold_grid
        self.input_data_grid = input_data_grid
        self.grid = xr.Dataset()
        self.gr_vars = None
        
        if input_data_grid.shape[-2:] != threshold_grid.shape:
            raise ValueError("Data and threshold grid must have the same area")
        
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


