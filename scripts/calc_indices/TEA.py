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
    
    def __init__(self, input_data_grid=None, threshold_grid=None, min_area=None, area_grid=None, low_extreme=False,
                 testing=False):
        """
        Initialize TEAIndicators object
        Args:
            input_data_grid: gridded input data (e.g. temperature, precipitation)
            threshold_grid: gridded threshold values
            area_grid: results containing the area of each results cell, if None, area is assumed to be 1 for each cell
                       nan values mask out the corresponding results cells
            min_area: minimum area for a timestep to be considered as exceedance (same unit as area_grid)
        """
        self.threshold_grid = threshold_grid
        if area_grid is None and threshold_grid is not None:
            area_grid = xr.ones_like(threshold_grid)
        self.area_grid = area_grid
        self.input_data_grid = input_data_grid
        self.results = xr.Dataset()
        self.min_area = min_area
        self.gr_vars = None
        self.low_extreme = low_extreme
        self.testing = testing
        
        if input_data_grid is not None:
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
    
    def calc_DTEC_GR(self, min_area=None):
        """
        calculate Daily Threshold Exceedance Count (GR) (equation 03)
        note that 0 values are stored as NaN for optimization

        @param min_area: minimum area for a timestep to be considered as exceedance (same unit as area_grid)
        """
        if min_area is None:
            min_area = self.min_area
        if self.results['DTEA_GR'] is None:
            self.calc_DTEA_GR()
        dtea_gr = self.results.DTEA_GR
        dtec_gr = xr.where(dtea_gr >= min_area, 1, np.nan)
        dtec_gr.attrs = get_attrs(vname='DTEC_GR')
        self.results['DTEC_GR'] = dtec_gr
    
    def calc_DTEEC(self):
        """
        calculate Daily Threshold Exceedance Event Count (equation 04)
        """
        if self.results['DTEC'] is None:
            self.calc_DTEC()
        dtec = self.results.DTEC
        
        dteec = xr.full_like(dtec, np.nan)
        
        # do not calculate DTEEC in testing mode
        if not self.testing:
            dtec_3d = dtec.values
            # loop through all rows and calculate DTEEC
            for iy in range(len(dtec_3d[0, :, 0])):
                dtec_row = dtec_3d[:, iy, :]
                # skip all nan rows
                if np.isnan(dtec_row).all():
                    continue
                dteec_row = np.apply_along_axis(self._calc_dteec_1d, axis=0, arr=dtec_row)
                dteec[:, iy, :] = dteec_row
        dteec.attrs = get_attrs(vname='DTEEC')
        self.results['DTEEC'] = dteec
    
    def calc_DTEEC_GR(self):
        """
        calculate Daily Threshold Exceedance Event Count (GR) (equation 05)
        """
        if self.results['DTEC_GR'] is None:
            self.calc_DTEC_GR()
            
        dtec_gr = self.results.DTEC_GR
        dteec_np = self._calc_dteec_1d(dtec_cell=dtec_gr.values)
        dteec_gr = xr.DataArray(dteec_np, coords=dtec_gr.coords, dims=dtec_gr.dims)

        dteec_gr.attrs = get_attrs(vname='DTEEC_GR')
        self.results['DTEEC_GR'] = dteec_gr

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
        if self.low_extreme:
            dtem = self.threshold_grid - self.input_data_grid
        else:
            dtem = self.input_data_grid - self.threshold_grid
        dtem = dtem.where(dtem > 0).astype('float32')
        dtem.attrs = get_attrs(vname='DTEM')
        self.results['DTEM'] = dtem
        
    def calc_DTEM_max_gr(self):
        """
        calculate maximum DTEM for GR (equation 09)
        """
        if self.results['DTEM'] is None:
            self.calc_DTEM()
        if self.results['DTEC_GR'] is None:
            self.calc_DTEC_GR()
        dtem = self.results.DTEM
        dtem_max = dtem.max(dim=self.threshold_grid.dims)
        dtem_max = dtem_max.where(self.results.DTEC_GR == 1)
        dtem_max = dtem_max.rename('DTEM_max_gr')
        dtem_max.attrs = get_attrs(vname='DTEM_Max')
        self.results['DTEM_max_gr'] = dtem_max

    def calc_DTEM_GR(self):
        """
        calculate Daily Threshold Exceedance Magnitude (GR) (equation 08)
        """
        if self.results['DTEA_GR'] is None:
            self.calc_DTEA_GR()
        if self.results['DTEM'] is None:
            self.calc_DTEM()
        if self.results['DTEC_GR'] is None:
            self.calc_DTEC_GR()
        dtea_gr = self.results.DTEA_GR
        dtem = self.results.DTEM
        dtec_gr = self.results.DTEC_GR
        area_fac = self.area_grid / dtea_gr
        dtem_gr = (dtem * area_fac).sum(axis=(1, 2), skipna=True)
        dtem_gr = dtem_gr.where(dtec_gr == 1)
        dtem_gr = dtem_gr.rename(f'{dtem.name}_GR')
        dtem_gr.attrs = get_attrs(vname='DTEM_GR')
        self.results['DTEM_GR'] = dtem_gr
    
    def calc_daily_basis_vars(self):
        """
        calculate all daily basis variables
        """
        self.calc_DTEM()
        self.calc_DTEC()
        self.calc_DTEA()
        self.calc_DTEA_GR()
        self.calc_DTEC_GR()
        self.calc_DTEM_GR()
        self.calc_DTEM_max_gr()
        self.calc_DTEEC()
        self.calc_DTEEC_GR()
    
    def save_results(self, filepath):
        """
        save all variables to filepath
        """
        self.results.to_netcdf(filepath)
    
    def load_results(self, filepath):
        """
        load all variables from filepath
        """
        self.results = xr.open_dataset(filepath)
        
    @staticmethod
    def _calc_dteec_1d(dtec_cell):
        """
        calculate DTEEC according to equation 04 and equation 05
        Args:
            dtec_cell: 1D array with daily threshold exceedance count
        """
        # Convert to a NumPy array and change NaN to 0
        dtec_np = np.nan_to_num(dtec_cell, nan=0)
        
        # Find the starts and ends of sequences (change NaNs to 0 before the diff operation)
        change = np.diff(np.concatenate(
            ([np.zeros((1,) + dtec_np.shape[1:]), dtec_np, np.zeros((1,) + dtec_np.shape[1:])]), axis=0), axis=0)
        starts = np.where(change == 1)
        ends = np.where(change == -1)
        
        # Calculate the middle points (as flat indices)
        middle_indices = (starts[0] + ends[0] - 1) // 2
        
        # Create an output array filled with NaNs
        events_np = np.full(dtec_cell.shape, np.nan)
        
        # Set the middle points to 1 (use flat indices to index into the 3D array)
        events_np[middle_indices] = 1
        
        return events_np
    