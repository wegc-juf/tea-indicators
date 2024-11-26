"""
Threshold Exceedance Amount (TEA) indicators Class implementation
Based on:
TODO: add reference to the paper
Equation numbers refer to Supplementary Notes
"""
import xarray as xr
import pandas as pd
import numpy as np
import datetime as dt

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
        self.daily_results = xr.Dataset()
        self.min_area = min_area
        self.gr_vars = None
        self.low_extreme = low_extreme
        self.testing = testing
        
        # Climatic Time Period (CTP) variables
        self.CTP = None
        self.CTP_freqs = {'annual': 'AS', 'seasonal': 'QS-DEC', 'WAS': 'AS-APR', 'ESS': 'AS-MAY', 'JJA': 'AS-JUN',
                          'DJF': 'AS-DEC', 'EWS': 'AS-NOV', 'monthly': 'MS'}
        self._overlap_ctps = ['EWS', 'DJF']
        self.CTP_months = {'WAS': [4, 5, 6, 7, 8, 9, 10], 'ESS': [5, 6, 7, 8, 9], 'EWS': [11, 12, 1, 2, 3],
                           'JJA': [6, 7, 8], 'DJF': [12, 1, 2]}
        self._CTP_resampler = None
        self._CTP_resample_sum = None
        self.CTP_results = xr.Dataset()
        
        if input_data_grid is not None:
            # set time index
            if 'days' in input_data_grid.dims:
                self.input_data_grid = self.input_data_grid.rename({'days': 'time'})
            elif 'time' in input_data_grid.dims:
                pass
            else:
                raise ValueError("Input data must have a 'days' or 'time' dimension")
            if input_data_grid.shape[-2:] != threshold_grid.shape:
                raise ValueError("Input data and threshold results must have the same area")
            if input_data_grid.shape[-2:] != area_grid.shape:
                raise ValueError("Input data and area results must have the same shape")
        
    def calc_DTEC(self):
        """
        calculate Daily Threshold Exceedance Count (equation 01)
        note that 0 values are stored as NaN for optimization
        """
        if self.daily_results['DTEM'] is None:
            self.calc_DTEM()
        dtem = self.daily_results.DTEM
        dtec = dtem.where(dtem.isnull(), 1)
        dtec.attrs = get_attrs(vname='DTEC')
        self.daily_results['DTEC'] = dtec
    
    def calc_DTEC_GR(self, min_area=None):
        """
        calculate Daily Threshold Exceedance Count (GR) (equation 03)
        note that 0 values are stored as NaN for optimization

        @param min_area: minimum area for a timestep to be considered as exceedance (same unit as area_grid)
        """
        if min_area is None:
            min_area = self.min_area
        if self.daily_results['DTEA_GR'] is None:
            self.calc_DTEA_GR()
        dtea_gr = self.daily_results.DTEA_GR
        dtec_gr = xr.where(dtea_gr >= min_area, 1, np.nan)
        dtec_gr.attrs = get_attrs(vname='DTEC_GR')
        self.daily_results['DTEC_GR'] = dtec_gr
    
    def calc_DTEEC(self):
        """
        calculate Daily Threshold Exceedance Event Count (equation 04)
        """
        if self.daily_results['DTEC'] is None:
            self.calc_DTEC()
        dtec = self.daily_results.DTEC
        
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
        self.daily_results['DTEEC'] = dteec
    
    def calc_DTEEC_GR(self):
        """
        calculate Daily Threshold Exceedance Event Count (GR) (equation 05)
        """
        if self.daily_results['DTEC_GR'] is None:
            self.calc_DTEC_GR()
            
        dtec_gr = self.daily_results.DTEC_GR
        dteec_np = self._calc_dteec_1d(dtec_cell=dtec_gr.values)
        dteec_gr = xr.DataArray(dteec_np, coords=dtec_gr.coords, dims=dtec_gr.dims)

        dteec_gr.attrs = get_attrs(vname='DTEEC_GR')
        self.daily_results['DTEEC_GR'] = dteec_gr

    def calc_DTEA(self):
        """
        calculate Daily Threshold Exceedance Area (equation 02)
        note that 0 values are stored as NaN for optimization
        """
        if self.daily_results['DTEC'] is None:
            self.calc_DTEC()
        dtec = self.daily_results.DTEC
        # equation 02_1 not needed (cells with TEC == 0 are already nan)
        # equation 02_2
        dtea = dtec * self.area_grid
        dtea.attrs = get_attrs(vname='DTEA')
        self.daily_results['DTEA'] = dtea
        
    def calc_DTEA_GR(self):
        """
        calculate Daily Threshold Exceedance Area (GR) (equation 06)
        """
        if self.daily_results['DTEA'] is None:
            self.calc_DTEA()
        dtea = self.daily_results.DTEA
        dtea_gr = dtea.sum(axis=(1, 2), skipna=True)
        dtea_gr = dtea_gr.rename('DTEA_GR')
        dtea_gr.attrs = get_attrs(vname='DTEA_GR')
        self.daily_results['DTEA_GR'] = dtea_gr
    
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
        self.daily_results['DTEM'] = dtem
        
    def calc_DTEM_max_gr(self):
        """
        calculate maximum DTEM for GR (equation 09)
        """
        if self.daily_results['DTEM'] is None:
            self.calc_DTEM()
        if self.daily_results['DTEC_GR'] is None:
            self.calc_DTEC_GR()
        dtem = self.daily_results.DTEM
        dtem_max = dtem.max(dim=self.threshold_grid.dims)
        dtem_max = dtem_max.where(self.daily_results.DTEC_GR == 1)
        dtem_max = dtem_max.rename('DTEM_max_gr')
        dtem_max.attrs = get_attrs(vname='DTEM_Max')
        self.daily_results['DTEM_max_gr'] = dtem_max

    def calc_DTEM_GR(self):
        """
        calculate Daily Threshold Exceedance Magnitude (GR) (equation 08)
        """
        if self.daily_results['DTEA_GR'] is None:
            self.calc_DTEA_GR()
        if self.daily_results['DTEM'] is None:
            self.calc_DTEM()
        if self.daily_results['DTEC_GR'] is None:
            self.calc_DTEC_GR()
        dtea_gr = self.daily_results.DTEA_GR
        dtem = self.daily_results.DTEM
        dtec_gr = self.daily_results.DTEC_GR
        area_fac = self.area_grid / dtea_gr
        dtem_gr = (dtem * area_fac).sum(axis=(1, 2), skipna=True)
        dtem_gr = dtem_gr.where(dtec_gr == 1)
        dtem_gr = dtem_gr.rename(f'{dtem.name}_GR')
        dtem_gr.attrs = get_attrs(vname='DTEM_GR')
        self.daily_results['DTEM_GR'] = dtem_gr
    
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
    
    def save_daily_results(self, filepath):
        """
        save all variables to filepath
        """
        self.daily_results.to_netcdf(filepath)
    
    def load_daily_results(self, filepath):
        """
        load all variables from filepath
        """
        self.daily_results = xr.open_dataset(filepath)
    
    def update_min_area(self, min_area):
        """
        update the minimum area for a timestep to be considered as exceedance
        """
        self.min_area = min_area
        self.calc_DTEC_GR()
        self.calc_DTEM_GR()
        self.calc_DTEM_max_gr()
        self.calc_DTEEC_GR()
        
    # ### Climatic Time Period (CTP) functions ###
    def set_ctp(self, ctp):
        """
        set Climatic Time Period (CTP)

        args:
            ctp: Climatic Time Period (CTP) to resample to
                allowed values: 'annual', 'seasonal', 'WAS', 'ESS', 'JJA', 'DJF', 'EWS', 'monthly'
                'WAS': warm season (April to October)
                'ESS': extended summer season (May to September)
                'JJA': summer season (June to August)
                'DJF': winter season (December to February)
                'EWS': extended winter season (November to March)
        """
        self.CTP = ctp
    
    def calc_event_frequency(self):
        """
        calculate event frequency (equation 11 and equation 12)
        """
        if self.CTP is None:
            raise ValueError("CTP must be set before calculating event frequency")
        if self.daily_results['DTEEC'] is None:
            self.calc_DTEEC()
        if self.daily_results['DTEEC_GR'] is None:
            self.calc_DTEEC_GR()
            
        if self._CTP_resample_sum is None:
            self._resample_to_CTP()
            
        ef = self._CTP_resample_sum.DTEEC
        ef_gr = self._CTP_resample_sum.DTEEC_GR
        
        ef.attrs = get_attrs(vname='EF')
        self.CTP_results['EF'] = ef
        ef_gr.attrs = get_attrs(vname='EF_GR')
        self.CTP_results['EF_GR'] = ef_gr

        
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

    def _filter_and_shift_CTP(self, ctp=None):
        """
        keep only values according to Climatic Time Period (CTP) definition
        """
        if ctp is not None:
            self.CTP = ctp
            
        if self.CTP not in self.CTP_months:
            # no filtering necessary
            self._daily_results_filtered = self.daily_results.copy()
            return
        
        months = self.CTP_months[self.CTP]
        self._daily_results_filtered = self.daily_results.sel(time=self.daily_results.time.dt.month.isin(months))
        
    def _resample_to_CTP(self, ctp=None):
        """
        resample daily results to Climatic Time Period (CTP)
        
        """
        if ctp is not None:
            self.set_ctp(ctp)
        elif self.CTP is None:
            raise ValueError("CTP must be set before resampling")
        
        self._filter_and_shift_CTP()
        self._CTP_resampler = self._daily_results_filtered.resample(time=self.CTP_freqs[self.CTP])
        self._CTP_resample_sum = self._CTP_resampler.sum('time')
        if self.CTP in self._overlap_ctps:
            # remove first and last year
            self._CTP_resample_sum = self._CTP_resample_sum.isel(time=slice(1, -1))
        del self._daily_results_filtered
        
    