"""
Threshold Exceedance Amount (TEA) indicators Class implementation
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
from scripts.general_stuff.TEA_logger import logger


class TEAIndicators:
    """
    Class to calculate TEA indicators
    """
    
    def __init__(self, input_data_grid=None, threshold=None, min_area=1., area_grid=None,
                 low_extreme=False,
                 unit='', mask=None, apply_mask=True, ctp=None):
        """
        Initialize TEAIndicators object
        Args:
            input_data_grid: gridded input data (e.g. temperature, precipitation)
            threshold: either gridded threshold values (xarray DataArray) or a constant threshold value (int, float)
            area_grid: results containing the area of each results cell, if None, area is assumed to be 1 for each cell
                       nan values mask out the corresponding results cells
            min_area: minimum area for a timestep to be considered as exceedance (same unit as area_grid). Default: 1
            low_extreme: set to True if values below the threshold are considered as extreme events. Default: False
            unit: unit of the input data. Default: ''
            mask: mask grid for input data containing nan values for cells that should be masked. Default: None
            ctp: Climatic Time Period (CTP) to resample to. For allowed values see set_ctp method. Default: None
        """
        if threshold is not None and isinstance(threshold, (int, float)):
            threshold = xr.full_like(input_data_grid[0], threshold)
        self.threshold_grid = threshold
        
        self.area_grid = None
        if area_grid is None:
            if input_data_grid is not None:
                self._create_area_grid(input_data_grid)
            elif threshold is not None:
                self.area_grid = xr.ones_like(self.threshold_grid)
        else:
            self.area_grid = area_grid
            
        self.mask = mask
        self.apply_mask = apply_mask
        self.input_data_grid = None
        self.daily_results = xr.Dataset()
        self._daily_results_filtered = None
        self._min_area = min_area
        self._low_extreme = low_extreme
        self.unit = unit
        
        self._calc_grid = True
        self._calc_gr = True
        
        if input_data_grid is not None:
            if self.threshold_grid is None:
                raise ValueError("Threshold grid must be set together with input data")
            self._set_input_data_grid(input_data_grid)
        
        # size of whole GeoRegion
        if area_grid is not None:
            self.gr_size = area_grid.sum().values
        else:
            self.gr_size = None
        
        # Climatic Time Period (CTP) variables
        self.CTP = ctp
        self.CTP_freqs = {'annual': 'AS', 'seasonal': 'QS-DEC', 'WAS': 'AS-APR', 'ESS': 'AS-MAY', 'JJA': 'AS-JUN',
                          'DJF': 'AS-DEC', 'EWS': 'AS-NOV', 'monthly': 'MS'}
        self._overlap_ctps = ['EWS', 'DJF']
        self.CTP_months = {'WAS': [4, 5, 6, 7, 8, 9, 10], 'ESS': [5, 6, 7, 8, 9], 'EWS': [11, 12, 1, 2, 3],
                           'JJA': [6, 7, 8], 'DJF': [12, 1, 2]}
        self._CTP_resampler = None
        self._CTP_resample_sum = None
        self.CTP_results = xr.Dataset()
        
        # Decadal data
        self.decadal_results = xr.Dataset()
        
        # Amplification factors
        self._cc_mean = None
        self.cc_period = None
        self._ref_mean = None
        self.ref_period = None
        self.amplification_factors = xr.Dataset()
        
        self.null_val = 0
    
    def _crop_to_rect(self, lat_range, lon_range):
        """
        crop all grids to a rectangular area
        Args:
            lat_range: Latitude range (min, max)
            lon_range: Longitude range (min, max)

        Returns:

        """
        self.input_data_grid = self.input_data_grid.sel(lat=slice(lat_range[1], lat_range[0]),
                                                        lon=slice(lon_range[0], lon_range[1]))
        self.area_grid = self.area_grid.sel(lat=slice(lat_range[1], lat_range[0]),
                                            lon=slice(lon_range[0], lon_range[1]))
        self.mask = self.mask.sel(lat=slice(lat_range[1], lat_range[0]),
                                  lon=slice(lon_range[0], lon_range[1]))
        self.threshold_grid = self.threshold_grid.sel(lat=slice(lat_range[1], lat_range[0]),
                                                      lon=slice(lon_range[0], lon_range[1]))
    
    def _crop_to_mask_extents(self):
        """
        crop all grids to the mask extents
        """
        mask = self.mask
        idx_with_data = np.where(mask > 0)
        lon_min = mask.lon[idx_with_data[1]].min().values
        lon_max = mask.lon[idx_with_data[1]].max().values
        lat_min = mask.lat[idx_with_data[0]].min().values
        lat_max = mask.lat[idx_with_data[0]].max().values
        lat_range = [lat_min, lat_max]
        lon_range = [lon_min, lon_max]
        self._crop_to_rect(lat_range, lon_range)
        
    def _set_input_data_grid(self, input_data_grid):
        """
        set input data grid
        Args:
            input_data_grid: gridded input data (e.g. temperature, precipitation)
        """
        if self.mask is not None and self.apply_mask:
            self.input_data_grid = input_data_grid.where(self.mask > 0)
            self._crop_to_mask_extents()
        else:
            self.input_data_grid = input_data_grid
            
        if self.input_data_grid is not None:
            # set time index
            if 'days' in input_data_grid.dims:
                self.input_data_grid = self.input_data_grid.rename({'days': 'time'})
            elif 'time' in input_data_grid.dims:
                pass
            else:
                raise ValueError("Input data must have a 'days' or 'time' dimension")
            if self.input_data_grid.shape[-2:] != self.threshold_grid.shape:
                raise ValueError("Input data and threshold results must have the same area")
            if self.input_data_grid.shape[-2:] != self.area_grid.shape:
                raise ValueError("Input data and area results must have the same shape")
        
    def _calc_DTEC(self):
        """
        calculate Daily Threshold Exceedance Count (equation 01)
        """
        if self.daily_results['DTEM'] is None:
            self._calc_DTEM()
        dtem = self.daily_results.DTEM
        dtec = xr.where(dtem > 0, 1, dtem)
        dtec.attrs = get_attrs(vname='DTEC')
        self.daily_results['DTEC'] = dtec
    
    def _calc_DTEC_GR(self, min_area=None):
        """
        calculate Daily Threshold Exceedance Count (GR) (equation 03)

        @param min_area: minimum area for a timestep to be considered as exceedance (same unit as area_grid)
        """
        if min_area is None:
            min_area = self._min_area
        if 'DTEA_GR' not in self.daily_results:
            self._calc_DTEA_GR()
        dtea_gr = self.daily_results.DTEA_GR
        dtec_gr = xr.where(dtea_gr >= min_area, 1, self.null_val)
        dtec_gr.attrs = get_attrs(vname='DTEC_GR')
        self.daily_results['DTEC_GR'] = dtec_gr
    
    def _calc_DTEEC(self):
        """
        calculate Daily Threshold Exceedance Event Count (equation 04)
        """
        if self.daily_results['DTEC'] is None:
            self._calc_DTEC()
        dtec = self.daily_results.DTEC
        
        dteec = xr.full_like(dtec, np.nan)
        
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
    
    def _calc_DTEEC_GR(self):
        """
        calculate Daily Threshold Exceedance Event Count (GR) (equation 05)
        """
        if 'DTEC_GR' not in self.daily_results:
            self._calc_DTEC_GR()
            
        dtec_gr = self.daily_results.DTEC_GR
        dteec_np = self._calc_dteec_1d(dtec_cell=dtec_gr.values)
        dteec_gr = xr.DataArray(dteec_np, coords=dtec_gr.coords, dims=dtec_gr.dims)

        dteec_gr.attrs = get_attrs(vname='DTEEC_GR')
        self.daily_results['DTEEC_GR'] = dteec_gr

    def _calc_DTEA(self):
        """
        calculate Daily Threshold Exceedance Area (equation 02)
        """
        if 'DTEC' not in self.daily_results:
            self._calc_DTEC()
        dtec = self.daily_results.DTEC
        # equation 02_1 not needed (cells with TEC == 0 are already nan)
        # equation 02_2
        dtea = dtec * self.area_grid
        dtea.attrs = get_attrs(vname='DTEA')
        self.daily_results['DTEA'] = dtea
        
    def _calc_DTEA_GR(self, relative=False):
        """
        calculate Daily Threshold Exceedance Area (GR) (equation 06)
        
        Args:
            relative: calculate area relative to full GR area. Default: False
        """
        if 'DTEA' not in self.daily_results:
            self._calc_DTEA()
        dtea = self.daily_results.DTEA
        dtea_gr = dtea.sum(axis=(1, 2), skipna=True)
        if relative:
            dtea_gr = dtea_gr / self.gr_size
            dtea_gr.attrs = get_attrs(vname='DTEA_GR', data_unit='%')
        else:
            dtea_gr.attrs = get_attrs(vname='DTEA_GR')
        dtea_gr = dtea_gr.rename('DTEA_GR')
        self.daily_results['DTEA_GR'] = dtea_gr
    
    def _calc_DTEM(self):
        """
        calculate Daily Threshold Exceedance Magnitude (equation 07)
        """
        if self._low_extreme:
            dtem = self.threshold_grid - self.input_data_grid
        else:
            dtem = self.input_data_grid - self.threshold_grid
        dtem = xr.where(dtem <= 0, 0, dtem)
        dtem.attrs = get_attrs(vname='DTEM', data_unit=self.unit)
        self.daily_results['DTEM'] = dtem
        
    def _calc_DTEM_Max_GR(self):
        """
        calculate maximum DTEM for GR (equation 09)
        """
        if 'DTEM' not in self.daily_results:
            self._calc_DTEM()
        if 'DTEC_GR' not in self.daily_results:
            self._calc_DTEC_GR()
        dtem = self.daily_results.DTEM
        dtem_max = dtem.max(dim=self.area_grid.dims)
        dtem_max = dtem_max.where(self.daily_results.DTEC_GR == 1, self.null_val)
        dtem_max.attrs = get_attrs(vname='DTEM_Max_GR', data_unit=self.unit)
        self.daily_results['DTEM_Max_GR'] = dtem_max

    def _calc_DTEM_GR(self):
        """
        calculate Daily Threshold Exceedance Magnitude (GR) (equation 08)
        """
        if 'DTEA_GR' not in self.daily_results:
            self._calc_DTEA_GR()
        if 'DTEM' not in self.daily_results:
            self._calc_DTEM()
        if 'DTEC_GR' not in self.daily_results:
            self._calc_DTEC_GR()
        dtea_gr = self.daily_results.DTEA_GR
        dtem = self.daily_results.DTEM
        dtec_gr = self.daily_results.DTEC_GR
        # replace 0 values with nan to avoid division by 0
        dtec_gr = dtec_gr.where(dtec_gr > 0, np.nan)
        area_fac = self.area_grid / dtea_gr
        dtem_gr = (dtem * area_fac).sum(axis=(1, 2), skipna=True)
        dtem_gr = dtem_gr.where(dtec_gr == 1, self.null_val)
        dtem_gr = dtem_gr.rename(f'{dtem.name}_GR')
        dtem_gr.attrs = get_attrs(vname='DTEM_GR', data_unit=self.unit)
        dtema_gr = dtem_gr * dtea_gr
        dtema_gr.attrs = get_attrs(vname='DTEMA_GR', data_unit=self.unit)
        self.daily_results['DTEM_GR'] = dtem_gr
        self.daily_results['DTEMA_GR'] = dtema_gr
    
    def calc_daily_basis_vars(self, grid=True, gr=True):
        """
        calculate all daily basis variables
        
        Args:
            grid: calculate grid variables. Default: True
            gr: calculate GR variables. Default: True
        """
        self._calc_grid = False
        self._calc_gr = False
        if grid:
            self._calc_DTEM()
            self._calc_DTEC()
            self._calc_DTEA()
            self._calc_DTEEC()
            self._calc_grid = True
        if gr:
            self._calc_DTEA_GR()
            self._calc_DTEC_GR()
            self._calc_DTEM_GR()
            self._calc_DTEM_Max_GR()
            self._calc_DTEEC_GR()
            self._calc_gr = True
    
    def save_daily_results(self, filepath):
        """
        save all variables to filepath
        """
        with warnings.catch_warnings():
            # ignore warnings due to nan multiplication
            warnings.simplefilter("ignore")
            self.daily_results.to_netcdf(filepath)
    
    def load_daily_results(self, filepath):
        """
        load all variables from filepath
        """
        self.daily_results = xr.open_dataset(filepath)
        self.unit = self.daily_results.DTEM.attrs['units']
        
    def set_daily_results(self, daily_results):
        """
        set daily results
        """
        self.daily_results = daily_results
        self.unit = self.daily_results.DTEM.attrs['units']
    
    def get_daily_results(self, grid=True, gr=True):
        """
        get daily basis variable results

        Args:
            grid: get gridded results. Default: True
            gr: get GR results. Default: True
        """
        gr_vars = [var for var in self.daily_results.data_vars if 'GR' in var]
        grid_vars = [var for var in self.daily_results.data_vars if 'GR' not in var]
        if not grid:
            return self.daily_results.drop_vars(grid_vars)
        if not gr:
            return self.daily_results.drop_vars(gr_vars)
        else:
            return self.daily_results
    
    def update_min_area(self, min_area):
        """
        update the minimum area for a timestep to be considered as exceedance
        """
        self._min_area = min_area
        self._calc_DTEC_GR()
        self._calc_DTEM_GR()
        self._calc_DTEM_Max_GR()
        self._calc_DTEEC_GR()
        
    # ### Climatic Time Period (CTP) functions ###
    def _set_ctp(self, ctp):
        """
        set annual Climatic Time Period (CTP)

        args:
            ctp: Climatic Time Period (CTP) to resample to
                allowed values: 'annual', 'seasonal', 'WAS', 'ESS', 'JJA', 'DJF', 'EWS', 'monthly'
                'WAS': warm season (April to October)
                'ESS': extended summer season (May to September)
                'JJA': summer season (June to August)
                'DJF': winter season (December to February)
                'EWS': extended winter season (November to March)
        """
        valid_dec_periods = ['annual', 'seasonal', 'monthly', 'WAS', 'ESS', 'EWS', 'JJA', 'DJF']
        if ctp not in valid_dec_periods:
            raise ValueError(f"Invalid CTP: {ctp}. Allowed values: {valid_dec_periods}")
        self.CTP = ctp
        ctp_attrs = get_attrs(vname='CTP_global_attrs', period=ctp)
        # TODO: add CF-Convention compatible attributes...
        self.CTP_results.attrs = ctp_attrs
    
    def _calc_event_frequency(self):
        """
        calculate event frequency (equation 11 and equation 12)
        """
        if self.CTP is None:
            raise ValueError("CTP must be set before calculating event frequency")
        if 'DTEEC' not in self.daily_results:
            self._calc_DTEEC()
        if 'DTEEC_GR' not in self.daily_results:
            self._calc_DTEEC_GR()
            
        if self._CTP_resample_sum is None:
            self._resample_to_CTP()
            
        if 'DTEEC' in self._CTP_resample_sum:
            # # process grid data
            ef = self._CTP_resample_sum.DTEEC
            ef = ef.where(ef.notnull(), 0)
            ef.attrs = get_attrs(vname='EF')
            self.CTP_results['EF'] = ef
            
        # # process GR data
        ef_gr = self._CTP_resample_sum.DTEEC_GR
        ef_gr = ef_gr.where(ef_gr.notnull(), 0)
        
        ef_gr.attrs = get_attrs(vname='EF_GR')
        self.CTP_results['EF_GR'] = ef_gr
    
    def _calc_supplementary_event_vars(self):
        """
        calculate supplementary event variables (equation 13)
        """
        if 'EF' not in self.CTP_results and 'EF_GR' not in self.CTP_results:
            self._calc_event_frequency()
        
        doy = [pd.Timestamp(dy).day_of_year for dy in self._daily_results_filtered.time.values]
        self._daily_results_filtered.coords['doy'] = ('time', doy)
        
        if 'DTEEC' in self._daily_results_filtered:
            # # process grid data
            event_doy = self._daily_results_filtered.doy.where(self._daily_results_filtered.DTEEC > 0)
            resampler = event_doy.resample(time=self.CTP_freqs[self.CTP])
        
            # equation 13_1
            doy_first = resampler.min('time')
            doy_first.attrs = get_attrs(vname='doy_first')
            
            # equation 13_2
            doy_last = resampler.max('time')
            doy_last.attrs = get_attrs(vname='doy_last')
            
            # equation 13_3
            aep = (doy_last - doy_first + 1) / 30.5
            # set aep values where EF == 0 to 0
            aep = xr.where(self.CTP_results.EF == 0, 0, aep)
            aep.attrs = get_attrs(vname='AEP')
            
            self.CTP_results['doy_first'] = doy_first
            self.CTP_results['doy_last'] = doy_last
            self.CTP_results['AEP'] = aep
            # indent

        # # process GR data
        event_doy_gr = self._daily_results_filtered.doy.where(self._daily_results_filtered.DTEEC_GR > 0)
        resampler_gr = event_doy_gr.resample(time=self.CTP_freqs[self.CTP])
        
        # equation 13_4
        doy_first_gr = resampler_gr.min('time')
        
        # equation 13_5
        doy_last_gr = resampler_gr.max('time')
        
        # equation 13_6
        aep_gr = (doy_last_gr - doy_first_gr + 1) / 30.5
        # set aep values where EF == 0 to 0
        aep_gr = xr.where(self.CTP_results.EF_GR == 0, 0, aep_gr)
        
        doy_first_gr.attrs = get_attrs(vname='doy_first_GR')
        doy_last_gr.attrs = get_attrs(vname='doy_last_GR')
        aep_gr.attrs = get_attrs(vname='AEP_GR')
        
        self.CTP_results['doy_first_GR'] = doy_first_gr
        self.CTP_results['doy_last_GR'] = doy_last_gr
        self.CTP_results['AEP_GR'] = aep_gr
    
    def _calc_event_duration(self):
        """
        calculate event duration (equation 14 and equation 15)
        """
        if 'EF' not in self.CTP_results and 'EF_GR' not in self.CTP_results:
            self._calc_event_frequency()
        
        if 'DTEC' in self._CTP_resample_sum:
            # # process grid data
            
            # equation 14_2
            ed = self._CTP_resample_sum.DTEC
            ed.attrs = get_attrs(vname='ED')
            self.CTP_results['ED'] = ed
            
            ef = self.CTP_results['EF']
            
            # calc average event duration (equation 14_1)
            ed_avg = ed / ef
            ed_avg = xr.where(ef == 0, 0, ed_avg)
            ed_avg.attrs = get_attrs(vname='ED_avg')
            self.CTP_results['ED_avg'] = ed_avg
            # indent
        
        # # process GR data
        # equation 15_2
        ed_gr = self._CTP_resample_sum.DTEC_GR
        ed_gr.attrs = get_attrs(vname='ED_GR')
        self.CTP_results['ED_GR'] = ed_gr
        
        ef_gr = self.CTP_results['EF_GR']
        
        # calc average event duration (equation 15_1)
        ed_avg_gr = ed_gr / ef_gr
        ed_avg_gr = xr.where(ef_gr == 0, 0, ed_avg_gr)
        ed_avg_gr.attrs = get_attrs(vname='ED_avg_GR')
        self.CTP_results['ED_avg_GR'] = ed_avg_gr
    
    def _calc_exceedance_magnitude(self):
        """
        calculate average (EM_avg) and cumulative (tEX=EM) exceedance magnitude (equation 17 and equation 18),
        median exceedance magnitude (equation 19), and maximum exceedance magnitude (equation 20)
        """
        
        if 'ED' not in self.CTP_results and 'ED_GR' not in self.CTP_results:
            self._calc_event_duration()
            
        if 'DTEM' in self._CTP_resample_sum:
            # process grid data
        
            # equation 17_2
            em = self._CTP_resample_sum.DTEM
            
            # calc average exceedance magnitude (equation 17_1)
            ed = self.CTP_results.ED
            em_avg = em / ed
            em_avg = xr.where(ed == 0, 0, em_avg)
            
            em.attrs = get_attrs(vname='EM', data_unit=self.unit)
            em_avg.attrs = get_attrs(vname='EM_avg', data_unit=self.unit)
            self.CTP_results['EM'] = em
            self.CTP_results['EM_avg'] = em_avg
            
            # calc median exceedance magnitude (equation 19_1)
            em_avg_med = self._CTP_resample_median.DTEM
            em_avg_med.attrs = get_attrs(vname='EM_avg_Md', data_unit=self.unit)

            # equation 19_2
            em_med = self.CTP_results.ED * em_avg_med
            em_med.attrs = get_attrs(vname='EM_Md', data_unit=self.unit)
            
            self.CTP_results['EM_avg_Md'] = em_avg_med
            self.CTP_results['EM_Md'] = em_med
            
            # indent
            
        # # process GR data
        # equation 18_2
        em_gr = self._CTP_resample_sum.DTEM_GR
    
        # calc average exceedance magnitude (equation 18_1)
        ed_gr = self.CTP_results.ED_GR
        em_avg_gr = em_gr / ed_gr
        em_avg_gr = xr.where(ed_gr == 0, 0, em_avg_gr)
        
        em_gr.attrs = get_attrs(vname='EM_GR', data_unit=self.unit)
        em_avg_gr.attrs = get_attrs(vname='EM_avg_GR', data_unit=self.unit)
        
        self.CTP_results['EM_GR'] = em_gr
        self.CTP_results['EM_avg_GR'] = em_avg_gr
        
        # calc median exceedance magnitude (equation 19_3)
        em_avg_gr_med = self._CTP_resample_median.DTEM_GR
        em_avg_gr_med.attrs = get_attrs(vname='EM_avg_GR_Md', data_unit=self.unit)
        self.CTP_results['EM_avg_GR_Md'] = em_avg_gr_med

        # equation 19_4
        em_gr_med = self.CTP_results.ED_GR * em_avg_gr_med
        em_gr_med.attrs = get_attrs(vname='EM_GR_Md', data_unit=self.unit)
        self.CTP_results['EM_GR_Md'] = em_gr_med
        
        # calc maximum exceedance magnitude (equation 20_2)
        em_gr_max = self._CTP_resample_sum.DTEM_Max_GR
        em_gr_max.attrs = get_attrs(vname='EM_Max_GR', data_unit=self.unit)
        self.CTP_results['EM_Max_GR'] = em_gr_max
        
        # calc average maximum exceedance magnitude (equation 20_1)
        em_gr_avg_max = em_gr_max / self.CTP_results.ED_GR
        em_gr_avg_max.attrs = get_attrs(vname='EM_avg_Max_GR', data_unit=self.unit)
        self.CTP_results['EM_avg_Max_GR'] = em_gr_avg_max
    
    def _calc_annual_total_events_extremity(self):
        """
        calculate annual total events extremity (equation 21_3)
        """
        # equation 21_3
        tex = self._CTP_resample_sum.DTEMA_GR
        tex.attrs = get_attrs(vname='TEX_GR', data_unit=self.unit)
        self.CTP_results['TEX_GR'] = tex
    
    def _calc_total_events_extremity(self, f, d=None, m=None, a=None, s=None):
        """
        calculate total events extremity (equation 21_4)
        
        Args:
            f: event frequency
            d: average event duration
            m: average exceedance magnitude
            a: average exceedance area
            s: event severity
        Either f, d, m, and a, or f and s must be provided
        """
        # equation 21_4
        if s is None:
            if d is None or m is None or a is None:
                raise ValueError("Either f, d, m, and a, or f and s must be provided")
            s = self._calc_event_severity(d=d, m=m, a=a)
        tex = f * s
        tex.attrs = get_attrs(vname='TEX_GR', data_unit=self.unit)
        tex.rename('TEX_GR')
        return tex
    
    def _calc_exceedance_area(self):
        """
        calculate exceedance area (equation 21_1)
        """
        if self.CTP_results['TEX_GR'] is None:
            self._calc_annual_total_events_extremity()
        if self.CTP_results['EM_GR'] is None:
            self._calc_exceedance_magnitude()
        
        # equation 21_1
        ea_avg = self.CTP_results.TEX_GR / self.CTP_results.EM_GR
        ea_avg = xr.where(self.CTP_results.EM_GR == 0, 0, ea_avg)
        ea_avg.attrs = get_attrs(vname='EA_avg_GR')
        self.CTP_results['EA_avg_GR'] = ea_avg
    
    def _calc_annual_event_severity(self):
        """
        calculate event severity (equation 21_2)
        """
        if self.CTP_results['EA_avg_GR'] is None:
            self._calc_exceedance_area()
        if self.CTP_results['EM_avg_GR'] is None:
            self._calc_exceedance_magnitude()
        if self.CTP_results['ED_avg_GR'] is None:
            self._calc_event_duration()
        
        # equation 21_2
        es_avg = self._calc_event_severity(self.CTP_results.ED_avg_GR, self.CTP_results.EM_avg_GR,
                                           self.CTP_results.EA_avg_GR)
        self.CTP_results['ES_avg_GR'] = es_avg
    
    def _calc_annual_exceedance_heat_content(self):
        """
        calculate annual exceedance heat content (equation 22)
        """
        if self.CTP_results['ED_avg_GR'] is None:
            self._calc_event_duration()
        if self.CTP_results['TEX_GR'] is None:
            self._calc_annual_total_events_extremity()
        if self.CTP_results['ES_avg_GR'] is None:
            self._calc_annual_event_severity()
        
        # equation 22
        H_AEHC_avg_GR, H_AEHC_GR = self._calc_exceedance_heat_content(self.CTP_results.ES_avg_GR,
                                                                      self.CTP_results.ED_avg_GR,
                                                                      self.CTP_results.TEX_GR)
        H_AEHC_avg_GR.attrs = get_attrs(vname='H_AEHC_avg_GR')
        H_AEHC_GR.attrs = get_attrs(vname='H_AEHC_GR')
        self.CTP_results['H_AEHC_avg_GR'] = H_AEHC_avg_GR
        self.CTP_results['H_AEHC_GR'] = H_AEHC_GR
    
    def _calc_event_severity(self, d, m, a):
        """
        calculate event severity (equation 21_2)
        
        Args:
            d: average event duration
            m: average exceedance magnitude
            a: average exceedance area
        """
        # equation 21_5
        es_avg = d * m * a
        es_avg.attrs = get_attrs(vname='ES_avg_GR', data_unit=self.unit)
        es_avg.rename('ES_avg_GR')
        return es_avg
    
    def _calc_cumulative_events_duration(self, f, d):
        """
        calculate cumulative events duration (equation 17_2)
        
        Args:
            f: event frequency
            d: average event duration
        """
        ced = f * d
        ced.attrs = get_attrs(vname='ED')
        ced.rename('ED')
        return ced
    
    def _calc_temporal_events_extremity(self, f=None, d=None, ed=None, m=None):
        """
        calculate temporal events extremity (equation 18_2)
        
        Args:
            f: event frequency
            d: average event duration
            ed: cumulative events duration
            m: average exceedance magnitude
        either f, d, and m, or ed and m must be provided
        """
        if ed is None:
            if f is None or d is None or m is None:
                raise ValueError("Either f, d, and m, or ed and m must be provided")
            ed = self._calc_cumulative_events_duration(f, d)
        tem = ed * m
        tem.attrs = get_attrs(vname='EM', data_unit=self.unit)
        tem.rename('EM')
        return tem
    
    @staticmethod
    def _calc_exceedance_heat_content(s_avg, d_avg, tex):
        """
        calculate exceedance heat content (equation 22)
        
        Args:
            s_avg: average event severity
            d_avg: average event duration
            tex: total events extremity
            
        Returns:
            H_AEHC_avg_GR: average daily atmospheric boundary layer exceedance heat content
            H_AEHC_GR: cumulative atmospheric boundary layer exceedance heat content
        """
        
        # approximate atmospheric boundary layer daily exceedance heat energy uptake capacity
        # [PJ/(areal °C day)] (equation 22_3)
        ct_abl = 0.1507
        
        H_AEHC_avg_GR = ct_abl * s_avg / d_avg
        H_AEHC_GR = ct_abl * tex
        return H_AEHC_avg_GR, H_AEHC_GR
    
    def calc_annual_CTP_indicators(self, ctp=None, drop_daily_results=False):
        """
        calculate all annual Climatic Time Period (CTP) indicators
        
        Args:
            ctp: Climatic Time Period (CTP) to resample to
                allowed values: 'annual', 'seasonal', 'WAS', 'ESS', 'JJA', 'DJF', 'EWS', 'monthly'
                'WAS': warm season (April to October)
                'ESS': extended summer season (May to September)
                'JJA': summer season (June to August)
                'DJF': winter season (December to February)
                'EWS': extended winter season (November to March)
            drop_daily_results: delete daily results after calculation
        """
        if ctp is not None:
            self._set_ctp(ctp)
        self._calc_event_frequency()
        self._calc_supplementary_event_vars()
        self._calc_event_duration()
        self._calc_exceedance_magnitude()
        self._calc_annual_total_events_extremity()
        self._calc_exceedance_area()
        self._calc_annual_event_severity()
        self._calc_annual_exceedance_heat_content()
        if drop_daily_results:
            del self._daily_results_filtered
            del self.daily_results
        ctp_attrs = get_attrs(vname='CTP', period=self.CTP)
        self.CTP_results['time'].attrs = ctp_attrs
        self.CTP_results.attrs['CTP'] = self.CTP

    def save_CTP_results(self, filepath):
        """
        save all CTP results to filepath
        """
        with warnings.catch_warnings():
            # ignore warnings due to nan multiplication
            warnings.simplefilter("ignore")
            self.CTP_results.to_netcdf(filepath)
        
    def load_CTP_results(self, filepath):
        """
        load all CTP results from filepath
        """
        self.CTP_results = xr.open_mfdataset(filepath)
        if 'units' not in self.CTP_results.EM_avg.attrs:
            logger.warning("No unit attribute found in CTP results. Please set the unit attribute manually.")
        else:
            self.unit = self.CTP_results.EM_avg.attrs['units']
    
    def get_CTP_results(self, grid=True, gr=True):
        """
        get CTP results
        
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
        
    # ### Decadal mean functions ###
    
    def calc_decadal_indicators(self, calc_spread=False, drop_annual_results=True):
        """
        calculate decadal mean for all CTP indicators
        equation 23_1 and equation 23_2
        
        Args:
            calc_spread: calculate spread estimators (equation 25)
            drop_annual_results: delete annual results after calculation
        """
        if self.CTP_results is None:
            raise ValueError("CTP results must be calculated before calculating decadal mean")
        
        self._calc_decadal_mean()
        self._calc_decadal_compound_vars()
        # TODO: optional calculation of AEHC
        if calc_spread:
            self._calc_spread_estimators()
        self.CTP = self.CTP_results.attrs['CTP']
        self.decadal_results['time'].attrs = get_attrs(vname='decadal', period=self.CTP)
        self.decadal_results.attrs = get_attrs(vname='decadal_global_attrs', period=self.CTP)
        self.decadal_results.attrs['CTP'] = self.CTP

        if drop_annual_results:
            del self.CTP_results
            del self._CTP_resampler
            del self._CTP_resample_sum
    
    def save_decadal_results(self, filepath):
        """
        save all decadal results to filepath
        """
        with warnings.catch_warnings():
            # ignore warnings due to nan multiplication
            warnings.simplefilter("ignore")
            self.decadal_results.to_netcdf(filepath)
    
    def load_decadal_results(self, filepath):
        """
        load all decadal results from filepath
        """
        self.decadal_results = xr.open_dataset(filepath)
        if 'CTP' in self.decadal_results.attrs:
            self.CTP = self.decadal_results.attrs['CTP']
        
    def _calc_decadal_mean(self):
        """
        calculate decadal mean for all basic CTP indicators (equation 23_1)
        """
        # check if CTP_results contain at least a decade of data
        if len(self.CTP_results.time) < 10:
            raise ValueError("For calculating decadal results, CTP results and daily data must contain at least a "
                             "decade of data")
        for var in self.CTP_results.data_vars:
            if self.CTP_results[var].attrs['metric_type'] == 'basic':
                self.decadal_results[var] = self.CTP_results[var].rolling(time=10, center=True, min_periods=1).mean(
                    skipna=True)
                # set first and last 5 years to nan
                self.decadal_results[var][:5] = np.nan
                self.decadal_results[var][-4:] = np.nan
                self.decadal_results[var].attrs = get_attrs(vname=var, dec=True, data_unit=self.unit)
        
    def _calc_decadal_compound_vars(self):
        """
        calculate decadal values for compound variables (equation 23_2)
        
        Returns:

        """
        self.decadal_results = self._calc_compound_vars(self.decadal_results)
        
    def _calc_compound_vars(self, data):
        """
        calculate values for compound variables (equation 23_2)
        
        Args:
            data: Xarray dataset containing the basic indicators EF, ED, EM, and EA
        
        Returns:
            data: Xarray dataset containing the additional compound indicators ED, EM, EM_Md, EM_Max, ES_avg, and TEX
        """
        
        # calculate cumulative events duration (cf. equation 14_2)
        ED = self._calc_cumulative_events_duration(f=data['EF'], d=data['ED_avg'])
        ED.attrs = get_attrs(vname='ED', dec=True, data_unit=self.unit)
        data['ED'] = ED
        
        # calculate temporal events extremity tEX (equals cumulative exceedance magnitude EM) (cf. equation 17_2)
        EM = self._calc_temporal_events_extremity(ed=data['ED'], m=data['EM_avg'])
        EM.attrs = get_attrs(vname='EM', dec=True, data_unit=self.unit)
        data['EM'] = EM
        
        # calculate cumulative median exceedance magnitude (cf. equation 19_2)
        EM_Md = self._calc_temporal_events_extremity(ed=data['ED'], m=data['EM_avg_Md'])
        EM_Md.attrs = get_attrs(vname='EM_Md', dec=True, data_unit=self.unit)
        data['EM_Md'] = EM_Md
        
        if 'EF_GR' in data:
            # calculate cumulative events duration (equation 15_2)
            ED_GR = self._calc_cumulative_events_duration(f=data['EF_GR'], d=data['ED_avg_GR'])
            ED_GR.attrs = get_attrs(vname='ED_GR', dec=True, data_unit=self.unit)
            data['ED_GR'] = ED_GR
            
            # calculate temporal events extremity tEX (equals cumulative exceedance magnitude EM) (cf. equation 18_2)
            EM_GR = self._calc_temporal_events_extremity(ed=data['ED_GR'],
                                                         m=data['EM_avg_GR'])
            EM_GR.attrs = get_attrs(vname='EM_GR', dec=True, data_unit=self.unit)
            data['EM_GR'] = EM_GR
            
            # calculate cumulative median exceedance magnitude (equation 19_4)
            EM_GR_Md = self._calc_temporal_events_extremity(ed=data['ED_GR'], m=data[
                'EM_avg_GR_Md'])
            EM_GR_Md.attrs = get_attrs(vname='EM_GR_Md', dec=True, data_unit=self.unit)
            data['EM_GR_Md'] = EM_GR_Md
            
        if 'EM_avg_Max_GR' in data:
            gvar = '_GR'
        else:
            gvar = ''
            
        # calculate cumulative maximum exceedance magnitude (cf. equation 20_2)
        EM_Max = data[f'EM_avg_Max{gvar}'] * data[f'ED{gvar}']
        EM_Max.attrs = get_attrs(vname=f'EM_Max{gvar}', dec=True, data_unit=self.unit)
        data[f'EM_Max{gvar}'] = EM_Max

        # calculate event severity (cf. equation 21_2)
        es_avg = self._calc_event_severity(d=data[f'ED_avg{gvar}'], m=data[
            f'EM_avg{gvar}'],
                                           a=data[f'EA_avg{gvar}'])
        es_avg.attrs = get_attrs(vname=f'ES_avg{gvar}', dec=True, data_unit=self.unit)
        data[f'ES_avg{gvar}'] = es_avg
    
        # calculate total events extremity (cf. equation 21_4)
        TEX = self._calc_total_events_extremity(f=data[f'EF{gvar}'],
                                                s=data[f'ES_avg{gvar}'])
        TEX.attrs = get_attrs(vname=f'TEX{gvar}', dec=True, data_unit=self.unit)
        data[f'TEX{gvar}'] = TEX
    
        # calculate exceedance heat content (cf. equation 22)
        H_AEHC_avg, H_AEHC = self._calc_exceedance_heat_content(s_avg=es_avg, d_avg=data[f'ED_avg{gvar}'],
                                                                tex=TEX)
        H_AEHC_avg.attrs = get_attrs(vname=f'H_AEHC_avg{gvar}', dec=True, data_unit=self.unit)
        H_AEHC.attrs = get_attrs(vname=f'H_AEHC{gvar}', dec=True, data_unit=self.unit)
        data[f'H_AEHC_avg{gvar}'] = H_AEHC_avg
        data[f'H_AEHC{gvar}'] = H_AEHC
        
        return data

    def _calc_spread_estimators(self):
        """
        calculate spread estimators (equation 25)
        """
        annual_data = self.CTP_results
        dec_data = self.decadal_results
        supp, slow = xr.full_like(dec_data, np.nan), xr.full_like(dec_data, np.nan)
        for icy, cy in enumerate(annual_data.time):
            # skip first and last 5 years
            if icy < 5 or icy > len(annual_data.time) - 4:
                continue
            one_decade = annual_data.isel(time=slice(icy - 5, icy + 5))
            center_val = dec_data.isel(time=icy)
            # equation 25_1
            cupp = xr.where(one_decade > center_val, 1, 0)
            
            cupp_sum = cupp.sum(dim='time')
            cupp_sum = cupp_sum.where(cupp_sum > 0, 1)
            supp_per = np.sqrt(1 / cupp_sum * ((cupp * (one_decade - center_val)**2).sum(dim='time')))
            
            clow_sum = (1 - cupp).sum(dim='time')
            clow_sum = clow_sum.where(clow_sum > 0, 1)
            slow_per = np.sqrt(1 / clow_sum * (((1 - cupp) * (one_decade - center_val)**2).sum(dim='time')))
            
            supp.loc[{'time': cy}] = supp_per
            slow.loc[{'time': cy}] = slow_per
        
        for vvar in supp.data_vars:
            supp[vvar].attrs = get_attrs(vname=vvar, spread='upper', data_unit=self.unit)
            self.decadal_results[vvar + '_supp'] = supp[vvar]
        for vvar in slow.data_vars:
            slow[vvar].attrs = get_attrs(vname=vvar, spread='lower', data_unit=self.unit)
            self.decadal_results[vvar + '_slow'] = slow[vvar]
            
    # ### amplification factors ###
    def _calc_cc(self, period=None):
        """
        calculate geometric mean of CC period (equation 26)
        
        Args:
            period: current climate period: tuple(start year, end year). Default: self.cc_period
        """
        if period is None:
            period = self.cc_period
            
        start_year, end_year = period
        cc_mean = self._calc_gmean_decadal(start_year=start_year, end_year=end_year)
        for vvar in cc_mean.data_vars:
            cc_mean[vvar].attrs = self.decadal_results[vvar].attrs
            if 'long_name' in cc_mean[vvar].attrs:
                cc_mean[vvar].attrs['long_name'] = 'CC mean of ' + cc_mean[vvar].attrs['long_name']
        self._cc_mean = cc_mean
    
    def _calc_ref(self, period=None):
        """
        calculate geometric mean of ref period (equation 26)
        
        Args:
            period: reference period: tuple(start year, end year). Default: self.ref_period
        """
        if period is None:
            period = self.ref_period
            
        start_year, end_year = period
        ref_mean = self._calc_gmean_decadal(start_year=start_year, end_year=end_year)
        
        for vvar in ref_mean.data_vars:
            ref_mean[vvar].attrs = self.decadal_results[vvar].attrs
            if 'long_name' in ref_mean[vvar].attrs:
                ref_mean[vvar].attrs['long_name'] = 'Ref mean of ' + ref_mean[vvar].attrs['long_name']
        self._ref_mean = ref_mean
    
    def _calc_gmean_decadal(self, start_year, end_year, data=None):
        """
        calculate geometric mean for given period
        Args:
            start_year: start year of selected period
            end_year: end year of selected period
            data: data to calculate geometric mean. Default: self.decadal_results

        Returns:
            geometric mean of selected period
        """
        if data is None:
            data = self.decadal_results
            
        start_cy = start_year + 5
        end_cy = end_year - 4
        start_cy_date = f'{start_cy}-01-01'
        end_cy_date = f'{end_cy}-12-31'
        if start_cy < data.time.min().dt.year or end_cy > data.time.max().dt.year:
            raise ValueError(f"Selected period {start_cy} - {end_cy} not within time range of decadal results")
        
        period_data = data.sel(time=slice(start_cy_date, end_cy_date))
        
        period_mean = self._gmean_custom(period_data, dim='time')
        doy_first, doy_last = self._calc_doy_adjustment(doy_first=period_mean.doy_first.values,
                                                        doy_last=period_mean.doy_last.values,
                                                        aep=period_mean.AEP.values)
        period_mean['doy_first'].values = doy_first
        period_mean['doy_last'].values = doy_last
        
        if 'doy_first_GR' in period_mean:
            doy_first_gr, doy_last_gr = self._calc_doy_adjustment(doy_first=period_mean.doy_first_GR.values,
                                                                  doy_last=period_mean.doy_last_GR.values,
                                                                  aep=period_mean.AEP_GR.values)
            period_mean['doy_first_GR'].values = doy_first_gr
            period_mean['doy_last_GR'].values = doy_last_gr
        return period_mean
    
    @staticmethod
    def _apply_min_duration(ds, min_duration):
        """
        keep only values above min_duration
        
        Args:
            ds: Xarray dataset (must contain 'ED' variable)
            min_duration: minimum duration in days
        """
        for vvar in ds.data_vars:
            if len(ds[vvar].dims) > 1:
                ds[vvar] = xr.where(ds.ED >= min_duration, ds[vvar], np.nan)
                
    def calc_amplification_factors(self, ref_period=(1961, 1990), cc_period=(2008, 2024), min_duration=0):
        """
        calculate amplification factors (equation 27)
        
        Args:
            ref_period: reference period: tuple(start year, end year). Default: (1961, 1990)
            cc_period: current climate period: tuple(start year, end year). Default: (2008, 2022)
            min_duration: minimum cumulative event duration over reference period in days. Default: 0. To get
            statistically robust results set to at least 3 days
        """
        self.ref_period = ref_period
        self.cc_period = cc_period
        if self._cc_mean is None:
            self._calc_cc()
        if self._ref_mean is None:
            self._calc_ref()
        cc_mean = self._cc_mean
        ref_mean = self._ref_mean
        if min_duration > 0:
            self._apply_min_duration(ref_mean, min_duration)
        
        amplification_factors = self.decadal_results / ref_mean
        amplification_factors = amplification_factors.where(ref_mean > 0)
        cc_amplification = cc_mean / ref_mean
        cc_amplification = cc_amplification.where(ref_mean > 0)
        
        # drop all spread variables
        amplification_factors = amplification_factors.drop_vars([vvar for vvar in amplification_factors.data_vars if
                                                                'slow' in vvar or 'supp' in vvar])
        cc_amplification = cc_amplification.drop_vars([vvar for vvar in cc_amplification.data_vars if
                                                       'slow' in vvar or 'supp' in vvar])

        for vvar in amplification_factors.data_vars:
            # update attributes
            amplification_factors[vvar].attrs = self.decadal_results[vvar].attrs
            cc_amplification[vvar].attrs = self.decadal_results[vvar].attrs
            if 'long_name' in amplification_factors[vvar].attrs:
                amplification_factors[vvar].attrs['long_name'] += ' amplification'
                amplification_factors[vvar].attrs['units'] = '1'
                cc_amplification[vvar].attrs['long_name'] += ' current climate amplification factor'
                cc_amplification[vvar].attrs['units'] = '1'
        
        # rename vars
        rename_dict_af = {vvar: f'{vvar}_AF' for vvar in amplification_factors.data_vars}
        rename_dict_af_cc = {vvar: f'{vvar}_AF_CC' for vvar in cc_amplification.data_vars}
        amplification_factors = amplification_factors.rename(rename_dict_af)
        cc_amplification = cc_amplification.rename(rename_dict_af_cc)
        
        self.amplification_factors = xr.merge([amplification_factors, cc_amplification])
        self.amplification_factors.time.attrs = get_attrs(vname='amplification',
                                                          period=self.CTP)
        self.amplification_factors.attrs = get_attrs(vname='amplification_global_attrs',
                                                     period=self.CTP)
        self.amplification_factors = self._duplicate_vars(self.amplification_factors)
    
    @staticmethod
    def _duplicate_vars(ds):
        """
        duplicate vars that have multiple possible names
        
        Args:
            ds: Xarray dataset
        Returns:
            ds: Xarray dataset with duplicated vars
        """
        for vvar in ds.data_vars:
            # loop through equal_vars dict
            for equal_var, repl_var in equal_vars.items():
                if equal_var in vvar and 'avg' not in vvar and 'Md' not in vvar and 'Max' not in vvar:
                    ds[vvar.replace(equal_var, repl_var)] = ds[vvar]
        return ds
    
    def save_amplification_factors(self, filepath):
        """
        save amplification factors to filepath
        """
        with warnings.catch_warnings():
            # ignore warnings due to nan multiplication
            warnings.simplefilter("ignore")
            self.amplification_factors.to_netcdf(filepath)
    
    def load_amplification_factors(self, filepath):
        """
        load amplification factors from filepath
        """
        self.amplification_factors = xr.open_dataset(filepath)
    
    @staticmethod
    def _gmean_custom(x, dim):
        """
        calculate geometric mean
        """
        return np.exp((np.log(x).mean(dim=dim)))
    
    @staticmethod
    def _calc_doy_adjustment(doy_first, doy_last, aep):
        """
        calculate adjustment for doy_first and doy_last (Equation 24)
        Args:
            doy_first:
            doy_last:
            aep: annual exposure period

        Returns:
            doy_first_adjusted, doy_last_adjusted

        """
        doy_offset = 0.5 * (30.5 * aep - (doy_last - doy_first + 1))
        doy_first_adjusted = doy_first - doy_offset
        doy_last_adjusted = doy_last + doy_offset
        return doy_first_adjusted, doy_last_adjusted
    
    def _calc_dteec_1d(self, dtec_cell):
        """
        calculate DTEEC according to equation 04 and equation 05
        Args:
            dtec_cell: 1D array with daily threshold exceedance count
        """
        # Convert to a NumPy array and change NaN to 0
        dtec_np = np.nan_to_num(dtec_cell, nan=0)
        
        # Find the starts and ends of sequences
        change = np.diff(np.concatenate(
            ([np.zeros((1,) + dtec_np.shape[1:]), dtec_np, np.zeros((1,) + dtec_np.shape[1:])]), axis=0), axis=0)
        starts = np.where(change == 1)
        ends = np.where(change == -1)
        
        # Calculate the middle points (as flat indices)
        middle_indices = (starts[0] + ends[0] - 1) // 2
        
        # Create an output array filled with NaNs
        events_np = np.full(dtec_cell.shape, self.null_val)
        
        # Set the middle points to 1 (use flat indices to index into the 3D array)
        events_np[middle_indices] = 1
        
        return events_np

    def _filter_CTP(self, ctp=None):
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
            self._set_ctp(ctp)
        elif self.CTP is None:
            raise ValueError("CTP must be set before resampling")
        
        # drop all non GR variables
        if not self._calc_grid:
            self.daily_results = self.daily_results.drop_vars([var for var in self.daily_results.data_vars if 'GR' not
                                                               in var])
        self._filter_CTP()
        self._CTP_resampler = self._daily_results_filtered.resample(time=self.CTP_freqs[self.CTP])
        self._CTP_resample_sum = self._CTP_resampler.sum('time')
        
        # dask does not support median for resampling so resample only what is necessary
        self._CTP_resample_median = xr.Dataset()
        for var in ['DTEM', 'DTEM_GR']:
            if var in self._daily_results_filtered:
                self._daily_results_filtered[var].load()
                # equation 19_1 and equation 19_3
                daily_results_gt0 = self._daily_results_filtered[var].where(self._daily_results_filtered[var] > 0)
                resampler = daily_results_gt0.resample(time=self.CTP_freqs[self.CTP])
                self._CTP_resample_median[var] = resampler.median('time')
        if self.CTP in self._overlap_ctps:
            # remove first and last year
            self._CTP_resample_sum = self._CTP_resample_sum.isel(time=slice(1, -1))
            self._CTP_resample_median = self._CTP_resample_median.isel(time=slice(1, -1))
    
    def _create_area_grid(self, input_data_grid):
        """
        create area grid for grid cells out of lat lon info
        """
        # circumference of earth at equator
        c_lon_eq = 40075
        # circumference of earth though poles
        c_lat = 40008
        
        # size of one grid cell (in km)
        lat_size = abs(input_data_grid.lat[1] - input_data_grid.lat[0]) * c_lat / 360
        lon_size = ((input_data_grid.lon[1] - input_data_grid.lon[0]) * c_lon_eq / 360 *
                    np.cos(np.deg2rad(input_data_grid.lat)))
        
        # create area size grid (in areals)
        self.area_grid = lon_size * (lat_size * xr.ones_like(input_data_grid.lon)) / 100
    
    # ### general functions ###
    def create_history(self, history, result_type):
        """
        create history of all functions called
        
        Args:
            history: history string
            result_type: type of result (daily, CTP, decadal)
        """
        ds = getattr(self, f'{result_type}_results')
        if 'history' in ds.attrs:
            ds.attrs['history'] = ds.attrs['history'] + history
        else:
            ds.attrs['history'] = history
        
    