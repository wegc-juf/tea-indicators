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
    
    def __init__(self, input_data_grid=None, threshold_grid=None, min_area=1., area_grid=None, low_extreme=False,
                 testing=False, treat_zero_as_nan=False):
        """
        Initialize TEAIndicators object
        Args:
            input_data_grid: gridded input data (e.g. temperature, precipitation)
            threshold_grid: gridded threshold values
            area_grid: results containing the area of each results cell, if None, area is assumed to be 1 for each cell
                       nan values mask out the corresponding results cells
            min_area: minimum area for a timestep to be considered as exceedance (same unit as area_grid). Default: 1
        """
        self.threshold_grid = threshold_grid
        if area_grid is None and threshold_grid is not None:
            area_grid = xr.ones_like(threshold_grid)
        self.area_grid = area_grid
        self.input_data_grid = input_data_grid
        self.daily_results = xr.Dataset()
        self._daily_results_filtered = None
        self.min_area = min_area
        self.gr_vars = None
        self.low_extreme = low_extreme
        self.testing = testing
        self.treat_zero_as_nan = treat_zero_as_nan
        
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
        
        if self.treat_zero_as_nan:
            self.null_val = np.nan
        else:
            self.null_val = 0
        
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
        """
        if self.daily_results['DTEM'] is None:
            self.calc_DTEM()
        dtem = self.daily_results.DTEM
        if self.treat_zero_as_nan:
            dtec = dtem.where(dtem.isnull(), 1)
        else:
            dtec = xr.where(dtem > 0, 1, dtem)
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
        if 'DTEA_GR' not in self.daily_results:
            self.calc_DTEA_GR()
        dtea_gr = self.daily_results.DTEA_GR
        dtec_gr = xr.where(dtea_gr >= min_area, 1, self.null_val)
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
        if 'DTEC_GR' not in self.daily_results:
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
        if 'DTEC' not in self.daily_results:
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
        if 'DTEA' not in self.daily_results:
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
        if self.treat_zero_as_nan:
            dtem = dtem.where(dtem > 0).astype('float32')
        else:
            dtem = xr.where(dtem <= 0, 0, dtem)
        dtem.attrs = get_attrs(vname='DTEM')
        self.daily_results['DTEM'] = dtem
        
    def calc_DTEM_Max_GR(self):
        """
        calculate maximum DTEM for GR (equation 09)
        """
        if 'DTEM' not in self.daily_results:
            self.calc_DTEM()
        if 'DTEC_GR' not in self.daily_results:
            self.calc_DTEC_GR()
        dtem = self.daily_results.DTEM
        dtem_max = dtem.max(dim=self.threshold_grid.dims)
        dtem_max = dtem_max.where(self.daily_results.DTEC_GR == 1, self.null_val)
        dtem_max.attrs = get_attrs(vname='DTEM_Max_GR')
        self.daily_results['DTEM_Max_GR'] = dtem_max

    def calc_DTEM_GR(self):
        """
        calculate Daily Threshold Exceedance Magnitude (GR) (equation 08)
        """
        if 'DTEA_GR' not in self.daily_results:
            self.calc_DTEA_GR()
        if 'DTEM' not in self.daily_results:
            self.calc_DTEM()
        if 'DTEC_GR' not in self.daily_results:
            self.calc_DTEC_GR()
        dtea_gr = self.daily_results.DTEA_GR
        dtem = self.daily_results.DTEM
        dtec_gr = self.daily_results.DTEC_GR
        area_fac = self.area_grid / dtea_gr
        dtem_gr = (dtem * area_fac).sum(axis=(1, 2), skipna=True)
        dtem_gr = dtem_gr.where(dtec_gr == 1, self.null_val)
        dtem_gr = dtem_gr.rename(f'{dtem.name}_GR')
        dtem_gr.attrs = get_attrs(vname='DTEM_GR')
        dtema_gr = dtem_gr * dtea_gr
        self.daily_results['DTEM_GR'] = dtem_gr
        self.daily_results['DTEMA_GR'] = dtema_gr
    
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
        self.calc_DTEM_Max_GR()
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
        self.calc_DTEM_Max_GR()
        self.calc_DTEEC_GR()
        
    # ### Climatic Time Period (CTP) functions ###
    def set_ctp(self, ctp):
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
        self.CTP = ctp
        ctp_attrs = get_attrs(vname='CTP_global_attrs', period=ctp)
        # TODO: add CF-Convention compatible attributes...
        self.CTP_results.attrs = ctp_attrs
    
    def calc_event_frequency(self):
        """
        calculate event frequency (equation 11 and equation 12)
        """
        if self.CTP is None:
            raise ValueError("CTP must be set before calculating event frequency")
        if 'DTEEC' not in self.daily_results:
            self.calc_DTEEC()
        if 'DTEEC_GR' not in self.daily_results:
            self.calc_DTEEC_GR()
            
        if self._CTP_resample_sum is None:
            self._resample_to_CTP()
            
        ef = self._CTP_resample_sum.DTEEC
        ef = ef.where(ef.notnull(), 0)
        ef_gr = self._CTP_resample_sum.DTEEC_GR
        ef_gr = ef_gr.where(ef_gr.notnull(), 0)
        
        ef.attrs = get_attrs(vname='EF')
        self.CTP_results['EF'] = ef
        ef_gr.attrs = get_attrs(vname='EF_GR')
        self.CTP_results['EF_GR'] = ef_gr
    
    def calc_supplementary_event_vars(self):
        """
        calculate supplementary event variables (equation 13)
        """
        if 'EF' not in self.CTP_results:
            self.calc_event_frequency()
        
        doy = [pd.Timestamp(dy).day_of_year for dy in self._daily_results_filtered.time.values]
        self._daily_results_filtered.coords['doy'] = ('time', doy)
        
        if self.treat_zero_as_nan:
            event_doy = self._daily_results_filtered.doy.where(self._daily_results_filtered.DTEEC.notnull())
            event_doy_gr = self._daily_results_filtered.doy.where(self._daily_results_filtered.DTEEC_GR.notnull())
        else:
            event_doy = self._daily_results_filtered.doy.where(self._daily_results_filtered.DTEEC > 0)
            event_doy_gr = self._daily_results_filtered.doy.where(self._daily_results_filtered.DTEEC_GR > 0)
        resampler = event_doy.resample(time=self.CTP_freqs[self.CTP])
        resampler_gr = event_doy_gr.resample(time=self.CTP_freqs[self.CTP])
        
        # equation 13_1
        doy_first = resampler.min('time')
        # equation 13_4
        doy_first_gr = resampler_gr.min('time')
        # equation 13_2
        doy_last = resampler.max('time')
        # equation 13_5
        doy_last_gr = resampler_gr.max('time')
        
        # equation 13_3
        aep = (doy_last - doy_first + 1) / 30.5
        # equation 13_6
        aep_gr = (doy_last_gr - doy_first_gr + 1) / 30.5
        
        doy_first.attrs = get_attrs(vname='doy_first')
        doy_first_gr.attrs = get_attrs(vname='doy_first_GR')
        doy_last.attrs = get_attrs(vname='doy_last')
        doy_last_gr.attrs = get_attrs(vname='doy_last_GR')
        aep.attrs = get_attrs(vname='delta_y')
        aep_gr.attrs = get_attrs(vname='delta_y_GR')
        
        self.CTP_results['doy_first'] = doy_first
        self.CTP_results['doy_last'] = doy_last
        self.CTP_results['AEP'] = aep
        self.CTP_results['doy_first_GR'] = doy_first_gr
        self.CTP_results['doy_last_GR'] = doy_last_gr
        self.CTP_results['AEP_GR'] = aep_gr
    
    def calc_event_duration(self):
        """
        calculate event duration (equation 14 and equation 15)
        """
        if 'EF' not in self.CTP_results:
            self.calc_event_frequency()
        
        # equation 14_2
        ed = self._CTP_resample_sum.DTEC
        # equation 15_2
        ed_gr = self._CTP_resample_sum.DTEC_GR
        
        ed.attrs = get_attrs(vname='ED')
        ed_gr.attrs = get_attrs(vname='ED_GR')
        
        self.CTP_results['ED'] = ed
        self.CTP_results['ED_GR'] = ed_gr
        
        # set EF = 0 to nan
        ef = self.CTP_results['EF'].where(self.CTP_results['EF'] > 0)
        ef_gr = self.CTP_results['EF_GR'].where(self.CTP_results['EF_GR'] > 0)
        
        # calc average event duration
        # equation 14_1
        ed_avg = ed / ef
        # equation 15_1
        ed_avg_gr = ed_gr / ef_gr
        
        ed_avg.attrs = get_attrs(vname='ED_avg')
        ed_avg_gr.attrs = get_attrs(vname='ED_avg_GR')
        self.CTP_results['ED_avg'] = ed_avg
        self.CTP_results['ED_avg_GR'] = ed_avg_gr
    
    def calc_exceedance_magnitude(self):
        """
        calculate exceedance magnitude (equation 17 and equation 18), median exceedance magnitude (equation 19), and
        maximum exceedance magnitude (equation 20)
        """
        
        if 'ED' not in self.CTP_results:
            self.calc_event_duration()
            
        # equation 17_2
        em = self._CTP_resample_sum.DTEM
        # equation 18_2
        em_gr = self._CTP_resample_sum.DTEM_GR
    
        # calc average exceedance magnitude
        # equation 17_1
        em_avg = em / self.CTP_results.ED
        # equation 18_1
        em_avg_gr = em_gr / self.CTP_results.ED_GR
        
        em.attrs = get_attrs(vname='EM')
        em_gr.attrs = get_attrs(vname='EM_GR')
        em_avg.attrs = get_attrs(vname='EM_avg')
        em_avg_gr.attrs = get_attrs(vname='EM_avg_GR')
        
        self.CTP_results['EM'] = em
        self.CTP_results['EM_GR'] = em_gr
        self.CTP_results['EM_avg'] = em_avg
        self.CTP_results['EM_avg_GR'] = em_avg_gr
        
        # calc median exceedance magnitude
        # equation 19_1
        em_avg_med = self._CTP_resample_median.DTEM
        # equation 19_3
        em_avg_gr_med = self._CTP_resample_median.DTEM_GR
        # equation 19_2
        em_med = self.CTP_results.ED * em_avg_med
        # equation 19_4
        em_gr_med = self.CTP_results.ED_GR * em_avg_gr_med
        
        em_avg_med.attrs = get_attrs(vname='EM_avg_Md')
        em_avg_gr_med.attrs = get_attrs(vname='EM_avg_GR_Md')
        em_med.attrs = get_attrs(vname='EM_Md')
        em_gr_med.attrs = get_attrs(vname='EM_GR_Md')
        
        self.CTP_results['EM_avg_Md'] = em_avg_med
        self.CTP_results['EM_avg_GR_Md'] = em_avg_gr_med
        self.CTP_results['EM_Md'] = em_med
        self.CTP_results['EM_GR_Md'] = em_gr_med
        
        # calc maximum exceedance magnitude
        # equation 20_2
        em_gr_max = self._CTP_resample_sum.DTEM_Max_GR
        # equation 20_1
        em_gr_avg_max = em_gr_max / self.CTP_results.ED_GR
        
        em_gr_max.attrs = get_attrs(vname='EM_Max_GR')
        em_gr_avg_max.attrs = get_attrs(vname='EM_avg_Max_GR')
        
        self.CTP_results['EM_Max_GR'] = em_gr_max
        self.CTP_results['EM_avg_Max_GR'] = em_gr_avg_max
    
    def calc_total_events_extremity(self):
        """
        calculate total events extremity (equation 21_3)
        """
        # equation 21_3
        tex = self._CTP_resample_sum.DTEMA_GR
        tex.attrs = get_attrs(vname='TEX_GR')
        self.CTP_results['TEX_GR'] = tex
    
    def calc_exceedance_area(self):
        """
        calculate exceedance area (equation 21_1)
        """
        if self.CTP_results['TEX_GR'] is None:
            self.calc_total_events_extremity()
        if self.CTP_results['EM_GR'] is None:
            self.calc_exceedance_magnitude()
        
        # equation 21_1
        ea_avg = self.CTP_results.TEX_GR / self.CTP_results.EM_GR
        ea_avg.attrs = get_attrs(vname='EA_avg_GR')
        self.CTP_results['EA_avg_GR'] = ea_avg
    
    def calc_event_severity(self):
        """
        calculate event severity (equation 21_2)
        """
        if self.CTP_results['EA_avg_GR'] is None:
            self.calc_exceedance_area()
        if self.CTP_results['EM_avg_GR'] is None:
            self.calc_exceedance_magnitude()
        if self.CTP_results['ED_avg_GR'] is None:
            self.calc_event_duration()
        
        # equation 21_2
        es_avg = self.CTP_results.ED_avg_GR * self.CTP_results.EM_avg_GR * self.CTP_results.EA_avg_GR
        es_avg.attrs = get_attrs(vname='ES_avg_GR')
        self.CTP_results['ES_avg_GR'] = es_avg
    
    def calc_annual_CTP_indicators(self, ctp, delete_daily_results=True):
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
            delete_daily_results: delete daily results after calculation
        """
        self.set_ctp(ctp)
        self.calc_event_frequency()
        self.calc_supplementary_event_vars()
        self.calc_event_duration()
        self.calc_exceedance_magnitude()
        self.calc_total_events_extremity()
        self.calc_exceedance_area()
        self.calc_event_severity()
        if delete_daily_results:
            del self._daily_results_filtered
            del self.daily_results
        ctp_attrs = get_attrs(vname='CTP', period=self.CTP)
        self.CTP_results['time'].attrs = ctp_attrs

    def save_CTP_results(self, filepath):
        """
        save all CTP results to filepath
        """
        self.CTP_results.to_netcdf(filepath)
        
    def load_CTP_results(self, filepath):
        """
        load all CTP results from filepath
        """
        self.CTP_results = xr.open_dataset(filepath)

    def _calc_dteec_1d(self, dtec_cell):
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
            self.set_ctp(ctp)
        elif self.CTP is None:
            raise ValueError("CTP must be set before resampling")
        
        self._filter_CTP()
        self._CTP_resampler = self._daily_results_filtered.resample(time=self.CTP_freqs[self.CTP])
        self._CTP_resample_sum = self._CTP_resampler.sum('time')
        # dask does not support median for resampling so resample only what is necessary
        self._CTP_resample_median = xr.Dataset()
        for var in ['DTEM', 'DTEM_GR']:
            self._daily_results_filtered[var].load()
            resampler = self._daily_results_filtered[var].resample(time=self.CTP_freqs[self.CTP])
            self._CTP_resample_median[var] = resampler.median('time')
        if self.CTP in self._overlap_ctps:
            # remove first and last year
            self._CTP_resample_sum = self._CTP_resample_sum.isel(time=slice(1, -1))
            self._CTP_resample_median = self._CTP_resample_median.isel(time=slice(1, -1))
        
    