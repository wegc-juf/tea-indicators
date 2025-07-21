import numpy as np


class NaturalVariability:
    def __init__(self, station_data_af=None, spcus_data_af=None, param=None, ref_period=None):
        self.station_data_af = station_data_af
        self.spcus_data_af = spcus_data_af

        if param == 'Tx':
            self.em_var = 'EM_avg'
            self.dm_var = 'DM_avg'
        else:
            self.em_var = 'EM_Md'
            self.dm_var = 'DM_Md'

        self.ref_period = ref_period

        # store data of REF period (from start center year to end center year)
        self.ref_station_data = self.station_data_af.sel(time=slice(f'{ref_period[0] + 5}-01-01',
                                                                    f'{ref_period[1] - 4}-12-31'))
        self.ref_spcus_data = self.spcus_data_af.sel(time=slice(f'{ref_period[0] + 5}-01-01',
                                                                f'{ref_period[1] - 4}-12-31'))

        self.ref_spcus_std = None
        self.ref_station_std = None
        self.factors = None
        self.nv = None

    def calc_ref_std(self):
        """
        calculates standard diviation during REF period (s^REF_{AX,GR}, Eq. 32_4)
        Returns:

        """
        ref_station_std = self.ref_station_data.std(dim='time')
        ref_spcus_std = self.ref_spcus_data.std(dim='time')
        self.ref_station_std = ref_station_std
        self.ref_spcus_std = ref_spcus_std

    def calc_factors(self):
        """
        calculates scaling factors, i.e. station specific variance scaled to GR variability
        (first term of Eq. 32_5)
        Returns:

        """
        facs = self.ref_spcus_std / np.sqrt((self.ref_station_std ** 2).sum()
                                            / len(self.ref_station_std.station))
        self.factors = facs

    def calc_natvar(self):
        """
        calculates natural variability (Eq. 32_5 & 32_6)

        Returns:

        """
        data = self.station_data_af.sel(time=slice(self.station_data_af.time[0],
                                                   f'{self.ref_period[1]}-12-31'))
        cupp = (data >= 1).astype(int)
        supp = np.sqrt((1 / cupp.sum()) * (cupp * (data - 1) ** 2).sum())
        slow = np.sqrt((1 / (1 - cupp).sum()) * ((1 - cupp) * (data - 1) ** 2).sum())

        supp_nv = np.sqrt((supp ** 2).mean()) * self.factors
        slow_nv = np.sqrt((slow ** 2).mean()) * self.factors

        # rename variables
        rename_dict_supp = dict(zip(supp_nv.data_vars, [f's_{v}_NVupp' for v in supp_nv.data_vars]))
        rename_dict_slow = dict(zip(slow_nv.data_vars, [f's_{v}_NVlow' for v in slow_nv.data_vars]))
        supp_nv = supp_nv.rename(rename_dict_supp)
        slow_nv = slow_nv.rename(rename_dict_slow)

        # combine to one ds
        nv = slow_nv.merge(supp_nv)

        self.nv = nv

    def calc_combined(self):
        """
        calculate natural variability of variables with event area (Eq. 33)
        Returns:

        """
        s_a_ref = np.sqrt((1 / len(self.ref_spcus_data.time))
                          * ((self.ref_spcus_data['EA_avg_AF'] - 1) ** 2).sum(dim='time'))
        s_dm_ref = np.sqrt((1 / len(self.ref_spcus_data.time))
                           * ((self.ref_spcus_data[f'{self.dm_var}_AF'] - 1) ** 2).sum(dim='time'))

        self.nv['s_EA_avg_AF_NVlow'] = (
                (s_a_ref / s_dm_ref) * self.nv[f's_{self.dm_var}_AF_NVlow']).values
        self.nv['s_EA_avg_AF_NVupp'] = (
                (s_a_ref / s_dm_ref) * self.nv[f's_{self.dm_var}_AF_NVupp']).values

        self.nv['s_ES_AF_NVlow'] = np.sqrt(self.nv[f's_{self.dm_var}_AF_NVlow'] ** 2
                                           + self.nv['s_EA_avg_AF_NVlow'] ** 2)
        self.nv['s_ES_AF_NVupp'] = np.sqrt(self.nv[f's_{self.dm_var}_AF_NVupp'] ** 2
                                           + self.nv['s_EA_avg_AF_NVupp'] ** 2)

        self.nv['s_TEX_AF_NVlow'] = np.sqrt(
            self.nv['s_EF_AF_NVlow'] ** 2 + self.nv['s_ES_AF_NVlow'] ** 2)
        self.nv['s_TEX_AF_NVupp'] = np.sqrt(
            self.nv['s_EF_AF_NVupp'] ** 2 + self.nv['s_ES_AF_NVupp'] ** 2)

        self.nv['std_scaling_EA_DM'] = (s_a_ref / s_dm_ref).values

    def save_results(self, outname):
        """
        save results to netcdf file
        Args:
            outname: name of output file

        Returns:

        """
        # rename factor variables
        rename_dict = {v: f'GR_scaling_{v}' for v in self.factors.data_vars}
        self.factors = self.factors.rename(rename_dict)

        # combine factors with nv ds
        self.nv = self.nv.merge(self.factors)

        # add variable attributes
        for vvar in self.nv.data_vars:
            if vvar == 'std_scaling_EA_DM':
                self.nv[vvar].attrs['long_name'] = 'Standard deviation scaling factor (EA/DM)'
            elif 's_' in vvar:
                var_name = vvar.split('_NV')[0].split('s_')[1]
                bound = vvar.split('_NV')[1]
                self.nv[vvar].attrs[
                    'long_name'] = f'{bound}er bound of {var_name} natural variability'
            else:
                var_name = vvar.split('_scaling')[1]
                self.nv[vvar].attrs['long_name'] = f'GR {var_name} scaling factor'

        print(f'Saving natural variability results to {outname}')
        self.nv.to_netcdf(outname)
        