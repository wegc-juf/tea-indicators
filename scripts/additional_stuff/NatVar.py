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

        # store data of REF period (from start center year to end center year)
        self.ref_station_data = self.station_data_af.sel(time=slice(f'{ref_period[0] + 5}-01-01',
                                                                    f'{ref_period[1] - 4}-12-31'))
        self.ref_spcus_data = self.spcus_data_af.sel(time=slice(f'{ref_period[0] + 5}-01-01',
                                                                f'{ref_period[1] - 4}-12-31'))

        # TODO: check if this is the best way to initialize these variables
        self.ref_spcus_std = None
        self.ref_station_std = None
        self.factors = None
        self.nv_low = None
        self.nv_upp = None

    def calc_ref_std(self):
        # TODO: should center year calculation be hard coded like it is now?
        ref_station_std = self.ref_station_data.std(dim='time')
        ref_spcus_std = self.ref_spcus_data.std(dim='time')
        self.ref_station_std = ref_station_std
        self.ref_spcus_std = ref_spcus_std

    def calc_factors(self):
        facs = self.ref_spcus_std / np.sqrt((self.ref_station_std ** 2).sum()
                                            / len(self.ref_station_std.station))
        self.factors = facs

    def calc_natvar(self, ref_eyr):
        data = self.station_data_af.sel(time=slice(self.station_data_af.time[0],
                                                   f'{ref_eyr}-12-31'))
        cupp = (data >= 1).astype(int)
        supp = np.sqrt((1 / cupp.sum()) * (cupp * (data - 1) ** 2).sum())
        slow = np.sqrt((1 / (1 - cupp).sum()) * ((1 - cupp) * (data - 1) ** 2).sum())

        supp_nv = np.sqrt((supp ** 2).mean()) * self.factors
        slow_nv = np.sqrt((slow ** 2).mean()) * self.factors

        self.nv_low = slow_nv
        self.nv_upp = supp_nv

    def calc_combined(self):
        s_a_ref = np.sqrt((1 / len(self.ref_spcus_data.time))
                          * ((self.ref_spcus_data['EA_avg_AF'] - 1) ** 2).sum(dim='time'))
        s_dm_ref = np.sqrt((1 / len(self.ref_spcus_data.time))
            * ((self.ref_spcus_data[f'{self.dm_var}_AF'] - 1) ** 2).sum(dim='time'))

        self.nv_low['EA_avg_AF'] = ((s_a_ref / s_dm_ref) * self.nv_low[f'{self.dm_var}_AF']).values
        self.nv_upp['EA_avg_AF'] = ((s_a_ref / s_dm_ref) * self.nv_upp[f'{self.dm_var}_AF']).values

        self.nv_low['ES_AF'] = np.sqrt(self.nv_low[f'{self.dm_var}_AF']**2 + self.nv_low['EA_avg_AF'] ** 2)
        self.nv_upp['ES_AF'] = np.sqrt(self.nv_upp[f'{self.dm_var}_AF']**2 + self.nv_upp['EA_avg_AF'] ** 2)

        self.nv_low['TEX_AF'] = np.sqrt(self.nv_low['EF_AF']**2 + self.nv_low['ES_AF'] ** 2)
        self.nv_upp['TEX_AF'] = np.sqrt(self.nv_upp['EF_AF']**2 + self.nv_upp['ES_AF'] ** 2)

        self.scaling = (s_a_ref / s_dm_ref).values

