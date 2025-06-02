import numpy as np

class NaturalVariability:
    def __init__(self, station_data=None, spcus_data=None,
                 station_data_af=None, spcus_data_af=None):
        self.station_data = station_data
        self.spcus_data = spcus_data
        self.station_data_af = station_data_af
        self.spcus_data_af = spcus_data_af
        # TODO: check if this is the best way to initialize these variables
        self.ref_spcus_std = None
        self.ref_station_std = None
        self.factors = None
        self.nv_low = None
        self.nv_upp = None

    def calc_ref_std(self, ref_period):
        ref_station_data = self.station_data_af.sel(time=slice(f'{ref_period[0]}-01-01',
                                                    f'{ref_period[1]}-12-31'))
        ref_spcus_data = self.spcus_data_af.sel(time=slice(f'{ref_period[0]}-01-01',
                                                    f'{ref_period[1]}-12-31'))
        ref_station_std = ref_station_data.std(dim='time', skipna=True)
        ref_spcus_std = ref_spcus_data.std(dim='time', skipna=True)
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
        supp = np.sqrt((1/cupp.sum()) * (cupp*(data - 1)**2).sum())
        slow = np.sqrt((1/(1 - cupp).sum()) * ((1 - cupp)*(data - 1)**2).sum())

        supp_nv = np.sqrt((supp ** 2).mean()) * self.factors
        slow_nv = np.sqrt((slow ** 2).mean()) * self.factors

        self.nv_low = slow_nv
        self.nv_upp = supp_nv
