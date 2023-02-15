# -*- coding: utf-8 -*-
"""
    th-e-srvy.system
    ~~~~~~~~~~~~~~~~
    
    
"""
from __future__ import annotations
from typing import Callable

import os
import json
import logging
import pandas as pd
import th_e_core
from th_e_core import Component
from th_e_core.configs import Configurations
from pvlib import solarposition
from .pv import PVSystem
from .model import Model
from .location import Location
from .evaluation import Evaluation

logger = logging.getLogger(__name__)

AC_E = 'Energy yield [kWh]'
AC_Y = 'Specific yield [kWh/kWp]'


class System(th_e_core.System, Evaluation):

    def __configure__(self, configs):
        super().__configure__(configs)
        data_dir = configs.dirs.data
        if not os.path.exists(data_dir):
            os.makedirs(data_dir, exist_ok=True)

        self._results_json = os.path.join(data_dir, 'results.json')
        self._results_excel = os.path.join(data_dir, 'results.xlsx')
        self._results_csv = os.path.join(data_dir, 'results.csv')
        self._results_dir = os.path.join(data_dir, 'results')
        if not os.path.exists(self._results_dir):
            os.makedirs(self._results_dir)

    def __location__(self, configs: Configurations) -> Location:
        # FIXME: location necessary for for weather instantiation, but called afterwards here
        # if isinstance(self.weather, TMYWeather):
        #     return Location.from_tmy(self.weather.meta)
        # elif isinstance(self.weather, EPWWeather):
        #     return Location.from_epw(self.weather.meta)

        return Location(configs.getfloat('Location', 'latitude'),
                        configs.getfloat('Location', 'longitude'),
                        timezone=configs.get('Location', 'timezone', fallback='UTC'),
                        altitude=configs.getfloat('Location', 'altitude', fallback=None),
                        country=configs.get('Location', 'country', fallback=None),
                        state=configs.get('Location', 'state', fallback=None))

    def __cmpt_types__(self):
        return super().__cmpt_types__('solar', 'array')

    # noinspection PyShadowingBuiltins
    def __cmpt__(self, configs: Configurations, type: str) -> Component:
        if type in ['pv', 'solar', 'array']:
            return PVSystem(self, configs)

        return super().__cmpt__(configs, type)

    def __call__(self,
                 results=None,
                 results_json=None) -> pd.DataFrame:
        progress = Progress(len(self) + 1, file=results_json)

        weather = self._get_result(results, f"{self.id}/input", self._get_weather)
        progress.update()

        result = pd.DataFrame(columns=['pv_power', 'dc_power'], index=weather.index).fillna(0)
        result.index.name = 'time'
        for cmpt in self.values():
            if cmpt.type == 'pv':
                result_pv = self._get_result(results, f"{self.id}/{cmpt.id}/output", self._get_solar_yield, cmpt, weather)
                result[['pv_power', 'dc_power']] += result_pv[['pv_power', 'dc_power']].abs()

            progress.update()

        return pd.concat([result, weather], axis=1)

    # noinspection PyUnresolvedReferences
    @staticmethod
    def _get_result(results, key: str, func: Callable, *args, **kwargs) -> pd.DataFrame:
        from th_e_core.tools import to_bool
        if results is None or key not in results:
            concat = to_bool(kwargs.pop('concat', False))
            result = func(*args, **kwargs)
            if results is not None:
                results.set(key, result, concat=concat)
            return result

        return results.get(key)

    def _get_weather(self) -> pd.DataFrame:
        weather = self.weather.get()
        if 'precipitable_water' not in weather.columns or weather['precipitable_water'].sum() == 0:
            from pvlib.atmosphere import gueymard94_pw
            weather['precipitable_water'] = gueymard94_pw(weather['temp_air'], weather['relative_humidity'])
        if 'albedo' in weather.columns and weather['albedo'].sum() == 0:
            weather.drop('albedo', axis=1, inplace=True)

        solar_position = self._get_solar_position(weather.index)
        return pd.concat([weather, solar_position], axis=1)

    def _get_solar_position(self, index: pd.DatetimeIndex) -> pd.DataFrame:
        data = pd.DataFrame(index=index)
        try:
            # TODO: use weather pressure for solar position
            data = solarposition.get_solarposition(index,
                                                   self.location.latitude,
                                                   self.location.longitude,
                                                   altitude=self.location.altitude)
            data = data.loc[:, ['azimuth', 'apparent_zenith', 'apparent_elevation']]
            data.columns = ['solar_azimuth', 'solar_zenith', 'solar_elevation']

        except ImportError as e:
            logger.warning("Unable to generate solar position: {}".format(str(e)))

        return data

    # noinspection PyMethodMayBeStatic
    def _get_solar_yield(self, pv: PVSystem, weather: pd.DataFrame) -> pd.DataFrame:
        model = Model.read(pv)
        return model(weather).rename(columns={'p_ac': 'pv_power',
                                              'p_dc': 'dc_power'})


class Progress:

    def __init__(self, total, value=0, file=None):
        self._file = file
        self._total = total + 1
        self._value = value

    def update(self):
        self._value += 1
        self._update(self._value)

    def _update(self, value):
        progress = value / self._total * 100
        if progress % 1 <= 1 / self._total * 100 and self._file is not None:
            with open(self._file, 'w', encoding='utf-8') as f:
                results = {
                    'status': 'running',
                    'progress': int(progress)
                }
                json.dump(results, f, ensure_ascii=False, indent=4)

