# -*- coding: utf-8 -*-
"""
    th-e-yield.system
    ~~~~~~~~~~~~~~~~~
    
    
"""
import datetime
import os
import json
import logging

import pandas as pd
import datetime as dt

import th_e_core
from th_e_core import Forecast, Component, ConfigurationUnavailableException
from th_e_core.weather import Weather, TMYWeather, EPWWeather
from th_e_core.pvsystem import PVSystem
from th_e_yield.model import Model
from th_e_yield.evaluation import Evaluation
from pvlib.location import Location
from configparser import ConfigParser as Configurations

logger = logging.getLogger(__name__)


class System(th_e_core.System, Evaluation):

    def _configure(self, configs, **kwargs):
        super()._configure(configs, **kwargs)

        data_dir = configs.get('General', 'data_dir')
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)

        self._results_dir = os.path.join(data_dir, 'results')
        if not os.path.exists(self._results_dir):
            os.makedirs(self._results_dir)

        self._results_json = os.path.join(self._results_dir, 'results.json')
        self._results_excel = os.path.join(self._results_dir, 'results.xlsx')
        self._results_csv = os.path.join(self._results_dir, 'results.csv')
        self._results_pdf = os.path.join(self._results_dir, 'results.pdf')

    def _activate(self, components, configs, **kwargs):
        super()._activate(components, configs, **kwargs)
        try:
            self.weather = Forecast.read(self, **kwargs)

        except ConfigurationUnavailableException:
            # Use weather instead of forecast, if forecast.cfg not present
            self.weather = Weather.read(self, **kwargs)
        
        if isinstance(self.weather, TMYWeather):
            self.location = Location.from_tmy(self.weather.meta)
        elif isinstance(self.weather, EPWWeather):
            self.location = Location.from_epw(self.weather.meta)
        else:
            self.location = self._location_read(configs, **kwargs)

    def _location_read(self, configs, **kwargs):
        return Location(configs.getfloat('Location', 'latitude'), 
                        configs.getfloat('Location', 'longitude'), 
                        tz=configs.get('Location', 'timezone', fallback='UTC'), 
                        altitude=configs.getfloat('Location', 'altitude', fallback=0), 
                        name=self.name, 
                        **kwargs)

    @property
    def _forecast(self):
        if isinstance(self.weather, Forecast):
            return self.weather
        
        raise AttributeError("System forecast not configured")

    @property
    def _component_types(self):
        return super()._component_types + ['array', 'modules']

    # noinspection PyShadowingBuiltins
    def _component(self, configs: Configurations, type: str, **kwargs) -> Component:
        if type in ['pv', 'array', 'modules']:
            return PVSystem(self, configs, **kwargs)

        return super()._component(configs, type, **kwargs)

    def run(self, *args, **kwargs):

        logger.info("Starting simulation for system: %s", self.name)
        start = dt.datetime.now()

        weather = self.weather.get(*args, **kwargs)
        result = pd.DataFrame(columns=['p_ac', 'p_dc'], index=weather.index).fillna(0)
        result.index.name = 'time'
        results = {}
        try:
            progress = Progress(len(self) + 1, file=self._results_json)

            for key, configs in self.items():
                if configs.type == 'pv':
                    model = Model.read(self, configs, **kwargs)
                    data = model.run(weather, **kwargs)

                    if self._database is not None:
                        self._database.write(data, **kwargs)

                    result[['p_ac', 'p_dc']] += data[['p_ac', 'p_dc']].abs()
                    results[key] = data
                    progress.update()

            results_json = self.evaluate(results, weather)

            with open(self._results_json, 'w', encoding='utf-8') as f:
                json.dump(results_json, f, ensure_ascii=False, indent=4)

        except Exception as e:
            with open(self._results_json, 'w', encoding='utf-8') as f:
                import traceback
                results_json = {
                    'status': 'error',
                    'message': str(e),
                    'error': type(e).__name__,
                    'trace': traceback.format_exc()
                }
                json.dump(results_json, f, ensure_ascii=False, indent=4)

            raise e

        logger.info('Simulation complete')
        logger.debug('Simulation complete in %i seconds', (dt.datetime.now() - start).total_seconds())

        return pd.concat([result, weather], axis=1)

    def evaluate(self, results, weather):
        json_results = {}
        json_results.update(self._evaluate_yield(results, weather))

        return json_results


class Progress:

    def __init__(self, total, value=0, file=None):
        self._file = file
        self._total = total
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
