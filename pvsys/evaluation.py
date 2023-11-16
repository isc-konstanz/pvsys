# -*- coding: utf-8 -*-
"""
    pvsys.evaluation
    ~~~~~~~~~~~~~~~~


"""
from __future__ import annotations
from typing import Dict
import os
import logging
import pandas as pd
import calendar
import traceback

# noinspection PyProtectedMember
from corsys.io._var import COLUMNS
from corsys.io import DatabaseUnavailableException
from corsys.cmpt import ElectricalEnergyStorage
from corsys import Configurations, Configurable, System
from scisys.io import excel, plot
from scisys import Results, Progress
from .pv import PVSystem

logger = logging.getLogger(__name__)

# noinspection SpellCheckingInspection
CMPTS = {
    'tes': 'Buffer Storage',
    'ees': 'Battery Storage',
    'ev': 'Electric Vehicle',
    'pv': 'Photovoltaics'
}

AC_Y = 'Specific yield [kWh/kWp]'
AC_E = 'Energy yield [kWh]'
DC_E = 'Energy yield (DC) [kWh]'

EES_CY = 'Cycles'


class Evaluation(Configurable):

    def __init__(self, system: System, name: str = 'Evaluation') -> None:
        super().__init__(system.configs)
        self.system = system
        self.name = name

    def __configure__(self, configs: Configurations) -> None:
        super().__configure__(configs)
        data_dir = configs.dirs.data
        if not os.path.exists(data_dir):
            os.makedirs(data_dir, exist_ok=True)

        self._results_json = os.path.join(data_dir, 'results.json')
        self._results_excel = os.path.join(data_dir, 'results.xlsx')
        self._results_csv = os.path.join(data_dir, 'results.csv')
        self._results_pdf = os.path.join(data_dir, 'results.pdf')
        self._results_dir = os.path.join(data_dir, 'results')
        if not os.path.exists(self._results_dir):
            os.makedirs(self._results_dir)

        self._plots_dir = os.path.join(data_dir, 'plots')
        if not os.path.exists(self._plots_dir):
            os.makedirs(self._plots_dir)

    # noinspection PyProtectedMember
    def __call__(self, *args, **kwargs) -> Results:
        logger.info("Starting evaluation for system: %s", self.system.name)
        progress = Progress.instance(desc=f"{self.name}: {self.system.name}",
                                     total=len(self.system)*2+1,
                                     file=self._results_json)

        results = Results(self.system)
        results.durations.start('Evaluation')
        results_key = f"{self.system.id}/output"
        try:
            if results_key not in results:
                results.durations.start('Prediction')
                input = _get(results, f"{self.system.id}/input", self.system._get_input, *args, **kwargs)
                progress.update()

                result = pd.DataFrame()
                result.index.name = 'time'
                for cmpt in self.system.values():
                    if cmpt.type == PVSystem.TYPE:
                        cmpt_key = f"{self.system.id}/{cmpt.id}/output"
                        result_pv = _get(results, cmpt_key, self.system._get_solar_yield, cmpt, input)
                        if result.empty:
                            result = result_pv
                        else:
                            result += result_pv
                        result[f'{cmpt.id}_power'] = result_pv[PVSystem.POWER]

                    progress.update()

                result = pd.concat([result, input], axis=1)
                results.set(results_key, result, how='combine')
                results.durations.stop('Prediction')
            else:
                # If this component was simulated already, load the results and skip the calculation
                results.load(results_key)
                progress.complete()
            try:
                reference = _get(results, f"{self.system.id}/reference", self.system.database.read, **kwargs)

                # noinspection PyShadowingBuiltins, PyShadowingNames
                def add_reference(type: str, unit: str = 'power'):
                    cmpts = self.system.get_type(type)
                    if len(cmpts) > 0 and all([f'{cmpt.id}_{unit}' in reference.columns for cmpt in cmpts]):
                        results.data[f'{type}_{unit}_ref'] = 0
                        for cmpt in self.system.get_type(f'{type}'):
                            cmpt_ref = reference[f'{cmpt.id}_{unit}']
                            results.data[f'{type}_{unit}_ref'] += cmpt_ref
                            results.data[f'{cmpt.id}_{unit}_ref'] = cmpt_ref
                            results.data[f'{cmpt.id}_{unit}_err'] = (results.data[f'{cmpt.id}_{unit}'] - cmpt_ref)

                        results.data[f'{type}_{unit}_err'] = (results.data[f'{type}_{unit}'] -
                                                              results.data[f'{type}_{unit}_ref'])
                    elif f'{type}_{unit}' in reference.columns:
                        results.data[f'{type}_{unit}_ref'] = reference[f'{type}_{unit}']

                add_reference('el')
                add_reference('th')
                add_reference('th_ht')
                add_reference('th_dom')
                add_reference(PVSystem.TYPE)

            except DatabaseUnavailableException as e:
                reference = None
                logger.debug("Unable to retrieve reference values for system %s: %s", self.system.name, str(e))

            def prepare_data(data: pd.DataFrame) -> pd.DataFrame:
                data = data.tz_convert(self.system.location.timezone).tz_localize(None)
                data = data[[column for column in COLUMNS.keys() if column in data.columns]]
                data.rename(columns=COLUMNS, inplace=True)
                data.index.name = 'Time'
                return data

            hours = pd.Series(results.data.index, index=results.data.index)
            hours = (hours - hours.shift(1)).fillna(method='bfill').dt.total_seconds() / 3600.

            if PVSystem.ENERGY not in results.data and PVSystem.POWER in results.data:
                results.data[PVSystem.ENERGY] = results.data[PVSystem.POWER] / 1000. * hours
            if f'{PVSystem.ENERGY}_ref' not in results.data and f'{PVSystem.POWER}_ref' in results.data:
                results.data[f'{PVSystem.ENERGY}_ref'] = results.data[f'{PVSystem.POWER}_ref'] / 1000. * hours
                results.data[f'{PVSystem.ENERGY}_err'] = (results.data[PVSystem.ENERGY] -
                                                          results.data[f'{PVSystem.ENERGY}_ref'])

            if PVSystem.ENERGY_DC not in results.data and PVSystem.POWER_DC in results.data:
                results.data[PVSystem.ENERGY_DC] = results.data[PVSystem.POWER_DC] / 1000. * hours

            summary = pd.DataFrame(columns=pd.MultiIndex.from_tuples((), names=['System', '']))
            summary_json = {
                'status': 'success'
            }
            self._evaluate(summary_json, summary, results.data)

            summary_data = {
                self.system.name: prepare_data(results.data)
            }
            if len(self.system) > 1:
                for cmpt in self.system.values():
                    cmpt_key = f"{self.system.id}/{cmpt.id}/output"
                    if cmpt_key in results:
                        results_suffix = cmpt.name
                        for cmpt_type in self.system.get_types():
                            results_suffix = results_suffix.replace(cmpt_type, '')
                        if len(results_suffix) < 1 and len(self.system.get_type(cmpt.type)) > 1:
                            results_suffix += str(list(self.system).index(cmpt) + 1)
                        results_name = CMPTS[cmpt.type] if cmpt.type in CMPTS else cmpt.type.upper()
                        results_name = f"{results_name.strip()} {results_suffix.strip()}".title()
                        summary_data[results_name] = prepare_data(results[cmpt_key])

            summary.to_csv(self._results_csv, encoding='utf-8-sig')
            excel.write(summary, summary_data, file=self._results_excel, index=False)
            progress.complete(summary_json)

        except Exception as e:
            logger.error("Error evaluating system %s: %s", self.system.name, str(e))
            logger.debug("%s: %s", type(e).__name__, traceback.format_exc())
            progress.complete({
                    'status': 'error',
                    'message': str(e),
                    'error': type(e).__name__,
                    'trace': traceback.format_exc()
                })
            raise e

        finally:
            results.durations.stop('Evaluation')
            results.close()
            progress.reset()

        logger.info("Evaluation complete")
        logger.debug('Evaluation complete in %i minutes', results.durations['Evaluation'])

        return results

    def _evaluate(self,
                  summary_json: Dict,
                  summary: pd.DataFrame,
                  results: pd.DataFrame) -> None:

        summary_json.update(self._evaluate_yield(summary, results))
        summary_json.update(self._evaluate_storage(summary, results))
        summary_json.update(self._evaluate_weather(summary, results))

    def _evaluate_yield(self, summary: pd.DataFrame, results: pd.DataFrame, reference: pd.DataFrame = None) -> Dict:
        if PVSystem.POWER not in results.columns or not self.system.contains_type(PVSystem.TYPE):
            return {}

        results_kwp = 0
        for pv in self.system.get_type(PVSystem.TYPE):
            if f'{pv.id}_power' in results.columns and f'{pv.id}_power_ref' not in results.columns:
                if System.POWER_EL not in results.columns and f'{System.POWER_EL}_ref' in results.columns:
                    results.loc[:, System.POWER_EL] = results[f'{System.POWER_EL}_ref']
                if System.POWER_EL in results.columns:
                    results.loc[:, System.POWER_EL] -= results[f'{pv.id}_power']

            results_kwp += pv.power_max / 1000.

        if System.POWER_EL in results.columns and f'{System.POWER_EL}_ref' in results.columns:
            results[f'{System.POWER_EL}_err'] = (results[f'{System.POWER_EL}'] -
                                                 results[f'{System.POWER_EL}_ref'])

        results[PVSystem.YIELD_SPECIFIC] = results[PVSystem.ENERGY] / results_kwp

        results = results.dropna(axis='index', how='all')

        plot_data = results[[PVSystem.ENERGY]].groupby(results.index.month).sum()
        plot.bar(x=plot_data.index, y=PVSystem.ENERGY, data=plot_data,
                 xlabel='Month', ylabel='Energy [kWh]', title='Monthly Yield',
                 colors=list(reversed(plot.COLORS)), file=os.path.join(self._plots_dir, 'yield_months.png'))

        plot_data = pd.concat([pd.Series(data=results.loc[results.index.month == m, PVSystem.POWER]/1000.,
                                         name=calendar.month_name[m]) for m in range(1, 13)], axis='columns')
        plot_data['hour'] = plot_data.index.hour + plot_data.index.minute/60.
        plot_melt = plot_data.melt(id_vars='hour', var_name='Months')
        plot.line(x='hour', y='value', data=plot_melt,
                  xlabel='Hour of the Day', ylabel='Power [kW]', title='Yield Profile', hue='Months',  # style='Months',
                  colors=list(reversed(plot.COLORS)), file=os.path.join(self._plots_dir, 'yield_months_profile.png'))

        yield_specific = round(results[PVSystem.YIELD_SPECIFIC].sum(), 2)
        yield_energy = round(results[PVSystem.ENERGY].sum(), 2)

        summary.loc[self.system.name, ('Yield', AC_Y)] = yield_specific
        summary.loc[self.system.name, ('Yield', AC_E)] = yield_energy

        summary_dict = {'yield_specific': yield_specific,
                        'yield_energy': yield_energy}

        if PVSystem.ENERGY_DC in results:
            dc_energy = round(results[PVSystem.ENERGY_DC].sum(), 2)

            summary.loc[self.system.name, ('Yield', DC_E)] = dc_energy
            summary_dict['yield_energy_dc'] = dc_energy

        return summary_dict

    def _evaluate_storage(self, summary: pd.DataFrame, results: pd.DataFrame) -> Dict:
        if System.POWER_EL not in results.columns and f'{System.POWER_EL}_ref' in results.columns:
            results.loc[:, System.POWER_EL] = results[f'{System.POWER_EL}_ref']

        elif System.POWER_EL not in results.columns or not self.system.contains_type(ElectricalEnergyStorage.TYPE):
            return {}

        ees_results = []
        for ees in self.system.get_type(ElectricalEnergyStorage.TYPE):
            ees_result = ees.infer_soc(results)
            ees_results.append(ees_result)
            results.loc[:, System.POWER_EL] = ees_result[System.POWER_EL]

        if System.POWER_EL in results.columns and f'{System.POWER_EL}_ref' in results.columns:
            results[f'{System.POWER_EL}_err'] = (results[f'{System.POWER_EL}'] -
                                                 results[f'{System.POWER_EL}_ref'])

        def insert(column: str, how: str) -> None:
            data = pd.concat(ees_results, axis='columns').loc[:, [column]]
            if len(data.columns) > 1:
                if how == 'sum':
                    data = data.sum(axis='columns')
                elif how == 'mean':
                    data = data.mean(axis='columns')
            results.loc[:, column] = data

        insert(ElectricalEnergyStorage.STATE_OF_CHARGE, how='mean')
        insert(ElectricalEnergyStorage.POWER_DISCHARGE, how='sum')
        insert(ElectricalEnergyStorage.POWER_CHARGE, how='sum')

        hours = pd.Series(results.index, index=results.index)
        hours = (hours - hours.shift(1)).fillna(method='bfill').dt.total_seconds() / 3600.

        ees_capacity = sum([ees.capacity for ees in self.system.get_type(ElectricalEnergyStorage.TYPE)])
        ees_cycles = (results[ElectricalEnergyStorage.POWER_CHARGE] / 1000 * hours).sum() / ees_capacity

        summary.loc[self.system.name, ('Battery storage', EES_CY)] = ees_cycles

        return {'ees_cycles': ees_cycles}

    def _evaluate_weather(self, summary: pd.DataFrame, results: pd.DataFrame) -> Dict:
        hours = pd.Series(results.index, index=results.index)
        hours = (hours - hours.shift(1)).fillna(method='bfill').dt.total_seconds() / 3600.
        ghi = round((results['ghi'] / 1000. * hours).sum(), 2)
        dhi = round((results['dhi'] / 1000. * hours).sum(), 2)

        summary.loc[self.system.name, ('Weather', 'GHI [kWh/m^2]')] = ghi
        summary.loc[self.system.name, ('Weather', 'DHI [kWh/m^2]')] = dhi

        return {}

    def _evaluate_legacy(self, summary: pd.DataFrame, results: pd.DataFrame, reference: pd.DataFrame = None) -> Dict:
        hours = pd.Series(results.index, index=results.index)
        hours = (hours - hours.shift(1)).fillna(method='bfill').dt.total_seconds() / 3600.
        key = 'ees'
        if key in self:
            # TODO: Abfrage ob tz-infos vorhanden
            # p_isc.index = p_isc.index.tz_localize('Europe/Berlin')
            # TODO: N.A./null Werte in der Zeitreihe
            configs = self.system[key]
            domestic = pd.read_csv(configs.dirs.data, delimiter=',', index_col='time', parse_dates=True,
                                   dayfirst=True)
            start = domestic.index[0]
            stop = domestic.index[-1]
            energy_price = configs.energy_price
            domestic.index = (domestic.index + dt.timedelta(minutes=30))[:].floor('H')
            # TODO: delete following line (used because ISC consumption used for domestic consumption)
            # domestic = domestic / 100
            power = domestic.rename(columns={'power': 'p_dom'})

            if results != {}:
                p_pv = results['array'][['p_ac']]
                p_pv.index = p_pv.index.tz_convert(None)
                p_pv.index = p_pv.index[:].floor('H')
                p_pv = p_pv.rename(columns={'p_ac': 'p_pv'})
                power = power.join(p_pv)
            else:
                print('No pv available.')
                power_pv = pd.DataFrame(columns={'p_pv'}, index=power.index, data=np.zeros(power.values.__len__()))
                power = power.join(power_pv)

            p_dom = domestic['power'].to_frame()
            p_dom = p_dom.groupby(p_dom.index.hour).mean()
            p_dom = p_dom.rename(columns={'power': 'p_dom'})
            if results != {}:
                p_dom = pd.concat([p_dom, results['daily']], axis=1, join='inner')
            results.update({'daily': p_dom})

            # energy yield and costs
            pv_yield = power['p_pv'][start:stop].sum()
            consumption = power['p_dom'][start:stop].sum()
            energy_yield = pv_yield - consumption
            energy_balance = (power['p_pv'][start:stop] - power['p_dom'][start:stop]) / 1000

            # energy costs raw
            energy_costs_raw = (-consumption / 1000 * configs.energy_price).sum()

            # energy costs with PV
            energy_costs_pv = 0
            for dW in energy_balance:
                if dW > 0:
                    energy_costs_pv += dW * configs.feed_in_tariff_pv
                elif dW < 0:
                    energy_costs_pv += dW * configs.energy_price

            # energy costs with PV and battery
            energy_costs_bat = 0
            storage = configs.capacity * 0.5
            for dW in energy_balance:
                if dW > 0:
                    if storage < configs.capacity:
                        storage += dW
                    if storage >= configs.capacity:
                        if storage > configs.capacity:
                            dW = storage - configs.capacity
                            storage = configs.capacity
                        energy_costs_bat += dW * configs.feed_in_tariff_pv
                elif dW < 0:
                    if storage > 0:
                        storage += dW
                    if storage <= 0:
                        if storage < 0:
                            dW = storage
                            storage = 0
                        energy_costs_bat += dW * configs.energy_price

            results_ees = {'start': power.index[0],
                           'stop': power.index[-1],
                           'costs_raw': [energy_costs_raw],
                           'costs_pv': [energy_costs_pv],
                           'costs_pv_bat': [energy_costs_bat]}

            results.update({key: pd.DataFrame(data=results_ees)})

        key = 'ev'
        if key in self:
            configs = self[key]
            start = domestic.index[0]
            stop = domestic.index[-1]
            if configs.charge_mode == 'SLOW':
                charge_duration = 10
            elif configs.charge_mode == 'NORMAL':
                charge_duration = 7
            elif configs.charge_mode == 'FAST':
                charge_duration = 4
            else:
                charge_duration = int(configs.charge_mode)
            charge_time = [int(x) for x in configs.charge_time.split('-')]

            data = configs.house_connection_point * 1000 + (results['daily']['p_pv'] - results['daily']['p_dom'])
            data_tmp = data[charge_time[0]:].append(data[:charge_time[1]])

            energy_potential = data_tmp.mean() / 1000 * charge_duration
            energy_potential_min = data_tmp.min() / 1000 * charge_duration
            km_potential = energy_potential / configs.fuel_consumption * 100
            km_potential_min = energy_potential_min / configs.fuel_consumption * 100
            ev_potential = int(km_potential / (configs.driving_distance / 7))
            ev_potential_min = int(km_potential_min / (configs.driving_distance / 7))
            ev_distance_total = int(configs.quantity * configs.driving_distance * (stop - start).days / 7)
            energy_costs_raw = np.round(ev_distance_total / 100 * configs.fuel_consumption * energy_price, 1)

            results_ev = {'energy costs': [energy_costs_raw],
                          'ev potential without LMS': [int(data.min() / 4600)],
                          'energy potential': [energy_potential_min],
                          'km potential': [km_potential_min],
                          'ev potential': [ev_potential_min],
                          'house connection point': [configs.house_connection_point]}
            results.update({key: pd.DataFrame(data=results_ev)})

        # for key, configs in self.items():
        #     if key == 'ees' or key == 'ev':
        #         self._write_pdf(results_summary, results_total, results)
        #         os.system(self._results_pdf)
        #         break
        return {}

    def _write_pdf(self, results_summary, results_total, results):
        from reportlab.lib.enums import TA_JUSTIFY, TA_LEFT, TA_CENTER, TA_RIGHT
        from reportlab.lib.pagesizes import letter
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib import colors

        self._create_graphics(results_summary, results_total, results)
        doc = SimpleDocTemplate(self._results_pdf, pagesize=letter,
                                rightMargin=72, leftMargin=72,
                                topMargin=72, bottomMargin=18)
        try:
            pdf = []
            logo = Image('data/isc_logo.jpg', 60, 60, hAlign='CENTER')
            img_1 = Image(os.path.join(self._results_dir, 'array.png'), 250, 200)
            img_2 = Image(os.path.join(self._results_dir, 'ees_1.png'), 400, 260)
            img_3 = Image(os.path.join(self._results_dir, 'ev_2.png'), 400, 260)
            client_name = "Steffen Friedriszik"
            address_parts = ["Rudolf Diesel Str. 17", "78467 Konstanz"]
            styles = getSampleStyleSheet()
            styles.add(ParagraphStyle(name='Normal_right', alignment=TA_RIGHT))
            styles.add(ParagraphStyle(name='Normal_center', alignment=TA_CENTER))
            styles.add(ParagraphStyle(name='Heading222', fontName='Helvetica-Bold', fontSize=18, alignment=TA_CENTER,
                                      textColor=colors.blue, leading=20))
            styles.add(ParagraphStyle(name='Heading', fontName='Helvetica-Bold', fontSize=18, alignment=TA_CENTER,
                                      textColor=colors.rgb2cmyk(0, 79, 159), leading=20))
            styles.add(ParagraphStyle(name='Normal_bold', fontSize=14, leading=16))
            title = 'Auswertung der Messergebnisse <br/> für <br/> %s' % self.name
            text_intro = 'Sehr geehrter Herr %s, im Folgenden sehen Sie die Auswertung der Messergebnisse und ihre ' \
                         'Interpretation. Die Analyse erfolgt hinsichtlich allgemeinen, wissenschaftlichen ' \
                         'Aspekten, sowie unter monetären Gesichtspunkten. <br/>' \
                         'Die Messung dauerte %s Tage. Sie startete am %s Uhr und endete am %s Uhr. ' \
                         % (client_name,
                            (results['ees']['stop'][0]-results['ees']['start'][0]).days,
                            results['ees']['start'][0].strftime("%d/%m/%Y, %H:%M:%S"),
                            results['ees']['stop'][0].strftime("%d/%m/%Y, %H:%M:%S"))
            text_img1 = 'In Abbildung 1 sehen Sie Ihre, in dem Messzeitraum erzeugten, Energiekosten im Vergleich ' \
                        'zu den Energiekosten, bei der ' \
                        'angegebenen PV-Anlage und der Speicherkapazität. Die gesamten Energiekosten, ohne PV und ' \
                        'Speicher belaufen sich auf %s €. Wenn Sie eine PV Anlage mit %s Modulen' \
                        'dazu kaufen, sparen sie im Messzeitraum %s €. <br/> ' \
                        'Wenn Sie sich für eine Kombination ' \
                        'aus PV-Anlage und Batteriespeicher mit %s kWh entscheiden, betragen die Energiekosten %s €. ' \
                        'Sie sparen damit %s € im Vergleich zu einem System ohne PV und Speicher. Die berechneten ' \
                        'Preise verstehen sich ohne Investitionskosten.'\
                        % ('%.1f' % -results['ees']['costs_raw'],
                            self['array'].modules_per_inverter,
                            '%.1f' % (-results['ees']['costs_raw'] + results['ees']['costs_pv']),
                            self['ees'].capacity,
                            '%.1f' % -results['ees']['costs_pv_bat'],
                            (results['ees']['costs_pv_bat'][0] - results['ees']['costs_raw'][0]).round(1))
            text_img2 = 'In Abbildung 2 ist ihr Energieverbrauch und die Erzeugung während eines Tages zu erkennen. Ihr ' \
                        'durchschnittlicher Verbrauch liegt bei ca. %s Watt. Ihren maximalen Verbrauch von %s W ' \
                        'haben Sie um %s Uhr. <br/>' \
                        % ('%.1f' % results['daily']['p_dom'].mean(),
                           '%.1f' % results['daily']['p_dom'].max(),
                           results['daily']['p_dom'].idxmax())
            text_img3 = 'Abbildung 3 zeigt die Energiebilanz in Ihrem Haus. Wenn mehr Energie erzeugt wird, wie sie ' \
                     'Verbrauchen, ist die Energiebilanz positiv. Wenn Sie mehr verbrauchen wie erzeugt wird, ist ' \
                     'die Bilanz negativ. Da ihr Hausanschluss für %s kW ausgelegt ist ergibt sich, unter ' \
                     'Berücksichtigung Ihres individuellen Lastprofils und unter Verwendung eines Lademanagement-' \
                     'Systems, ein tägliches Ladepotential von %s kWh. ' \
                     'Das entspricht einer Fahrleistung von %s km. Bei der angegebenen Strecke je Fahrzeug, könnten Sie damit %s ' \
                     'Fahrzeuge betreiben ohne Ihren Hausanschluss zu überlasten. <br/> ' \
                     'Wenn sie kein Lademanagement installieren, könnten sie maximal %s Fahrzeuge im angegebenen ' \
                     'Zeitraum laden, ohne dabei ihren Hausanschluss zu überlasten. (Vorraussetzung der Gleichzeitigkeit beim Ladevorgang)' \
                     % (results['ev']['house connection point'][0],
                        '%.1f' % results['ev']['energy potential'][0],
                        '%.1f' % results['ev']['km potential'][0],
                        results['ev']['ev potential'][0],
                        results['ev']['ev potential without LMS'][0])
            text_remarks = 'Die oben aufgeführten Berechnungen werden nach besten Gewissen aufgrund der eingegebenen ' \
                           'Angaben berechnet. Größen wie die Ladezeit variieren in der realität täglich, werden ' \
                           'hier jedoch als konstant angenommen.'

            pdf.append(logo)
            pdf.append(Paragraph('%s' % dt.date.today(), styles["Normal_right"]))
            pdf.append(Spacer(1, 12))
            pdf.append(Paragraph(client_name, styles["Normal"]))
            for part in address_parts:
                ptext = '%s' % part.strip()
                pdf.append(Paragraph(ptext, styles["Normal"]))
            pdf.append(Spacer(1, 12))
            pdf.append(Paragraph(title, styles["Heading"]))
            pdf.append(Spacer(1, 24))
            pdf.append(Paragraph(text_intro, styles["Normal"]))
            pdf.append(Spacer(1, 12))
            pdf.append(Paragraph(text_img1, styles["Normal"]))
            pdf.append(Spacer(1, 12))
            pdf.append(img_1)
            pdf.append(Spacer(1, 12))
            pdf.append(Paragraph(text_img2, styles["Normal"]))
            pdf.append(Spacer(1, 12))
            # pdf.append(Table([[img_2, img_3]], colWidths=[275, 275], rowHeights=[220]))
            pdf.append(img_2)
            pdf.append(Spacer(1, 12))
            pdf.append(Paragraph(text_img3, styles["Normal"]))
            pdf.append(Spacer(1, 12))
            pdf.append(img_3)
            pdf.append(Spacer(1, 12))
            pdf.append(Paragraph('Bemerkungen:', styles["Normal_bold"]))
            pdf.append(Paragraph(text_remarks, styles["Normal"]))
            pdf.append(Spacer(1, 24))
            pdf.append(
                Paragraph('mit freundlichen Grüßen und eine sonnige Zukunft wünscht Ihnen ihr ', styles["Normal"]))
            pdf.append(Spacer(1, 12))
            pdf.append(Paragraph('ISC Konstanz', styles["Normal_bold"]))
            pdf.append(Spacer(1, 12))
            doc.build(pdf)
        except ValueError:
            print('PDF not created %s' % ValueError)
        print('PDF created successfully')

    def _create_graphics(self, results_summary, results_total, results):
        from matplotlib import pyplot as plot

        size_large = 30
        size_medium = 27
        size_small = 24
        width = 0.6
        for key, configs in self.system.items():
            if key == 'array':
                fig1, ax1 = plot.subplots(figsize=(10, 10))
                data = [
                    -results['ees']['costs_raw'].values[0].round(1),
                    -results['ees']['costs_pv'].values[0].round(1),
                    -results['ees']['costs_pv_bat'].values[0].round(1),
                ]
                plot1 = ax1.bar(np.arange(3), data, width)
                # ax1.set_title('Energy costs in comparison', fontsize=size_large)
                ax1.set_ylabel('energy costs in €', fontsize=size_medium)
                ax1.tick_params(axis='y', labelsize=size_medium)
                ax1.set_xticks(np.arange(3))
                label = ['RAW', 'mit PV', 'mit PV und EES']
                ax1.set_xticklabels(label, fontsize=size_medium)
                ax1.yaxis.set_ticks_position('left')
                ax1.xaxis.set_ticks_position('bottom')
                ax1.spines['right'].set_visible(False)
                ax1.spines['top'].set_visible(False)
                ax1.bar_label(plot1, padding=3, fontsize=size_small)
                fig1.tight_layout()
                plot.savefig(self._results_dir + '\\' + key)

            if key == 'ees':
                fig2, ax2 = plot.subplots(figsize=(23, 15))
                # ax2.set_title('energy', fontsize=size_large)
                ax2.plot(results['daily'], linewidth=2.5)
                ax2.set_xlim([0, 23])
                ax2.set_xlabel('hour of the day', fontsize=size_medium)
                ax2.set_ylabel('power [KW]', fontsize=size_medium)
                ax2.legend(results['daily'].columns, fontsize=size_small)
                ax2.grid(which='both')
                ax2.grid(b=True, which='major', color='grey', linestyle='-')
                ax2.grid(b=True, which='minor', color='grey', linestyle='--', alpha=0.2)
                ax2.minorticks_on()
                ax2.tick_params(axis='y', labelsize=size_small)
                ax2.tick_params(axis='x', labelsize=size_small)
                plot.savefig(self._results_dir + '\\' + key + '_1')

            if key == 'ev':
                fig3, ax3 = plot.subplots(figsize=(23, 15))
                ax3.plot(results['daily']['p_pv'] - results['daily']['p_dom'], linewidth=2.5)
                ax3.set_xlim([0, 23])
                ax3.set_xlabel('hour of the day', fontsize=size_medium)
                ax3.set_ylabel('power [KW]', fontsize=size_medium)
                ax3.legend(['power balance'], fontsize=size_small)
                ax3.grid(which='both')
                ax3.grid(b=True, which='major', color='grey', linestyle='-')
                ax3.grid(b=True, which='minor', color='grey', linestyle='--', alpha=0.2)
                ax3.minorticks_on()
                ax3.tick_params(axis='y', labelsize=size_small)
                ax3.tick_params(axis='x', labelsize=size_small)
                plot.savefig(self._results_dir + '\\' + key + '_2')


# noinspection PyUnresolvedReferences
def _get(results, key: str, func: Callable, *args, **kwargs) -> pd.DataFrame:
    if results is None or key not in results:
        result = func(*args, **kwargs)
        if results is not None:
            results.set(key, result)
        return result

    return results.get(key)
