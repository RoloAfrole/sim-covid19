import numpy as np

import random
from tqdm import tqdm
from absl import flags

import constant as ct

import datetime
import models


class Initializer(object):
    def __init__(self, config):
        self.config = config

    def create_status(self):
        citys = self.create_citys()
        status = models.Status(citys)
        return status

    def create_srange(self):
        raise NotImplementedError

    def create_citys(self):
        raise NotImplementedError

    def create_manager(self, histroy=None):
        status = self.create_status()
        srange = self.create_srange()
        return models.Manager(status, srange, history=histroy)

    def _create_days(self, start_date, condition_list):
        days = []
        date_format = '%Y/%m/%d'
        tmp_date = start_date
        unit_day = datetime.timedelta(days=1)
        for condition in condition_list:
            while (tmp_date < condition[0]):
                days.append(
                    models.Day(tmp_date.strftime(date_format), condition[1]))
                tmp_date = tmp_date + unit_day

        return days

    def _create_citys(self, condition):
        citys = []
        for c in condition:
            citys.append(models.City(
                name=c['name'],
                peaple=self._create_peaple(c['peaple']),
                areas=self._create_areas(c['areas']),
                move_out=self._create_move_out(c['move_out']),
                p_remove=c['p_remove']))
        return citys

    def _create_peaple(self, condition):
        peaple = []
        ids = 0
        for c in condition:
            pop = c['population']
            total_pop = [int(r*pop) for r in c['condition_prop']]
            for condition_idx, tp in enumerate(total_pop):
                for i in range(tp):
                    peaple.append(
                        models.Person(
                            id=ids,
                            group=c['id'],
                            condition=condition_idx,
                            group_name=c['name']
                        )
                    )
                    ids += 1

        return peaple

    def _create_areas(self, condition):
        areas = []
        for c in condition:
            areas.append(models.Area(c['name'], c['id'], c['patterns']))
        return areas

    def _create_move_out(self, condition):
        move_out = models.MoveOut(condition)
        return move_out

    @staticmethod
    def person_group(id, group_name, population, conditons):
        dic = {}
        dic['id'] = id
        dic['name'] = group_name
        dic['population'] = population
        dic['condition_prop'] = conditons
        return dic

    @staticmethod
    def area_group(id, group_name, patterns):
        dic = {}
        dic['id'] = id
        dic['name'] = group_name
        dic['patterns'] = patterns
        return dic


class Default_Izer(Initializer):
    def __init__(self, config):
        super(Default_Izer, self).__init__(config)

    def create_srange(self):
        from settings import Day_Groups
        start_date = datetime.date(2020, 3, 1)
        condition_list = [
            [datetime.date(2020, 4, 7), Day_Groups['before_SoE']],
            [datetime.date(2020, 4, 14), Day_Groups['1week_SoE']],
            [datetime.date(2020, 4, 21), Day_Groups['2week_SoE']],
            [datetime.date(2020, 4, 28), Day_Groups['3week_SoE']],
            [datetime.date(2020, 5, 5), Day_Groups['4week_SoE']],
            [datetime.date(2020, 5, 31), Day_Groups['5__week_SoE']],
            [datetime.date(2020, 7, 31), Day_Groups['after_SoE']],
        ]

        days = self._create_days(start_date, condition_list)
        srange = models.SimRange(days,
                                 start_position=0,
                                 end_position=len(days))
        return srange

    def create_citys(self):
        condition = [
            {
                'name':
                'Tokyo',
                'p_remove':
                0.1,
                'peaple': [
                    self.person_group(0, 'general1', 7000000,
                                      [0.999996, 0.000002, 0.000002, 0.0]),
                    self.person_group(1, 'general2', 7000000,
                                      [0.999996, 0.000002, 0.000002, 0.0]),
                ],
                'areas': [
                    self.area_group(
                        0, 'ncloud', {
                            'before_SoE': [[0.0002] * 24, [0.0002] * 24],
                            '1week_SoE': [[0.0002] * 24, [0.0002] * 24],
                            '2week_SoE': [[0.0002] * 24, [0.0002] * 24],
                            '3week_SoE': [[0.0002] * 24, [0.0002] * 24],
                            '4week_SoE': [[0.0002] * 24, [0.0002] * 24],
                            '5__week_SoE': [[0.0002] * 24, [0.0002] * 24],
                            'after_SoE': [[0.0002] * 24, [0.0002] * 24],
                        }),
                    self.area_group(
                        1, 'mid', {
                            'before_SoE': [[0.002] * 24, [0.002] * 24],
                            '1week_SoE': [[0.002] * 24, [0.002] * 24],
                            '2week_SoE': [[0.002] * 24, [0.002] * 24],
                            '3week_SoE': [[0.002] * 24, [0.002] * 24],
                            '4week_SoE': [[0.002] * 24, [0.002] * 24],
                            '5__week_SoE': [[0.002] * 24, [0.002] * 24],
                            'after_SoE': [[0.002] * 24, [0.002] * 24],
                        }),
                    self.area_group(
                        2, 'croud', {
                            'before_SoE': [[0.02] * 24, [0.02] * 24],
                            '1week_SoE': [[0.02] * 24, [0.02] * 24],
                            '2week_SoE': [[0.02] * 24, [0.02] * 24],
                            '3week_SoE': [[0.02] * 24, [0.02] * 24],
                            '4week_SoE': [[0.02] * 24, [0.02] * 24],
                            '5__week_SoE': [[0.02] * 24, [0.02] * 24],
                            'after_SoE': [[0.02] * 24, [0.02] * 24],
                        }),
                ],
                'move_out': {
                    'before_SoE': {
                        'Outer': 0.1
                    },
                    '1week_SoE': {
                        'Outer': 0.08
                    },
                    '2week_SoE': {
                        'Outer': 0.06
                    },
                    '3week_SoE': {
                        'Outer': 0.05
                    },
                    '4week_SoE': {
                        'Outer': 0.05
                    },
                    '5__week_SoE': {
                        'Outer': 0.05
                    },
                    'after_SoE': {
                        'Outer': 0.1
                    },
                },
            },
            {
                'name':
                'Outer',
                'p_remove':
                0.1,
                'peaple': [
                    # self.person_group(0, 'general1', 7000,
                    #                   [0.996, 0.002, 0.002, 0.0]),
                    # self.person_group(1, 'general2', 7000,
                    #                   [0.996, 0.002, 0.002, 0.0]),
                    # self.person_group(0, 'general1', 700,
                    #                   [0.96, 0.02, 0.02, 0.0]),
                    # self.person_group(1, 'general2', 700,
                    #                   [0.96, 0.02, 0.02, 0.0]),
                    self.person_group(0, 'general1', 7000000,
                                      [0.999996, 0.000002, 0.000002, 0.0]),
                    self.person_group(1, 'general2', 7000000,
                                      [0.999996, 0.000002, 0.000002, 0.0]),
                ],
                'areas': [
                    self.area_group(
                        0, 'ncloud', {
                            'before_SoE': [[0.0002] * 24, [0.0002] * 24],
                            '1week_SoE': [[0.0002] * 24, [0.0002] * 24],
                            '2week_SoE': [[0.0002] * 24, [0.0002] * 24],
                            '3week_SoE': [[0.0002] * 24, [0.0002] * 24],
                            '4week_SoE': [[0.0002] * 24, [0.0002] * 24],
                            '5__week_SoE': [[0.0002] * 24, [0.0002] * 24],
                            'after_SoE': [[0.0002] * 24, [0.0002] * 24],
                        }),
                    self.area_group(
                        1, 'mid', {
                            'before_SoE': [[0.002] * 24, [0.002] * 24],
                            '1week_SoE': [[0.002] * 24, [0.002] * 24],
                            '2week_SoE': [[0.002] * 24, [0.002] * 24],
                            '3week_SoE': [[0.002] * 24, [0.002] * 24],
                            '4week_SoE': [[0.002] * 24, [0.002] * 24],
                            '5__week_SoE': [[0.002] * 24, [0.002] * 24],
                            'after_SoE': [[0.002] * 24, [0.002] * 24],
                        }),
                    self.area_group(
                        2, 'croud', {
                            'before_SoE': [[0.02] * 24, [0.02] * 24],
                            '1week_SoE': [[0.02] * 24, [0.02] * 24],
                            '2week_SoE': [[0.02] * 24, [0.02] * 24],
                            '3week_SoE': [[0.02] * 24, [0.02] * 24],
                            '4week_SoE': [[0.02] * 24, [0.02] * 24],
                            '5__week_SoE': [[0.02] * 24, [0.02] * 24],
                            'after_SoE': [[0.02] * 24, [0.02] * 24],
                        }),
                ],
                'move_out': {
                    'before_SoE': {
                        'Tokyo': 0.2
                    },
                    '1week_SoE': {
                        'Tokyo': 0.18
                    },
                    '2week_SoE': {
                        'Tokyo': 0.12
                    },
                    '3week_SoE': {
                        'Tokyo': 0.12
                    },
                    '4week_SoE': {
                        'Tokyo': 0.12
                    },
                    '5__week_SoE': {
                        'Tokyo': 0.12
                    },
                    'after_SoE': {
                        'Tokyo': 0.2
                    },
                },
            }
        ]

        return self._create_citys(condition)
