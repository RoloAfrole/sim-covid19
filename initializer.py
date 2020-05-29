import numpy as np

import random
from tqdm import tqdm
from absl import flags

import constant as ct

import datetime
import models

from multiprocessing.managers import SharedMemoryManager


class Initializer(object):
    def __init__(self, config):
        self.config = config

    def create_status(self, smm):
        citys = self.create_citys(smm)
        status = models.Status(citys, smm)
        return status

    def create_srange(self):
        raise NotImplementedError

    def create_citys(self, smm):
        raise NotImplementedError

    def create_manager(self, histroy=None):
        smm = SharedMemoryManager()
        smm.start()
        status = self.create_status(smm)
        srange = self.create_srange()
        return models.Manager(status, srange, smm, history=histroy)

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

    def _create_citys(self, condition, smm):
        citys = []
        for c in condition:
            citys.append(models.City(
                name=c['name'],
                peaple=self._create_peaple(c['peaple'], smm),
                areas=self._create_areas(c['areas']),
                move_out=self._create_move_out(c['move_out']),
                p_remove=c['p_remove'],
                smm=smm))
        return citys

    def _create_peaple(self, condition, smm):
        peaple = []
        ids = 0

        for c in condition:
            pop = c['population']
            total_pop = [round(r*pop) for r in c['condition_prop']]
            for condition_idx, tp in enumerate(total_pop):
                for i in range(tp):
                    peaple.append([ids, c['id'], condition_idx, -1])
                    ids += 1

        na = np.array(peaple, dtype=int)
        del peaple
        shm = smm.SharedMemory(size=na.nbytes)
        shm_array = np.ndarray(na.shape, dtype=na.dtype, buffer=shm.buf)
        shm_array[:] = na[:]
        del na

        return shm_array, shm.name, shm

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
            [datetime.date(2020, 3, 20), Day_Groups['before_SoE']],
            [datetime.date(2020, 4, 7), Day_Groups['before_2_SoE']],
            [datetime.date(2020, 4, 14), Day_Groups['1week_SoE']],
            [datetime.date(2020, 4, 21), Day_Groups['2week_SoE']],
            [datetime.date(2020, 4, 28), Day_Groups['3week_SoE']],
            [datetime.date(2020, 5, 5), Day_Groups['4week_SoE']],
            [datetime.date(2020, 5, 25), Day_Groups['5__week_SoE']],
            [datetime.date(2020, 7, 31), Day_Groups['after_SoE']],
        ]

        days = self._create_days(start_date, condition_list)
        srange = models.SimRange(days,
                                 start_position=0,
                                 end_position=len(days))
        return srange

    def create_citys(self, smm):
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

        return self._create_citys(condition, smm)


class Detailed_Izer(Default_Izer):
    def __init__(self, config):
        super(Detailed_Izer, self).__init__(config)

    def create_citys(self, smm):
        condition = [
            {
                'name':
                'Tokyo',
                'p_remove':
                0.1,
                'peaple': [
                    # 1
                    self.person_group(2, '00to19_1', 2080000,
                                      [0.9999984, 0.0000008, 0.0000008, 0.0]),
                    self.person_group(3, '20to44_1', 4780000,
                                      [0.9999934, 0.0000033, 0.0000033, 0.0]),
                    self.person_group(4, '45to64_1', 3390000,
                                      [0.9999928, 0.0000036, 0.0000036, 0.0]),
                    self.person_group(5, '65to99_1', 3000000,
                                      [0.9999934, 0.0000034, 0.0000034, 0.0]),
                    # # 2
                    # self.person_group(2, '00to19_1', 2080000,
                    #                   [0.9999984, 0.0000008, 0.0000008, 0.0]),
                    # self.person_group(3, '20to44_1', 1912000,
                    #                   [0.9999934, 0.0000033, 0.0000033, 0.0]),
                    # self.person_group(4, '45to64_1', 1356000,
                    #                   [0.9999928, 0.0000036, 0.0000036, 0.0]),
                    # self.person_group(5, '65to99_1', 3000000,
                    #                   [0.9999934, 0.0000034, 0.0000034, 0.0]),
                    # self.person_group(6, '20to44_2_tele1', 1195000,
                    #                   [0.9999934, 0.0000033, 0.0000033, 0.0]),
                    # self.person_group(7, '20to44_2_tele2', 1673000,
                    #                   [0.9999934, 0.0000033, 0.0000033, 0.0]),
                    # self.person_group(8, '45to64_2_tele1', 8475000,
                    #                   [0.9999928, 0.0000036, 0.0000036, 0.0]),
                    # self.person_group(9, '45to64_2_tele2', 1186500,
                    #                   [0.9999928, 0.0000036, 0.0000036, 0.0]),
                    # # 3
                    # self.person_group(2, '00to19_1', 1872000,
                    #                   [0.9999984, 0.0000008, 0.0000008, 0.0]),
                    # self.person_group(3, '20to44_1', 1434000,
                    #                   [0.9999934, 0.0000033, 0.0000033, 0.0]),
                    # self.person_group(4, '45to64_1', 1017000,
                    #                   [0.9999928, 0.0000036, 0.0000036, 0.0]),
                    # self.person_group(5, '65to99_1', 2700000,
                    #                   [0.9999934, 0.0000034, 0.0000034, 0.0]),
                    # self.person_group(6, '20to44_2_tele1', 1195000,
                    #                   [0.9999934, 0.0000033, 0.0000033, 0.0]),
                    # self.person_group(7, '20to44_2_tele2', 1673000,
                    #                   [0.9999934, 0.0000033, 0.0000033, 0.0]),
                    # self.person_group(8, '45to64_2_tele1', 8475000,
                    #                   [0.9999928, 0.0000036, 0.0000036, 0.0]),
                    # self.person_group(9, '45to64_2_tele2', 1186500,
                    #                   [0.9999928, 0.0000036, 0.0000036, 0.0]),
                    # self.person_group(10, '00to19_3_goout', 208000,
                    #                   [0.9999984, 0.0000008, 0.0000008, 0.0]),
                    # self.person_group(11, '20to44_3_goout', 478000,
                    #                   [0.9999934, 0.0000033, 0.0000033, 0.0]),
                    # self.person_group(12, '45to64_3_goout', 339000,
                    #                   [0.9999928, 0.0000036, 0.0000036, 0.0]),
                    # self.person_group(13, '65to99_3_goout', 300000,
                    #                   [0.9999934, 0.0000034, 0.0000034, 0.0]),
                ],
                'areas': [
                    self.area_group(
                        0, 'ncroud', {
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
                    self.person_group(2, '00to19_1', 2080000,
                                      [0.9999992, 0.0000004, 0.0000004, 0.0]),
                    self.person_group(3, '20to44_1', 4780000,
                                      [0.9999966, 0.0000016, 0.0000016, 0.0]),
                    self.person_group(4, '45to64_1', 3390000,
                                      [0.9999964, 0.0000018, 0.0000018, 0.0]),
                    self.person_group(5, '65to99_1', 3000000,
                                      [0.9999968, 0.0000017, 0.0000017, 0.0]),
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
                        'Tokyo': 0.14
                    },
                    'after_SoE': {
                        'Tokyo': 0.18
                    },
                },
            }
        ]

        return self._create_citys(condition, smm)


class Detailed_TL_Izer(Default_Izer):
    def __init__(self, config):
        super(Detailed_TL_Izer, self).__init__(config)

    def create_citys(self, smm):
        condition = [
            {
                'name':
                'Tokyo',
                'p_remove':
                0.1,
                'peaple': [
                    # # 1
                    # self.person_group(0, 'general1', 7000000,
                    #                   [0.999996, 0.000002, 0.000002, 0.0]),
                    # self.person_group(1, 'general2', 6000000,
                    #                   [0.999996, 0.000002, 0.000002, 0.0]),
                    # self.person_group(2, '00to19_1', 2080000,
                    #                   [0.9999984, 0.0000008, 0.0000008, 0.0]),
                    # self.person_group(3, '20to44_1', 4780000,
                    #                   [0.9999934, 0.0000033, 0.0000033, 0.0]),
                    # self.person_group(4, '45to64_1', 3390000,
                    #                   [0.9999928, 0.0000036, 0.0000036, 0.0]),
                    # self.person_group(5, '65to99_1', 3000000,
                    #                   [0.9999934, 0.0000034, 0.0000034, 0.0]),
                    # 2
                    self.person_group(2, '00to19_1', 2080000,
                                      [0.9999984, 0.0000008, 0.0000008, 0.0]),
                    self.person_group(3, '20to44_1', 1912000,
                                      [0.9999934, 0.0000033, 0.0000033, 0.0]),
                    self.person_group(4, '45to64_1', 1356000,
                                      [0.9999928, 0.0000036, 0.0000036, 0.0]),
                    self.person_group(5, '65to99_1', 3000000,
                                      [0.9999934, 0.0000034, 0.0000034, 0.0]),
                    self.person_group(6, '20to44_2_tele1', 1195000,
                                      [0.9999934, 0.0000033, 0.0000033, 0.0]),
                    self.person_group(7, '20to44_2_tele2', 1673000,
                                      [0.9999934, 0.0000033, 0.0000033, 0.0]),
                    self.person_group(8, '45to64_2_tele1', 8475000,
                                      [0.9999928, 0.0000036, 0.0000036, 0.0]),
                    self.person_group(9, '45to64_2_tele2', 1186500,
                                      [0.9999928, 0.0000036, 0.0000036, 0.0]),
                    # # 3
                    # self.person_group(2, '00to19_1', 1872000,
                    #                   [0.9999984, 0.0000008, 0.0000008, 0.0]),
                    # self.person_group(3, '20to44_1', 1434000,
                    #                   [0.9999934, 0.0000033, 0.0000033, 0.0]),
                    # self.person_group(4, '45to64_1', 1017000,
                    #                   [0.9999928, 0.0000036, 0.0000036, 0.0]),
                    # self.person_group(5, '65to99_1', 2700000,
                    #                   [0.9999934, 0.0000034, 0.0000034, 0.0]),
                    # self.person_group(6, '20to44_2_tele1', 1195000,
                    #                   [0.9999934, 0.0000033, 0.0000033, 0.0]),
                    # self.person_group(7, '20to44_2_tele2', 1673000,
                    #                   [0.9999934, 0.0000033, 0.0000033, 0.0]),
                    # self.person_group(8, '45to64_2_tele1', 8475000,
                    #                   [0.9999928, 0.0000036, 0.0000036, 0.0]),
                    # self.person_group(9, '45to64_2_tele2', 1186500,
                    #                   [0.9999928, 0.0000036, 0.0000036, 0.0]),
                    # self.person_group(10, '00to19_3_goout', 208000,
                    #                   [0.9999984, 0.0000008, 0.0000008, 0.0]),
                    # self.person_group(11, '20to44_3_goout', 478000,
                    #                   [0.9999934, 0.0000033, 0.0000033, 0.0]),
                    # self.person_group(12, '45to64_3_goout', 339000,
                    #                   [0.9999928, 0.0000036, 0.0000036, 0.0]),
                    # self.person_group(13, '65to99_3_goout', 300000,
                    #                   [0.9999934, 0.0000034, 0.0000034, 0.0]),
                ],
                'areas': [
                    self.area_group(
                        0, 'ncroud', {
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
                    self.person_group(2, '00to19_1', 2080000,
                                      [0.9999992, 0.0000004, 0.0000004, 0.0]),
                    self.person_group(3, '20to44_1', 1912000,
                                      [0.9999968, 0.0000016, 0.0000016, 0.0]),
                    self.person_group(4, '45to64_1', 1356000,
                                      [0.9999964, 0.0000018, 0.0000018, 0.0]),
                    self.person_group(5, '65to99_1', 3000000,
                                      [0.9999966, 0.0000017, 0.0000017, 0.0]),
                    self.person_group(6, '20to44_2_tele1', 1195000,
                                      [0.9999968, 0.0000016, 0.0000016, 0.0]),
                    self.person_group(7, '20to44_2_tele2', 1673000,
                                      [0.9999968, 0.0000016, 0.0000016, 0.0]),
                    self.person_group(8, '45to64_2_tele1', 8475000,
                                      [0.9999964, 0.0000018, 0.0000018, 0.0]),
                    self.person_group(9, '45to64_2_tele2', 1186500,
                                      [0.9999964, 0.0000018, 0.0000018, 0.0]),
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
                        'Tokyo': 0.14
                    },
                    'after_SoE': {
                        'Tokyo': 0.18
                    },
                },
            }
        ]

        return self._create_citys(condition, smm)


class Detailed_TL_GO_Izer(Default_Izer):
    def __init__(self, config):
        super(Detailed_TL_GO_Izer, self).__init__(config)

    def create_citys(self, smm):
        condition = [
            {
                'name':
                'Tokyo',
                'p_remove':
                0.1,
                'peaple': [
                    # 1
                    # self.person_group(0, 'general1', 7000000,
                    #                   [0.999996, 0.000002, 0.000002, 0.0]),
                    # self.person_group(1, 'general2', 6000000,
                    #                   [0.999996, 0.000002, 0.000002, 0.0]),
                    # self.person_group(2, '00to19_1', 2080000,
                    #                   [0.9999984, 0.0000008, 0.0000008, 0.0]),
                    # self.person_group(3, '20to44_1', 4780000,
                    #                   [0.9999934, 0.0000033, 0.0000033, 0.0]),
                    # self.person_group(4, '45to64_1', 3390000,
                    #                   [0.9999928, 0.0000036, 0.0000036, 0.0]),
                    # self.person_group(5, '65to99_1', 3000000,
                    #                   [0.9999934, 0.0000034, 0.0000034, 0.0]),
                    # 2
                    # self.person_group(2, '00to19_1', 2080000,
                    #                   [0.9999984, 0.0000008, 0.0000008, 0.0]),
                    # self.person_group(3, '20to44_1', 1912000,
                    #                   [0.9999934, 0.0000033, 0.0000033, 0.0]),
                    # self.person_group(4, '45to64_1', 1356000,
                    #                   [0.9999928, 0.0000036, 0.0000036, 0.0]),
                    # self.person_group(5, '65to99_1', 3000000,
                    #                   [0.9999934, 0.0000034, 0.0000034, 0.0]),
                    # self.person_group(6, '20to44_2_tele1', 1195000,
                    #                   [0.9999934, 0.0000033, 0.0000033, 0.0]),
                    # self.person_group(7, '20to44_2_tele2', 1673000,
                    #                   [0.9999934, 0.0000033, 0.0000033, 0.0]),
                    # self.person_group(8, '45to64_2_tele1', 8475000,
                    #                   [0.9999928, 0.0000036, 0.0000036, 0.0]),
                    # self.person_group(9, '45to64_2_tele2', 1186500,
                    #                   [0.9999928, 0.0000036, 0.0000036, 0.0]),
                    # 3
                    self.person_group(2, '00to19_1', 1872000,
                                      [0.9999984, 0.0000008, 0.0000008, 0.0]),
                    self.person_group(3, '20to44_1', 1434000,
                                      [0.9999934, 0.0000033, 0.0000033, 0.0]),
                    self.person_group(4, '45to64_1', 1017000,
                                      [0.9999928, 0.0000036, 0.0000036, 0.0]),
                    self.person_group(5, '65to99_1', 2700000,
                                      [0.9999934, 0.0000034, 0.0000034, 0.0]),
                    self.person_group(6, '20to44_2_tele1', 1195000,
                                      [0.9999934, 0.0000033, 0.0000033, 0.0]),
                    self.person_group(7, '20to44_2_tele2', 1673000,
                                      [0.9999934, 0.0000033, 0.0000033, 0.0]),
                    self.person_group(8, '45to64_2_tele1', 8475000,
                                      [0.9999928, 0.0000036, 0.0000036, 0.0]),
                    self.person_group(9, '45to64_2_tele2', 1186500,
                                      [0.9999928, 0.0000036, 0.0000036, 0.0]),
                    self.person_group(10, '00to19_3_goout', 208000,
                                      [0.9999984, 0.0000008, 0.0000008, 0.0]),
                    self.person_group(11, '20to44_3_goout', 478000,
                                      [0.9999934, 0.0000033, 0.0000033, 0.0]),
                    self.person_group(12, '45to64_3_goout', 339000,
                                      [0.9999928, 0.0000036, 0.0000036, 0.0]),
                    self.person_group(13, '65to99_3_goout', 300000,
                                      [0.9999934, 0.0000034, 0.0000034, 0.0]),
                ],
                'areas': [
                    self.area_group(
                        0, 'ncroud', {
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
                    self.person_group(2, '00to19_1', 1872000,
                                      [0.9999992, 0.0000004, 0.0000004, 0.0]),
                    self.person_group(3, '20to44_1', 1434000,
                                      [0.9999968, 0.0000016, 0.0000016, 0.0]),
                    self.person_group(4, '45to64_1', 1017000,
                                      [0.9999964, 0.0000018, 0.0000018, 0.0]),
                    self.person_group(5, '65to99_1', 2700000,
                                      [0.9999966, 0.0000017, 0.0000017, 0.0]),
                    self.person_group(6, '20to44_2_tele1', 1195000,
                                      [0.9999968, 0.0000016, 0.0000016, 0.0]),
                    self.person_group(7, '20to44_2_tele2', 1673000,
                                      [0.9999968, 0.0000016, 0.0000016, 0.0]),
                    self.person_group(8, '45to64_2_tele1', 8475000,
                                      [0.9999964, 0.0000018, 0.0000018, 0.0]),
                    self.person_group(9, '45to64_2_tele2', 1186500,
                                      [0.9999964, 0.0000018, 0.0000018, 0.0]),
                    self.person_group(10, '00to19_3_goout', 208000,
                                      [0.9999992, 0.0000004, 0.0000004, 0.0]),
                    self.person_group(11, '20to44_3_goout', 478000,
                                      [0.9999968, 0.0000016, 0.0000016, 0.0]),
                    self.person_group(12, '45to64_3_goout', 339000,
                                      [0.9999964, 0.0000018, 0.0000018, 0.0]),
                    self.person_group(13, '65to99_3_goout', 300000,
                                      [0.9999966, 0.0000017, 0.0000017, 0.0]),
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
                        'Tokyo': 0.14
                    },
                    'after_SoE': {
                        'Tokyo': 0.18
                    },
                },
            }
        ]

        return self._create_citys(condition, smm)


class Detailed_I10_Izer(Default_Izer):
    def __init__(self, config):
        super(Detailed_I10_Izer, self).__init__(config)

    def create_citys(self, smm):
        condition = [
            {
                'name':
                'Tokyo',
                'p_remove':
                0.1,
                'peaple': [
                    # 1
                    self.person_group(2, '00to19_1', 2080000,
                                      [0.9999912, 0.000008, 0.0000008, 0.0]),
                    self.person_group(3, '20to44_1', 4780000,
                                      [0.9999637, 0.000033, 0.0000033, 0.0]),
                    self.person_group(4, '45to64_1', 3390000,
                                      [0.9999604, 0.000036, 0.0000036, 0.0]),
                    self.person_group(5, '65to99_1', 3000000,
                                      [0.9999626, 0.000034, 0.0000034, 0.0]),
                ],
                'areas': [
                    self.area_group(
                        0, 'ncroud', {
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
                    self.person_group(2, '00to19_1', 2080000,
                                      [0.9999956, 0.000004, 0.0000004, 0.0]),
                    self.person_group(3, '20to44_1', 4780000,
                                      [0.9999824, 0.000016, 0.0000016, 0.0]),
                    self.person_group(4, '45to64_1', 3390000,
                                      [0.9999802, 0.000018, 0.0000018, 0.0]),
                    self.person_group(5, '65to99_1', 3000000,
                                      [0.9999813, 0.000017, 0.0000017, 0.0]),
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
                        'Tokyo': 0.14
                    },
                    'after_SoE': {
                        'Tokyo': 0.18
                    },
                },
            }
        ]

        return self._create_citys(condition, smm)


class Detailed_I10_TL_Izer(Default_Izer):
    def __init__(self, config):
        super(Detailed_I10_TL_Izer, self).__init__(config)

    def create_citys(self, smm):
        condition = [{
            'name':
            'Tokyo',
            'p_remove':
            0.1,
            'peaple': [
                self.person_group(2, '00to19_1', 2080000,
                                  [0.9999912, 0.000008, 0.0000008, 0.0]),
                self.person_group(3, '20to44_1', 1912000,
                                  [0.9999637, 0.000033, 0.0000033, 0.0]),
                self.person_group(4, '45to64_1', 1356000,
                                  [0.9999604, 0.000036, 0.0000036, 0.0]),
                self.person_group(5, '65to99_1', 3000000,
                                  [0.9999626, 0.000034, 0.0000034, 0.0]),
                self.person_group(6, '20to44_2_tele1', 1195000,
                                  [0.9999637, 0.000033, 0.0000033, 0.0]),
                self.person_group(7, '20to44_2_tele2', 1673000,
                                  [0.9999637, 0.000033, 0.0000033, 0.0]),
                self.person_group(8, '45to64_2_tele1', 8475000,
                                  [0.9999604, 0.000036, 0.0000036, 0.0]),
                self.person_group(9, '45to64_2_tele2', 1186500,
                                  [0.9999604, 0.000036, 0.0000036, 0.0]),
            ],
            'areas': [
                self.area_group(
                    0, 'ncroud', {
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
        }, {
            'name':
            'Outer',
            'p_remove':
            0.1,
            'peaple': [
                self.person_group(2, '00to19_1', 2080000,
                                  [0.9999956, 0.000004, 0.0000004, 0.0]),
                self.person_group(3, '20to44_1', 1912000,
                                  [0.9999824, 0.000016, 0.0000016, 0.0]),
                self.person_group(4, '45to64_1', 1356000,
                                  [0.9999802, 0.000018, 0.0000018, 0.0]),
                self.person_group(5, '65to99_1', 3000000,
                                  [0.9999813, 0.000017, 0.0000017, 0.0]),
                self.person_group(6, '20to44_2_tele1', 1195000,
                                  [0.9999824, 0.000016, 0.0000016, 0.0]),
                self.person_group(7, '20to44_2_tele2', 1673000,
                                  [0.9999824, 0.000016, 0.0000016, 0.0]),
                self.person_group(8, '45to64_2_tele1', 8475000,
                                  [0.9999802, 0.000018, 0.0000018, 0.0]),
                self.person_group(9, '45to64_2_tele2', 1186500,
                                  [0.9999802, 0.000018, 0.0000018, 0.0]),
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
                    'Tokyo': 0.14
                },
                'after_SoE': {
                    'Tokyo': 0.18
                },
            },
        }]

        return self._create_citys(condition, smm)


class Detailed_I10_TL_GO_Izer(Default_Izer):
    def __init__(self, config):
        super(Detailed_I10_TL_GO_Izer, self).__init__(config)

    def create_citys(self, smm):
        condition = [{
            'name':
            'Tokyo',
            'p_remove':
            0.1,
            'peaple': [
                self.person_group(2, '00to19_1', 1872000,
                                  [0.9999912, 0.000008, 0.0000008, 0.0]),
                self.person_group(3, '20to44_1', 1434000,
                                  [0.9999637, 0.000033, 0.0000033, 0.0]),
                self.person_group(4, '45to64_1', 1017000,
                                  [0.9999604, 0.000036, 0.0000036, 0.0]),
                self.person_group(5, '65to99_1', 2700000,
                                  [0.9999626, 0.000034, 0.0000034, 0.0]),
                self.person_group(6, '20to44_2_tele1', 1195000,
                                  [0.9999637, 0.000033, 0.0000033, 0.0]),
                self.person_group(7, '20to44_2_tele2', 1673000,
                                  [0.9999637, 0.000033, 0.0000033, 0.0]),
                self.person_group(8, '45to64_2_tele1', 8475000,
                                  [0.9999604, 0.000036, 0.0000036, 0.0]),
                self.person_group(9, '45to64_2_tele2', 1186500,
                                  [0.9999604, 0.000036, 0.0000036, 0.0]),
                self.person_group(10, '00to19_3_goout', 208000,
                                  [0.9999912, 0.000008, 0.0000008, 0.0]),
                self.person_group(11, '20to44_3_goout', 478000,
                                  [0.9999637, 0.000033, 0.0000033, 0.0]),
                self.person_group(12, '45to64_3_goout', 339000,
                                  [0.9999604, 0.000036, 0.0000036, 0.0]),
                self.person_group(13, '65to99_3_goout', 300000,
                                  [0.9999626, 0.000034, 0.0000034, 0.0]),
            ],
            'areas': [
                self.area_group(
                    0, 'ncroud', {
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
        }, {
            'name':
            'Outer',
            'p_remove':
            0.1,
            'peaple': [
                self.person_group(2, '00to19_1', 1872000,
                                  [0.9999956, 0.000004, 0.0000004, 0.0]),
                self.person_group(3, '20to44_1', 1434000,
                                  [0.9999824, 0.000016, 0.0000016, 0.0]),
                self.person_group(4, '45to64_1', 1017000,
                                  [0.9999802, 0.000018, 0.0000018, 0.0]),
                self.person_group(5, '65to99_1', 2700000,
                                  [0.9999813, 0.000017, 0.0000017, 0.0]),
                self.person_group(6, '20to44_2_tele1', 1195000,
                                  [0.9999824, 0.000016, 0.0000016, 0.0]),
                self.person_group(7, '20to44_2_tele2', 1673000,
                                  [0.9999824, 0.000016, 0.0000016, 0.0]),
                self.person_group(8, '45to64_2_tele1', 8475000,
                                  [0.9999802, 0.000018, 0.0000018, 0.0]),
                self.person_group(9, '45to64_2_tele2', 1186500,
                                  [0.9999802, 0.000018, 0.0000018, 0.0]),
                self.person_group(10, '00to19_3_goout', 208000,
                                  [0.9999956, 0.000004, 0.0000004, 0.0]),
                self.person_group(11, '20to44_3_goout', 478000,
                                  [0.9999824, 0.000016, 0.0000016, 0.0]),
                self.person_group(12, '45to64_3_goout', 339000,
                                  [0.9999802, 0.000018, 0.0000018, 0.0]),
                self.person_group(13, '65to99_3_goout', 300000,
                                  [0.9999813, 0.000017, 0.0000017, 0.0]),
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
                    'Tokyo': 0.14
                },
                'after_SoE': {
                    'Tokyo': 0.18
                },
            },
        }]

        return self._create_citys(condition, smm)


class Izer_kanto(Default_Izer):
    def __init__(self, config):
        super(Izer_kanto, self).__init__(config)

    def create_citys(self, smm):
        condition = [
            {
                'name':
                'Tokyo',
                'p_remove':
                0.1,
                'peaple': [
                    # 1
                    self.person_group(2, '00to19_1', 1, [2099983, 16, 2, 0.0]),
                    self.person_group(3, '20to44_1', 1,
                                      [4799828, 156, 16, 0.0]),
                    self.person_group(4, '45to64_1', 1,
                                      [3399871, 117, 12, 0.0]),
                    self.person_group(5, '65to99_1', 1,
                                      [2999888, 101, 10, 0.0]),
                ],
                'areas': [
                    self.area_group(
                        0, 'ncroud', {
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
                        'Chiba': 0.016,
                        'Kanagawa': 0.016,
                        'Saitama': 0.016,
                        'Gunma': 0.016,
                        'Tochigi': 0.016,
                        'Ibaraki': 0.016,
                    },
                    '1week_SoE': {
                        'Chiba': 0.013,
                        'Kanagawa': 0.013,
                        'Saitama': 0.013,
                        'Gunma': 0.013,
                        'Tochigi': 0.013,
                        'Ibaraki': 0.013,
                    },
                    '2week_SoE': {
                        'Chiba': 0.01,
                        'Kanagawa': 0.01,
                        'Saitama': 0.01,
                        'Gunma': 0.01,
                        'Tochigi': 0.01,
                        'Ibaraki': 0.01,
                    },
                    '3week_SoE': {
                        'Chiba': 0.008,
                        'Kanagawa': 0.008,
                        'Saitama': 0.008,
                        'Gunma': 0.008,
                        'Tochigi': 0.008,
                        'Ibaraki': 0.008,
                    },
                    '4week_SoE': {
                        'Chiba': 0.008,
                        'Kanagawa': 0.008,
                        'Saitama': 0.008,
                        'Gunma': 0.008,
                        'Tochigi': 0.008,
                        'Ibaraki': 0.008,
                    },
                    '5__week_SoE': {
                        'Chiba': 0.008,
                        'Kanagawa': 0.008,
                        'Saitama': 0.008,
                        'Gunma': 0.008,
                        'Tochigi': 0.008,
                        'Ibaraki': 0.008,
                    },
                    'after_SoE': {
                        'Chiba': 0.016,
                        'Kanagawa': 0.016,
                        'Saitama': 0.016,
                        'Gunma': 0.016,
                        'Tochigi': 0.016,
                        'Ibaraki': 0.016,
                    },
                },
            },
            {
                'name':
                'Chiba',
                'p_remove':
                0.1,
                'peaple': [
                    self.person_group(2, '00to19_1', 1, [1099992, 7, 1, 0.0]),
                    self.person_group(3, '20to44_1', 1, [1899925, 68, 7, 0.0]),
                    self.person_group(4, '45to64_1', 1, [1699944, 51, 5, 0.0]),
                    self.person_group(5, '65to99_1', 1, [1699951, 44, 4, 0.0]),
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
                        'Tokyo': 0.1,
                        'Kanagawa': 0.04,
                        'Saitama': 0.04,
                        'Gunma': 0.01,
                        'Tochigi': 0.01,
                        'Ibaraki': 0.04
                    },
                    '1week_SoE': {
                        'Tokyo': 0.09,
                        'Kanagawa': 0.036,
                        'Saitama': 0.036,
                        'Gunma': 0.009,
                        'Tochigi': 0.009,
                        'Ibaraki': 0.036
                    },
                    '2week_SoE': {
                        'Tokyo': 0.06,
                        'Kanagawa': 0.024,
                        'Saitama': 0.024,
                        'Gunma': 0.006,
                        'Tochigi': 0.006,
                        'Ibaraki': 0.024
                    },
                    '3week_SoE': {
                        'Tokyo': 0.06,
                        'Kanagawa': 0.024,
                        'Saitama': 0.024,
                        'Gunma': 0.006,
                        'Tochigi': 0.006,
                        'Ibaraki': 0.024
                    },
                    '4week_SoE': {
                        'Tokyo': 0.06,
                        'Kanagawa': 0.024,
                        'Saitama': 0.024,
                        'Gunma': 0.006,
                        'Tochigi': 0.006,
                        'Ibaraki': 0.024
                    },
                    '5__week_SoE': {
                        'Tokyo': 0.07,
                        'Kanagawa': 0.028,
                        'Saitama': 0.028,
                        'Gunma': 0.007,
                        'Tochigi': 0.007,
                        'Ibaraki': 0.028
                    },
                    'after_SoE': {
                        'Tokyo': 0.09,
                        'Kanagawa': 0.036,
                        'Saitama': 0.036,
                        'Gunma': 0.009,
                        'Tochigi': 0.009,
                        'Ibaraki': 0.028
                    },
                },
            },
            {
                'name':
                'Kanagawa',
                'p_remove':
                0.1,
                'peaple': [
                    self.person_group(2, '00to19_1', 1, [1499988, 11, 1, 0.0]),
                    self.person_group(3, '20to44_1', 1,
                                      [2799881, 108, 11, 0.0]),
                    self.person_group(4, '45to64_1', 1, [2499911, 81, 8, 0.0]),
                    self.person_group(5, '65to99_1', 1, [2299923, 70, 7, 0.0]),
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
                        'Tokyo': 0.1,
                        'Chiba': 0.04,
                        'Saitama': 0.04,
                        'Gunma': 0.01,
                        'Tochigi': 0.01,
                        'Ibaraki': 0.01
                    },
                    '1week_SoE': {
                        'Tokyo': 0.09,
                        'Chiba': 0.036,
                        'Saitama': 0.036,
                        'Gunma': 0.009,
                        'Tochigi': 0.009,
                        'Ibaraki': 0.009
                    },
                    '2week_SoE': {
                        'Tokyo': 0.06,
                        'Chiba': 0.024,
                        'Saitama': 0.024,
                        'Gunma': 0.006,
                        'Tochigi': 0.006,
                        'Ibaraki': 0.006
                    },
                    '3week_SoE': {
                        'Tokyo': 0.06,
                        'Chiba': 0.024,
                        'Saitama': 0.024,
                        'Gunma': 0.006,
                        'Tochigi': 0.006,
                        'Ibaraki': 0.006
                    },
                    '4week_SoE': {
                        'Tokyo': 0.06,
                        'Chiba': 0.024,
                        'Saitama': 0.024,
                        'Gunma': 0.006,
                        'Tochigi': 0.006,
                        'Ibaraki': 0.006
                    },
                    '5__week_SoE': {
                        'Tokyo': 0.07,
                        'Chiba': 0.028,
                        'Saitama': 0.028,
                        'Gunma': 0.007,
                        'Tochigi': 0.007,
                        'Ibaraki': 0.007
                    },
                    'after_SoE': {
                        'Tokyo': 0.09,
                        'Chiba': 0.036,
                        'Saitama': 0.036,
                        'Gunma': 0.009,
                        'Tochigi': 0.009,
                        'Ibaraki': 0.009
                    },
                },
            },
            {
                'name':
                'Saitama',
                'p_remove':
                0.1,
                'peaple': [
                    self.person_group(2, '00to19_1', 1, [1199998, 2, 0, 0.0]),
                    self.person_group(3, '20to44_1', 1, [2199982, 16, 2, 0.0]),
                    self.person_group(4, '45to64_1', 1, [1999987, 12, 1, 0.0]),
                    self.person_group(5, '65to99_1', 1, [1899989, 10, 1, 0.0]),
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
                        'Tokyo': 0.1,
                        'Chiba': 0.02,
                        'Kanagawa': 0.02,
                        'Gunma': 0.02,
                        'Tochigi': 0.02,
                        'Ibaraki': 0.02
                    },
                    '1week_SoE': {
                        'Tokyo': 0.09,
                        'Chiba': 0.018,
                        'Kanagawa': 0.018,
                        'Gunma': 0.018,
                        'Tochigi': 0.018,
                        'Ibaraki': 0.018
                    },
                    '2week_SoE': {
                        'Tokyo': 0.06,
                        'Chiba': 0.012,
                        'Kanagawa': 0.012,
                        'Gunma': 0.012,
                        'Tochigi': 0.012,
                        'Ibaraki': 0.012
                    },
                    '3week_SoE': {
                        'Tokyo': 0.06,
                        'Chiba': 0.012,
                        'Kanagawa': 0.012,
                        'Gunma': 0.012,
                        'Tochigi': 0.012,
                        'Ibaraki': 0.012
                    },
                    '4week_SoE': {
                        'Tokyo': 0.06,
                        'Chiba': 0.012,
                        'Kanagawa': 0.012,
                        'Gunma': 0.012,
                        'Tochigi': 0.012,
                        'Ibaraki': 0.012
                    },
                    '5__week_SoE': {
                        'Tokyo': 0.07,
                        'Chiba': 0.014,
                        'Kanagawa': 0.014,
                        'Gunma': 0.014,
                        'Tochigi': 0.014,
                        'Ibaraki': 0.014
                    },
                    'after_SoE': {
                        'Tokyo': 0.09,
                        'Chiba': 0.018,
                        'Kanagawa': 0.018,
                        'Gunma': 0.018,
                        'Tochigi': 0.018,
                        'Ibaraki': 0.018
                    },
                },
            },
            {
                'name':
                'Gunma',
                'p_remove':
                0.1,
                'peaple': [
                    self.person_group(2, '00to19_1', 1, [300000, 0, 0, 0.0]),
                    self.person_group(3, '20to44_1', 1, [500000, 0, 0, 0.0]),
                    self.person_group(4, '45to64_1', 1, [500000, 0, 0, 0.0]),
                    self.person_group(5, '65to99_1', 1, [600000, 0, 0, 0.0]),
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
                        'Tokyo': 0.01,
                        'Chiba': 0.01,
                        'Kanagawa': 0.01,
                        'Saitama': 0.04,
                        'Tochigi': 0.04,
                        'Ibaraki': 0.04
                    },
                    '1week_SoE': {
                        'Tokyo': 0.009,
                        'Chiba': 0.009,
                        'Kanagawa': 0.009,
                        'Saitama': 0.036,
                        'Tochigi': 0.036,
                        'Ibaraki': 0.036
                    },
                    '2week_SoE': {
                        'Tokyo': 0.006,
                        'Chiba': 0.006,
                        'Kanagawa': 0.006,
                        'Saitama': 0.024,
                        'Tochigi': 0.024,
                        'Ibaraki': 0.024
                    },
                    '3week_SoE': {
                        'Tokyo': 0.006,
                        'Chiba': 0.006,
                        'Kanagawa': 0.006,
                        'Saitama': 0.024,
                        'Tochigi': 0.024,
                        'Ibaraki': 0.024
                    },
                    '4week_SoE': {
                        'Tokyo': 0.006,
                        'Chiba': 0.006,
                        'Kanagawa': 0.006,
                        'Saitama': 0.024,
                        'Tochigi': 0.024,
                        'Ibaraki': 0.024
                    },
                    '5__week_SoE': {
                        'Tokyo': 0.007,
                        'Chiba': 0.007,
                        'Kanagawa': 0.007,
                        'Saitama': 0.028,
                        'Tochigi': 0.028,
                        'Ibaraki': 0.028
                    },
                    'after_SoE': {
                        'Tokyo': 0.009,
                        'Chiba': 0.009,
                        'Kanagawa': 0.009,
                        'Saitama': 0.036,
                        'Tochigi': 0.036,
                        'Ibaraki': 0.036
                    },
                },
            },
            {
                'name':
                'Tochigi',
                'p_remove':
                0.1,
                'peaple': [
                    self.person_group(2, '00to19_1', 1, [300000, 0, 0, 0.0]),
                    self.person_group(3, '20to44_1', 1, [499996, 4, 1, 0.0]),
                    self.person_group(4, '45to64_1', 1, [499997, 3, 0, 0.0]),
                    self.person_group(5, '65to99_1', 1, [599997, 3, 0, 0.0]),
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
                        'Tokyo': 0.01,
                        'Chiba': 0.01,
                        'Kanagawa': 0.01,
                        'Saitama': 0.04,
                        'Gunma': 0.04,
                        'Ibaraki': 0.04
                    },
                    '1week_SoE': {
                        'Tokyo': 0.009,
                        'Chiba': 0.009,
                        'Kanagawa': 0.009,
                        'Saitama': 0.036,
                        'Gunma': 0.036,
                        'Ibaraki': 0.036
                    },
                    '2week_SoE': {
                        'Tokyo': 0.006,
                        'Chiba': 0.006,
                        'Kanagawa': 0.006,
                        'Saitama': 0.024,
                        'Gunma': 0.024,
                        'Ibaraki': 0.024
                    },
                    '3week_SoE': {
                        'Tokyo': 0.006,
                        'Chiba': 0.006,
                        'Kanagawa': 0.006,
                        'Saitama': 0.024,
                        'Gunma': 0.024,
                        'Ibaraki': 0.024
                    },
                    '4week_SoE': {
                        'Tokyo': 0.006,
                        'Chiba': 0.006,
                        'Kanagawa': 0.006,
                        'Saitama': 0.024,
                        'Gunma': 0.024,
                        'Ibaraki': 0.024
                    },
                    '5__week_SoE': {
                        'Tokyo': 0.007,
                        'Chiba': 0.007,
                        'Kanagawa': 0.007,
                        'Saitama': 0.028,
                        'Gunma': 0.028,
                        'Ibaraki': 0.028
                    },
                    'after_SoE': {
                        'Tokyo': 0.009,
                        'Chiba': 0.009,
                        'Kanagawa': 0.009,
                        'Saitama': 0.036,
                        'Gunma': 0.036,
                        'Ibaraki': 0.036
                    },
                },
            },
            {
                'name':
                'Ibaraki',
                'p_remove':
                0.1,
                'peaple': [
                    self.person_group(2, '00to19_1', 1, [500000, 0, 0, 0.0]),
                    self.person_group(3, '20to44_1', 1, [800000, 0, 0, 0.0]),
                    self.person_group(4, '45to64_1', 1, [800000, 0, 0, 0.0]),
                    self.person_group(5, '65to99_1', 1, [800000, 0, 0, 0.0]),
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
                        'Tokyo': 0.01,
                        'Chiba': 0.04,
                        'Kanagawa': 0.01,
                        'Saitama': 0.04,
                        'Gunma': 0.04,
                        'Tochigi': 0.04
                    },
                    '1week_SoE': {
                        'Tokyo': 0.009,
                        'Chiba': 0.036,
                        'Kanagawa': 0.009,
                        'Saitama': 0.036,
                        'Gunma': 0.036,
                        'Tochigi': 0.036
                    },
                    '2week_SoE': {
                        'Tokyo': 0.006,
                        'Chiba': 0.024,
                        'Kanagawa': 0.006,
                        'Saitama': 0.024,
                        'Gunma': 0.024,
                        'Tochigi': 0.024
                    },
                    '3week_SoE': {
                        'Tokyo': 0.006,
                        'Chiba': 0.024,
                        'Kanagawa': 0.006,
                        'Saitama': 0.024,
                        'Gunma': 0.024,
                        'Tochigi': 0.024
                    },
                    '4week_SoE': {
                        'Tokyo': 0.006,
                        'Chiba': 0.024,
                        'Kanagawa': 0.006,
                        'Saitama': 0.024,
                        'Gunma': 0.024,
                        'Tochigi': 0.024
                    },
                    '5__week_SoE': {
                        'Tokyo': 0.007,
                        'Chiba': 0.028,
                        'Kanagawa': 0.007,
                        'Saitama': 0.028,
                        'Gunma': 0.028,
                        'Tochigi': 0.028
                    },
                    'after_SoE': {
                        'Tokyo': 0.009,
                        'Chiba': 0.036,
                        'Kanagawa': 0.009,
                        'Saitama': 0.036,
                        'Gunma': 0.036,
                        'Tochigi': 0.036
                    },
                },
            },
        ]

        return self._create_citys(condition, smm)


class Izer_TL_kanto(Default_Izer):
    def __init__(self, config):
        super(Izer_TL_kanto, self).__init__(config)

    def create_citys(self, smm):
        condition = [
            {
                'name':
                'Tokyo',
                'p_remove':
                0.1,
                'peaple': [
                    # 2
                    self.person_group(2, '00to19_1', 1, [2099984, 15, 1, 0.0]),
                    self.person_group(3, '20to44_1', 1,
                                      [1919837, 148, 15, 0.0]),
                    self.person_group(4, '45to64_1', 1,
                                      [1359878, 111, 11, 0.0]),
                    self.person_group(5, '65to99_1', 1,
                                      [2999894, 96, 10, 0.0]),
                    self.person_group(6, '20to44_2_tele1', 1,
                                      [1200000, 0, 0, 0.0]),
                    self.person_group(7, '20to44_2_tele2', 1,
                                      [1680000, 0, 0, 0.0]),
                    self.person_group(8, '45to64_2_tele1', 1,
                                      [850000, 0, 0, 0.0]),
                    self.person_group(9, '45to64_2_tele2', 1,
                                      [1190000, 0, 0, 0.0]),
                ],
                'areas': [
                    self.area_group(
                        0, 'ncroud', {
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
                        'Chiba': 0.016,
                        'Kanagawa': 0.016,
                        'Saitama': 0.016,
                        'Gunma': 0.016,
                        'Tochigi': 0.016,
                        'Ibaraki': 0.016,
                    },
                    '1week_SoE': {
                        'Chiba': 0.013,
                        'Kanagawa': 0.013,
                        'Saitama': 0.013,
                        'Gunma': 0.013,
                        'Tochigi': 0.013,
                        'Ibaraki': 0.013,
                    },
                    '2week_SoE': {
                        'Chiba': 0.01,
                        'Kanagawa': 0.01,
                        'Saitama': 0.01,
                        'Gunma': 0.01,
                        'Tochigi': 0.01,
                        'Ibaraki': 0.01,
                    },
                    '3week_SoE': {
                        'Chiba': 0.008,
                        'Kanagawa': 0.008,
                        'Saitama': 0.008,
                        'Gunma': 0.008,
                        'Tochigi': 0.008,
                        'Ibaraki': 0.008,
                    },
                    '4week_SoE': {
                        'Chiba': 0.008,
                        'Kanagawa': 0.008,
                        'Saitama': 0.008,
                        'Gunma': 0.008,
                        'Tochigi': 0.008,
                        'Ibaraki': 0.008,
                    },
                    '5__week_SoE': {
                        'Chiba': 0.008,
                        'Kanagawa': 0.008,
                        'Saitama': 0.008,
                        'Gunma': 0.008,
                        'Tochigi': 0.008,
                        'Ibaraki': 0.008,
                    },
                    'after_SoE': {
                        'Chiba': 0.016,
                        'Kanagawa': 0.016,
                        'Saitama': 0.016,
                        'Gunma': 0.016,
                        'Tochigi': 0.016,
                        'Ibaraki': 0.016,
                    },
                },
            },
            {
                'name':
                'Chiba',
                'p_remove':
                0.1,
                'peaple': [
                    self.person_group(2, '00to19_1', 1, [1099993, 6, 1, 0.0]),
                    self.person_group(3, '20to44_1', 1, [759930, 64, 6, 0.0]),
                    self.person_group(4, '45to64_1', 1, [679947, 48, 5, 0.0]),
                    self.person_group(5, '65to99_1', 1, [1699954, 42, 4, 0.0]),
                    self.person_group(6, '20to44_2_tele1', 1,
                                      [475000, 0, 0, 0.0]),
                    self.person_group(7, '20to44_2_tele2', 1,
                                      [665000, 0, 0, 0.0]),
                    self.person_group(8, '45to64_2_tele1', 1,
                                      [425000, 0, 0, 0.0]),
                    self.person_group(9, '45to64_2_tele2', 1,
                                      [595000, 0, 0, 0.0]),
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
                        'Tokyo': 0.1,
                        'Kanagawa': 0.04,
                        'Saitama': 0.04,
                        'Gunma': 0.01,
                        'Tochigi': 0.01,
                        'Ibaraki': 0.04
                    },
                    '1week_SoE': {
                        'Tokyo': 0.09,
                        'Kanagawa': 0.036,
                        'Saitama': 0.036,
                        'Gunma': 0.009,
                        'Tochigi': 0.009,
                        'Ibaraki': 0.036
                    },
                    '2week_SoE': {
                        'Tokyo': 0.06,
                        'Kanagawa': 0.024,
                        'Saitama': 0.024,
                        'Gunma': 0.006,
                        'Tochigi': 0.006,
                        'Ibaraki': 0.024
                    },
                    '3week_SoE': {
                        'Tokyo': 0.06,
                        'Kanagawa': 0.024,
                        'Saitama': 0.024,
                        'Gunma': 0.006,
                        'Tochigi': 0.006,
                        'Ibaraki': 0.024
                    },
                    '4week_SoE': {
                        'Tokyo': 0.06,
                        'Kanagawa': 0.024,
                        'Saitama': 0.024,
                        'Gunma': 0.006,
                        'Tochigi': 0.006,
                        'Ibaraki': 0.024
                    },
                    '5__week_SoE': {
                        'Tokyo': 0.07,
                        'Kanagawa': 0.028,
                        'Saitama': 0.028,
                        'Gunma': 0.007,
                        'Tochigi': 0.007,
                        'Ibaraki': 0.028
                    },
                    'after_SoE': {
                        'Tokyo': 0.09,
                        'Kanagawa': 0.036,
                        'Saitama': 0.036,
                        'Gunma': 0.009,
                        'Tochigi': 0.009,
                        'Ibaraki': 0.028
                    },
                },
            },
            {
                'name':
                'Kanagawa',
                'p_remove':
                0.1,
                'peaple': [
                    self.person_group(2, '00to19_1', 1, [1499989, 10, 1, 0.0]),
                    self.person_group(3, '20to44_1', 1,
                                      [1119890, 100, 10, 0.0]),
                    self.person_group(4, '45to64_1', 1, [999918, 75, 8, 0.0]),
                    self.person_group(5, '65to99_1', 1, [2299929, 65, 7, 0.0]),
                    self.person_group(6, '20to44_2_tele1', 1,
                                      [700000, 0, 0, 0.0]),
                    self.person_group(7, '20to44_2_tele2', 1,
                                      [980000, 0, 0, 0.0]),
                    self.person_group(8, '45to64_2_tele1', 1,
                                      [625000, 0, 0, 0.0]),
                    self.person_group(9, '45to64_2_tele2', 1,
                                      [875000, 0, 0, 0.0]),
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
                        'Tokyo': 0.1,
                        'Chiba': 0.04,
                        'Saitama': 0.04,
                        'Gunma': 0.01,
                        'Tochigi': 0.01,
                        'Ibaraki': 0.01
                    },
                    '1week_SoE': {
                        'Tokyo': 0.09,
                        'Chiba': 0.036,
                        'Saitama': 0.036,
                        'Gunma': 0.009,
                        'Tochigi': 0.009,
                        'Ibaraki': 0.009
                    },
                    '2week_SoE': {
                        'Tokyo': 0.06,
                        'Chiba': 0.024,
                        'Saitama': 0.024,
                        'Gunma': 0.006,
                        'Tochigi': 0.006,
                        'Ibaraki': 0.006
                    },
                    '3week_SoE': {
                        'Tokyo': 0.06,
                        'Chiba': 0.024,
                        'Saitama': 0.024,
                        'Gunma': 0.006,
                        'Tochigi': 0.006,
                        'Ibaraki': 0.006
                    },
                    '4week_SoE': {
                        'Tokyo': 0.06,
                        'Chiba': 0.024,
                        'Saitama': 0.024,
                        'Gunma': 0.006,
                        'Tochigi': 0.006,
                        'Ibaraki': 0.006
                    },
                    '5__week_SoE': {
                        'Tokyo': 0.07,
                        'Chiba': 0.028,
                        'Saitama': 0.028,
                        'Gunma': 0.007,
                        'Tochigi': 0.007,
                        'Ibaraki': 0.007
                    },
                    'after_SoE': {
                        'Tokyo': 0.09,
                        'Chiba': 0.036,
                        'Saitama': 0.036,
                        'Gunma': 0.009,
                        'Tochigi': 0.009,
                        'Ibaraki': 0.009
                    },
                },
            },
            {
                'name':
                'Saitama',
                'p_remove':
                0.1,
                'peaple': [
                    self.person_group(2, '00to19_1', 1, [1199998, 2, 0, 0.0]),
                    self.person_group(3, '20to44_1', 1, [879982, 16, 2, 0.0]),
                    self.person_group(4, '45to64_1', 1, [799987, 12, 1, 0.0]),
                    self.person_group(5, '65to99_1', 1, [1899989, 10, 1, 0.0]),
                    self.person_group(6, '20to44_2_tele1', 1,
                                      [550000, 0, 0, 0.0]),
                    self.person_group(7, '20to44_2_tele2', 1,
                                      [770000, 0, 0, 0.0]),
                    self.person_group(8, '45to64_2_tele1', 1,
                                      [500000, 0, 0, 0.0]),
                    self.person_group(9, '45to64_2_tele2', 1,
                                      [700000, 0, 0, 0.0]),
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
                        'Tokyo': 0.1,
                        'Chiba': 0.02,
                        'Kanagawa': 0.02,
                        'Gunma': 0.02,
                        'Tochigi': 0.02,
                        'Ibaraki': 0.02
                    },
                    '1week_SoE': {
                        'Tokyo': 0.09,
                        'Chiba': 0.018,
                        'Kanagawa': 0.018,
                        'Gunma': 0.018,
                        'Tochigi': 0.018,
                        'Ibaraki': 0.018
                    },
                    '2week_SoE': {
                        'Tokyo': 0.06,
                        'Chiba': 0.012,
                        'Kanagawa': 0.012,
                        'Gunma': 0.012,
                        'Tochigi': 0.012,
                        'Ibaraki': 0.012
                    },
                    '3week_SoE': {
                        'Tokyo': 0.06,
                        'Chiba': 0.012,
                        'Kanagawa': 0.012,
                        'Gunma': 0.012,
                        'Tochigi': 0.012,
                        'Ibaraki': 0.012
                    },
                    '4week_SoE': {
                        'Tokyo': 0.06,
                        'Chiba': 0.012,
                        'Kanagawa': 0.012,
                        'Gunma': 0.012,
                        'Tochigi': 0.012,
                        'Ibaraki': 0.012
                    },
                    '5__week_SoE': {
                        'Tokyo': 0.07,
                        'Chiba': 0.014,
                        'Kanagawa': 0.014,
                        'Gunma': 0.014,
                        'Tochigi': 0.014,
                        'Ibaraki': 0.014
                    },
                    'after_SoE': {
                        'Tokyo': 0.09,
                        'Chiba': 0.018,
                        'Kanagawa': 0.018,
                        'Gunma': 0.018,
                        'Tochigi': 0.018,
                        'Ibaraki': 0.018
                    },
                },
            },
            {
                'name':
                'Gunma',
                'p_remove':
                0.1,
                'peaple': [
                    self.person_group(2, '00to19_1', 1, [300000, 0, 0, 0.0]),
                    self.person_group(3, '20to44_1', 1, [200000, 0, 0, 0.0]),
                    self.person_group(4, '45to64_1', 1, [200000, 0, 0, 0.0]),
                    self.person_group(5, '65to99_1', 1, [600000, 0, 0, 0.0]),
                    self.person_group(6, '20to44_2_tele1', 1,
                                      [125000, 0, 0, 0.0]),
                    self.person_group(7, '20to44_2_tele2', 1,
                                      [175000, 0, 0, 0.0]),
                    self.person_group(8, '45to64_2_tele1', 1,
                                      [125000, 0, 0, 0.0]),
                    self.person_group(9, '45to64_2_tele2', 1,
                                      [175000, 0, 0, 0.0]),
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
                        'Tokyo': 0.01,
                        'Chiba': 0.01,
                        'Kanagawa': 0.01,
                        'Saitama': 0.04,
                        'Tochigi': 0.04,
                        'Ibaraki': 0.04
                    },
                    '1week_SoE': {
                        'Tokyo': 0.009,
                        'Chiba': 0.009,
                        'Kanagawa': 0.009,
                        'Saitama': 0.036,
                        'Tochigi': 0.036,
                        'Ibaraki': 0.036
                    },
                    '2week_SoE': {
                        'Tokyo': 0.006,
                        'Chiba': 0.006,
                        'Kanagawa': 0.006,
                        'Saitama': 0.024,
                        'Tochigi': 0.024,
                        'Ibaraki': 0.024
                    },
                    '3week_SoE': {
                        'Tokyo': 0.006,
                        'Chiba': 0.006,
                        'Kanagawa': 0.006,
                        'Saitama': 0.024,
                        'Tochigi': 0.024,
                        'Ibaraki': 0.024
                    },
                    '4week_SoE': {
                        'Tokyo': 0.006,
                        'Chiba': 0.006,
                        'Kanagawa': 0.006,
                        'Saitama': 0.024,
                        'Tochigi': 0.024,
                        'Ibaraki': 0.024
                    },
                    '5__week_SoE': {
                        'Tokyo': 0.007,
                        'Chiba': 0.007,
                        'Kanagawa': 0.007,
                        'Saitama': 0.028,
                        'Tochigi': 0.028,
                        'Ibaraki': 0.028
                    },
                    'after_SoE': {
                        'Tokyo': 0.009,
                        'Chiba': 0.009,
                        'Kanagawa': 0.009,
                        'Saitama': 0.036,
                        'Tochigi': 0.036,
                        'Ibaraki': 0.036
                    },
                },
            },
            {
                'name':
                'Tochigi',
                'p_remove':
                0.1,
                'peaple': [
                    self.person_group(2, '00to19_1', 1, [300000, 0, 0, 0.0]),
                    self.person_group(3, '20to44_1', 1, [199996, 4, 1, 0.0]),
                    self.person_group(4, '45to64_1', 1, [199997, 3, 0, 0.0]),
                    self.person_group(5, '65to99_1', 1, [599997, 3, 0, 0.0]),
                    self.person_group(6, '20to44_2_tele1', 1,
                                      [125000, 0, 0, 0.0]),
                    self.person_group(7, '20to44_2_tele2', 1,
                                      [175000, 0, 0, 0.0]),
                    self.person_group(8, '45to64_2_tele1', 1,
                                      [125000, 0, 0, 0.0]),
                    self.person_group(9, '45to64_2_tele2', 1,
                                      [175000, 0, 0, 0.0]),
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
                        'Tokyo': 0.01,
                        'Chiba': 0.01,
                        'Kanagawa': 0.01,
                        'Saitama': 0.04,
                        'Gunma': 0.04,
                        'Ibaraki': 0.04
                    },
                    '1week_SoE': {
                        'Tokyo': 0.009,
                        'Chiba': 0.009,
                        'Kanagawa': 0.009,
                        'Saitama': 0.036,
                        'Gunma': 0.036,
                        'Ibaraki': 0.036
                    },
                    '2week_SoE': {
                        'Tokyo': 0.006,
                        'Chiba': 0.006,
                        'Kanagawa': 0.006,
                        'Saitama': 0.024,
                        'Gunma': 0.024,
                        'Ibaraki': 0.024
                    },
                    '3week_SoE': {
                        'Tokyo': 0.006,
                        'Chiba': 0.006,
                        'Kanagawa': 0.006,
                        'Saitama': 0.024,
                        'Gunma': 0.024,
                        'Ibaraki': 0.024
                    },
                    '4week_SoE': {
                        'Tokyo': 0.006,
                        'Chiba': 0.006,
                        'Kanagawa': 0.006,
                        'Saitama': 0.024,
                        'Gunma': 0.024,
                        'Ibaraki': 0.024
                    },
                    '5__week_SoE': {
                        'Tokyo': 0.007,
                        'Chiba': 0.007,
                        'Kanagawa': 0.007,
                        'Saitama': 0.028,
                        'Gunma': 0.028,
                        'Ibaraki': 0.028
                    },
                    'after_SoE': {
                        'Tokyo': 0.009,
                        'Chiba': 0.009,
                        'Kanagawa': 0.009,
                        'Saitama': 0.036,
                        'Gunma': 0.036,
                        'Ibaraki': 0.036
                    },
                },
            },
            {
                'name':
                'Ibaraki',
                'p_remove':
                0.1,
                'peaple': [
                    self.person_group(2, '00to19_1', 1, [500000, 0, 0, 0.0]),
                    self.person_group(3, '20to44_1', 1, [320000, 0, 0, 0.0]),
                    self.person_group(4, '45to64_1', 1, [320000, 0, 0, 0.0]),
                    self.person_group(5, '65to99_1', 1, [800000, 0, 0, 0.0]),
                    self.person_group(6, '20to44_2_tele1', 1,
                                      [200000, 0, 0, 0.0]),
                    self.person_group(7, '20to44_2_tele2', 1,
                                      [280000, 0, 0, 0.0]),
                    self.person_group(8, '45to64_2_tele1', 1,
                                      [200000, 0, 0, 0.0]),
                    self.person_group(9, '45to64_2_tele2', 1,
                                      [280000, 0, 0, 0.0]),
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
                        'Tokyo': 0.01,
                        'Chiba': 0.04,
                        'Kanagawa': 0.01,
                        'Saitama': 0.04,
                        'Gunma': 0.04,
                        'Tochigi': 0.04
                    },
                    '1week_SoE': {
                        'Tokyo': 0.009,
                        'Chiba': 0.036,
                        'Kanagawa': 0.009,
                        'Saitama': 0.036,
                        'Gunma': 0.036,
                        'Tochigi': 0.036
                    },
                    '2week_SoE': {
                        'Tokyo': 0.006,
                        'Chiba': 0.024,
                        'Kanagawa': 0.006,
                        'Saitama': 0.024,
                        'Gunma': 0.024,
                        'Tochigi': 0.024
                    },
                    '3week_SoE': {
                        'Tokyo': 0.006,
                        'Chiba': 0.024,
                        'Kanagawa': 0.006,
                        'Saitama': 0.024,
                        'Gunma': 0.024,
                        'Tochigi': 0.024
                    },
                    '4week_SoE': {
                        'Tokyo': 0.006,
                        'Chiba': 0.024,
                        'Kanagawa': 0.006,
                        'Saitama': 0.024,
                        'Gunma': 0.024,
                        'Tochigi': 0.024
                    },
                    '5__week_SoE': {
                        'Tokyo': 0.007,
                        'Chiba': 0.028,
                        'Kanagawa': 0.007,
                        'Saitama': 0.028,
                        'Gunma': 0.028,
                        'Tochigi': 0.028
                    },
                    'after_SoE': {
                        'Tokyo': 0.009,
                        'Chiba': 0.036,
                        'Kanagawa': 0.009,
                        'Saitama': 0.036,
                        'Gunma': 0.036,
                        'Tochigi': 0.036
                    },
                },
            },
        ]

        return self._create_citys(condition, smm)


class Izer_TL_GO_kanto(Default_Izer):
    def __init__(self, config):
        super(Izer_TL_GO_kanto, self).__init__(config)

    def create_citys(self, smm):
        condition = [
            {
                'name':
                'Tokyo',
                'p_remove':
                0.1,
                'peaple': [
                    # 2
                    self.person_group(2, '00to19_1', 1, [1889987, 12, 1, 0.0]),
                    self.person_group(3, '20to44_1', 1,
                                      [1439867, 118, 15, 0.0]),
                    self.person_group(4, '45to64_1', 1,
                                      [1019900, 89, 11, 0.0]),
                    self.person_group(5, '65to99_1', 1,
                                      [2699913, 77, 10, 0.0]),
                    self.person_group(6, '20to44_2_tele1', 1,
                                      [1200000, 0, 0, 0.0]),
                    self.person_group(7, '20to44_2_tele2', 1,
                                      [1680000, 0, 0, 0.0]),
                    self.person_group(8, '45to64_2_tele1', 1,
                                      [850000, 0, 0, 0.0]),
                    self.person_group(9, '45to64_2_tele2', 1,
                                      [1190000, 0, 0, 0.0]),
                    self.person_group(10, '00to19_3_goout', 1,
                                      [210000, 0, 0, 0.0]),
                    self.person_group(11, '20to44_3_goout', 1,
                                      [480000, 0, 0, 0.0]),
                    self.person_group(12, '45to64_3_goout', 1,
                                      [340000, 0, 0, 0.0]),
                    self.person_group(13, '65to99_3_goout', 1,
                                      [300000, 0, 0, 0.0]),
                ],
                'areas': [
                    self.area_group(
                        0, 'ncroud', {
                            'before_SoE': [[0.0002] * 24, [0.0002] * 24],
                            'before_2_SoE': [[0.0002] * 24, [0.0002] * 24],
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
                            'before_2_SoE': [[0.002] * 24, [0.002] * 24],
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
                            'before_2_SoE': [[0.02] * 24, [0.02] * 24],
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
                        'Chiba': 0.016,
                        'Kanagawa': 0.016,
                        'Saitama': 0.016,
                        'Gunma': 0.016,
                        'Tochigi': 0.016,
                        'Ibaraki': 0.016,
                    },
                    'before_2_SoE': {
                        'Chiba': 0.018,
                        'Kanagawa': 0.018,
                        'Saitama': 0.018,
                        'Gunma': 0.018,
                        'Tochigi': 0.018,
                        'Ibaraki': 0.018,
                    },
                    '1week_SoE': {
                        'Chiba': 0.013,
                        'Kanagawa': 0.013,
                        'Saitama': 0.013,
                        'Gunma': 0.013,
                        'Tochigi': 0.013,
                        'Ibaraki': 0.013,
                    },
                    '2week_SoE': {
                        'Chiba': 0.01,
                        'Kanagawa': 0.01,
                        'Saitama': 0.01,
                        'Gunma': 0.01,
                        'Tochigi': 0.01,
                        'Ibaraki': 0.01,
                    },
                    '3week_SoE': {
                        'Chiba': 0.008,
                        'Kanagawa': 0.008,
                        'Saitama': 0.008,
                        'Gunma': 0.008,
                        'Tochigi': 0.008,
                        'Ibaraki': 0.008,
                    },
                    '4week_SoE': {
                        'Chiba': 0.008,
                        'Kanagawa': 0.008,
                        'Saitama': 0.008,
                        'Gunma': 0.008,
                        'Tochigi': 0.008,
                        'Ibaraki': 0.008,
                    },
                    '5__week_SoE': {
                        'Chiba': 0.008,
                        'Kanagawa': 0.008,
                        'Saitama': 0.008,
                        'Gunma': 0.008,
                        'Tochigi': 0.008,
                        'Ibaraki': 0.008,
                    },
                    'after_SoE': {
                        'Chiba': 0.016,
                        'Kanagawa': 0.016,
                        'Saitama': 0.016,
                        'Gunma': 0.016,
                        'Tochigi': 0.016,
                        'Ibaraki': 0.016,
                    },
                },
            },
            {
                'name':
                'Chiba',
                'p_remove':
                0.1,
                'peaple': [
                    self.person_group(2, '00to19_1', 1, [989994, 5, 1, 0.0]),
                    self.person_group(3, '20to44_1', 1, [569942, 51, 6, 0.0]),
                    self.person_group(4, '45to64_1', 1, [509957, 38, 5, 0.0]),
                    self.person_group(5, '65to99_1', 1, [1529963, 33, 4, 0.0]),
                    self.person_group(6, '20to44_2_tele1', 1,
                                      [475000, 0, 0, 0.0]),
                    self.person_group(7, '20to44_2_tele2', 1,
                                      [665000, 0, 0, 0.0]),
                    self.person_group(8, '45to64_2_tele1', 1,
                                      [425000, 0, 0, 0.0]),
                    self.person_group(9, '45to64_2_tele2', 1,
                                      [595000, 0, 0, 0.0]),
                    self.person_group(10, '00to19_3_goout', 1,
                                      [110000, 0, 0, 0.0]),
                    self.person_group(11, '20to44_3_goout', 1,
                                      [190000, 0, 0, 0.0]),
                    self.person_group(12, '45to64_3_goout', 1,
                                      [170000, 0, 0, 0.0]),
                    self.person_group(13, '65to99_3_goout', 1,
                                      [170000, 0, 0, 0.0]),
                ],
                'areas': [
                    self.area_group(
                        0, 'ncroud', {
                            'before_SoE': [[0.0002] * 24, [0.0002] * 24],
                            'before_2_SoE': [[0.0002] * 24, [0.0002] * 24],
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
                            'before_2_SoE': [[0.002] * 24, [0.002] * 24],
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
                            'before_2_SoE': [[0.02] * 24, [0.02] * 24],
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
                        'Tokyo': 0.09,
                        'Kanagawa': 0.036,
                        'Saitama': 0.036,
                        'Gunma': 0.009,
                        'Tochigi': 0.009,
                        'Ibaraki': 0.028
                    },
                    'before_2_SoE': {
                        'Tokyo': 0.1,
                        'Kanagawa': 0.04,
                        'Saitama': 0.04,
                        'Gunma': 0.01,
                        'Tochigi': 0.01,
                        'Ibaraki': 0.04
                    },
                    '1week_SoE': {
                        'Tokyo': 0.09,
                        'Kanagawa': 0.036,
                        'Saitama': 0.036,
                        'Gunma': 0.009,
                        'Tochigi': 0.009,
                        'Ibaraki': 0.036
                    },
                    '2week_SoE': {
                        'Tokyo': 0.06,
                        'Kanagawa': 0.024,
                        'Saitama': 0.024,
                        'Gunma': 0.006,
                        'Tochigi': 0.006,
                        'Ibaraki': 0.024
                    },
                    '3week_SoE': {
                        'Tokyo': 0.06,
                        'Kanagawa': 0.024,
                        'Saitama': 0.024,
                        'Gunma': 0.006,
                        'Tochigi': 0.006,
                        'Ibaraki': 0.024
                    },
                    '4week_SoE': {
                        'Tokyo': 0.06,
                        'Kanagawa': 0.024,
                        'Saitama': 0.024,
                        'Gunma': 0.006,
                        'Tochigi': 0.006,
                        'Ibaraki': 0.024
                    },
                    '5__week_SoE': {
                        'Tokyo': 0.07,
                        'Kanagawa': 0.028,
                        'Saitama': 0.028,
                        'Gunma': 0.007,
                        'Tochigi': 0.007,
                        'Ibaraki': 0.028
                    },
                    'after_SoE': {
                        'Tokyo': 0.09,
                        'Kanagawa': 0.036,
                        'Saitama': 0.036,
                        'Gunma': 0.009,
                        'Tochigi': 0.009,
                        'Ibaraki': 0.028
                    },
                },
            },
            {
                'name':
                'Kanagawa',
                'p_remove':
                0.1,
                'peaple': [
                    self.person_group(2, '00to19_1', 1, [1349991, 8, 1, 0.0]),
                    self.person_group(3, '20to44_1', 1, [839910, 80, 11, 0.0]),
                    self.person_group(4, '45to64_1', 1, [749933, 60, 8, 0.0]),
                    self.person_group(5, '65to99_1', 1, [2069942, 52, 7, 0.0]),
                    self.person_group(6, '20to44_2_tele1', 1,
                                      [700000, 0, 0, 0.0]),
                    self.person_group(7, '20to44_2_tele2', 1,
                                      [980000, 0, 0, 0.0]),
                    self.person_group(8, '45to64_2_tele1', 1,
                                      [625000, 0, 0, 0.0]),
                    self.person_group(9, '45to64_2_tele2', 1,
                                      [875000, 0, 0, 0.0]),
                    self.person_group(10, '00to19_3_goout', 1,
                                      [150000, 0, 0, 0.0]),
                    self.person_group(11, '20to44_3_goout', 1,
                                      [280000, 0, 0, 0.0]),
                    self.person_group(12, '45to64_3_goout', 1,
                                      [250000, 0, 0, 0.0]),
                    self.person_group(13, '65to99_3_goout', 1,
                                      [230000, 0, 0, 0.0]),
                ],
                'areas': [
                    self.area_group(
                        0, 'ncroud', {
                            'before_SoE': [[0.0002] * 24, [0.0002] * 24],
                            'before_2_SoE': [[0.0002] * 24, [0.0002] * 24],
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
                            'before_2_SoE': [[0.002] * 24, [0.002] * 24],
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
                            'before_2_SoE': [[0.02] * 24, [0.02] * 24],
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
                        'Tokyo': 0.09,
                        'Chiba': 0.036,
                        'Saitama': 0.036,
                        'Gunma': 0.009,
                        'Tochigi': 0.009,
                        'Ibaraki': 0.009
                    },
                    'before_2_SoE': {
                        'Tokyo': 0.1,
                        'Chiba': 0.04,
                        'Saitama': 0.04,
                        'Gunma': 0.01,
                        'Tochigi': 0.01,
                        'Ibaraki': 0.01
                    },
                    '1week_SoE': {
                        'Tokyo': 0.09,
                        'Chiba': 0.036,
                        'Saitama': 0.036,
                        'Gunma': 0.009,
                        'Tochigi': 0.009,
                        'Ibaraki': 0.009
                    },
                    '2week_SoE': {
                        'Tokyo': 0.06,
                        'Chiba': 0.024,
                        'Saitama': 0.024,
                        'Gunma': 0.006,
                        'Tochigi': 0.006,
                        'Ibaraki': 0.006
                    },
                    '3week_SoE': {
                        'Tokyo': 0.06,
                        'Chiba': 0.024,
                        'Saitama': 0.024,
                        'Gunma': 0.006,
                        'Tochigi': 0.006,
                        'Ibaraki': 0.006
                    },
                    '4week_SoE': {
                        'Tokyo': 0.06,
                        'Chiba': 0.024,
                        'Saitama': 0.024,
                        'Gunma': 0.006,
                        'Tochigi': 0.006,
                        'Ibaraki': 0.006
                    },
                    '5__week_SoE': {
                        'Tokyo': 0.07,
                        'Chiba': 0.028,
                        'Saitama': 0.028,
                        'Gunma': 0.007,
                        'Tochigi': 0.007,
                        'Ibaraki': 0.007
                    },
                    'after_SoE': {
                        'Tokyo': 0.09,
                        'Chiba': 0.036,
                        'Saitama': 0.036,
                        'Gunma': 0.009,
                        'Tochigi': 0.009,
                        'Ibaraki': 0.009
                    },
                },
            },
            {
                'name':
                'Saitama',
                'p_remove':
                0.1,
                'peaple': [
                    self.person_group(2, '00to19_1', 1, [1079999, 1, 0, 0.0]),
                    self.person_group(3, '20to44_1', 1, [659986, 13, 2, 0.0]),
                    self.person_group(4, '45to64_1', 1, [599989, 10, 1, 0.0]),
                    self.person_group(5, '65to99_1', 1, [1709991, 8, 1, 0.0]),
                    self.person_group(6, '20to44_2_tele1', 1,
                                      [550000, 0, 0, 0.0]),
                    self.person_group(7, '20to44_2_tele2', 1,
                                      [770000, 0, 0, 0.0]),
                    self.person_group(8, '45to64_2_tele1', 1,
                                      [500000, 0, 0, 0.0]),
                    self.person_group(9, '45to64_2_tele2', 1,
                                      [700000, 0, 0, 0.0]),
                    self.person_group(10, '00to19_3_goout', 1,
                                      [120000, 0, 0, 0.0]),
                    self.person_group(11, '20to44_3_goout', 1,
                                      [220000, 0, 0, 0.0]),
                    self.person_group(12, '45to64_3_goout', 1,
                                      [200000, 0, 0, 0.0]),
                    self.person_group(13, '65to99_3_goout', 1,
                                      [190000, 0, 0, 0.0]),
                ],
                'areas': [
                    self.area_group(
                        0, 'ncroud', {
                            'before_SoE': [[0.0002] * 24, [0.0002] * 24],
                            'before_2_SoE': [[0.0002] * 24, [0.0002] * 24],
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
                            'before_2_SoE': [[0.002] * 24, [0.002] * 24],
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
                            'before_2_SoE': [[0.02] * 24, [0.02] * 24],
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
                        'Tokyo': 0.09,
                        'Chiba': 0.018,
                        'Kanagawa': 0.018,
                        'Gunma': 0.018,
                        'Tochigi': 0.018,
                        'Ibaraki': 0.018
                    },
                    'before_2_SoE': {
                        'Tokyo': 0.1,
                        'Chiba': 0.02,
                        'Kanagawa': 0.02,
                        'Gunma': 0.02,
                        'Tochigi': 0.02,
                        'Ibaraki': 0.02
                    },
                    '1week_SoE': {
                        'Tokyo': 0.09,
                        'Chiba': 0.018,
                        'Kanagawa': 0.018,
                        'Gunma': 0.018,
                        'Tochigi': 0.018,
                        'Ibaraki': 0.018
                    },
                    '2week_SoE': {
                        'Tokyo': 0.06,
                        'Chiba': 0.012,
                        'Kanagawa': 0.012,
                        'Gunma': 0.012,
                        'Tochigi': 0.012,
                        'Ibaraki': 0.012
                    },
                    '3week_SoE': {
                        'Tokyo': 0.06,
                        'Chiba': 0.012,
                        'Kanagawa': 0.012,
                        'Gunma': 0.012,
                        'Tochigi': 0.012,
                        'Ibaraki': 0.012
                    },
                    '4week_SoE': {
                        'Tokyo': 0.06,
                        'Chiba': 0.012,
                        'Kanagawa': 0.012,
                        'Gunma': 0.012,
                        'Tochigi': 0.012,
                        'Ibaraki': 0.012
                    },
                    '5__week_SoE': {
                        'Tokyo': 0.07,
                        'Chiba': 0.014,
                        'Kanagawa': 0.014,
                        'Gunma': 0.014,
                        'Tochigi': 0.014,
                        'Ibaraki': 0.014
                    },
                    'after_SoE': {
                        'Tokyo': 0.09,
                        'Chiba': 0.018,
                        'Kanagawa': 0.018,
                        'Gunma': 0.018,
                        'Tochigi': 0.018,
                        'Ibaraki': 0.018
                    },
                },
            },
            {
                'name':
                'Gunma',
                'p_remove':
                0.1,
                'peaple': [
                    self.person_group(2, '00to19_1', 1, [270000, 0, 0, 0.0]),
                    self.person_group(3, '20to44_1', 1, [150000, 0, 0, 0.0]),
                    self.person_group(4, '45to64_1', 1, [150000, 0, 0, 0.0]),
                    self.person_group(5, '65to99_1', 1, [540000, 0, 0, 0.0]),
                    self.person_group(6, '20to44_2_tele1', 1,
                                      [125000, 0, 0, 0.0]),
                    self.person_group(7, '20to44_2_tele2', 1,
                                      [175000, 0, 0, 0.0]),
                    self.person_group(8, '45to64_2_tele1', 1,
                                      [125000, 0, 0, 0.0]),
                    self.person_group(9, '45to64_2_tele2', 1,
                                      [175000, 0, 0, 0.0]),
                    self.person_group(10, '00to19_3_goout', 1,
                                      [30000, 0, 0, 0.0]),
                    self.person_group(11, '20to44_3_goout', 1,
                                      [50000, 0, 0, 0.0]),
                    self.person_group(12, '45to64_3_goout', 1,
                                      [50000, 0, 0, 0.0]),
                    self.person_group(13, '65to99_3_goout', 1,
                                      [60000, 0, 0, 0.0]),
                ],
                'areas': [
                    self.area_group(
                        0, 'ncroud', {
                            'before_SoE': [[0.0002] * 24, [0.0002] * 24],
                            'before_2_SoE': [[0.0002] * 24, [0.0002] * 24],
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
                            'before_2_SoE': [[0.002] * 24, [0.002] * 24],
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
                            'before_2_SoE': [[0.02] * 24, [0.02] * 24],
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
                        'Tokyo': 0.009,
                        'Chiba': 0.009,
                        'Kanagawa': 0.009,
                        'Saitama': 0.036,
                        'Tochigi': 0.036,
                        'Ibaraki': 0.036
                    },
                    'before_2_SoE': {
                        'Tokyo': 0.01,
                        'Chiba': 0.01,
                        'Kanagawa': 0.01,
                        'Saitama': 0.04,
                        'Tochigi': 0.04,
                        'Ibaraki': 0.04
                    },
                    '1week_SoE': {
                        'Tokyo': 0.009,
                        'Chiba': 0.009,
                        'Kanagawa': 0.009,
                        'Saitama': 0.036,
                        'Tochigi': 0.036,
                        'Ibaraki': 0.036
                    },
                    '2week_SoE': {
                        'Tokyo': 0.006,
                        'Chiba': 0.006,
                        'Kanagawa': 0.006,
                        'Saitama': 0.024,
                        'Tochigi': 0.024,
                        'Ibaraki': 0.024
                    },
                    '3week_SoE': {
                        'Tokyo': 0.006,
                        'Chiba': 0.006,
                        'Kanagawa': 0.006,
                        'Saitama': 0.024,
                        'Tochigi': 0.024,
                        'Ibaraki': 0.024
                    },
                    '4week_SoE': {
                        'Tokyo': 0.006,
                        'Chiba': 0.006,
                        'Kanagawa': 0.006,
                        'Saitama': 0.024,
                        'Tochigi': 0.024,
                        'Ibaraki': 0.024
                    },
                    '5__week_SoE': {
                        'Tokyo': 0.007,
                        'Chiba': 0.007,
                        'Kanagawa': 0.007,
                        'Saitama': 0.028,
                        'Tochigi': 0.028,
                        'Ibaraki': 0.028
                    },
                    'after_SoE': {
                        'Tokyo': 0.009,
                        'Chiba': 0.009,
                        'Kanagawa': 0.009,
                        'Saitama': 0.036,
                        'Tochigi': 0.036,
                        'Ibaraki': 0.036
                    },
                },
            },
            {
                'name':
                'Tochigi',
                'p_remove':
                0.1,
                'peaple': [
                    self.person_group(2, '00to19_1', 1, [270000, 0, 0, 0.0]),
                    self.person_group(3, '20to44_1', 1, [149996, 3, 1, 0.0]),
                    self.person_group(4, '45to64_1', 1, [149997, 2, 0, 0.0]),
                    self.person_group(5, '65to99_1', 1, [539998, 2, 0, 0.0]),
                    self.person_group(6, '20to44_2_tele1', 1,
                                      [125000, 0, 0, 0.0]),
                    self.person_group(7, '20to44_2_tele2', 1,
                                      [175000, 0, 0, 0.0]),
                    self.person_group(8, '45to64_2_tele1', 1,
                                      [125000, 0, 0, 0.0]),
                    self.person_group(9, '45to64_2_tele2', 1,
                                      [175000, 0, 0, 0.0]),
                    self.person_group(10, '00to19_3_goout', 1,
                                      [30000, 0, 0, 0.0]),
                    self.person_group(11, '20to44_3_goout', 1,
                                      [50000, 0, 0, 0.0]),
                    self.person_group(12, '45to64_3_goout', 1,
                                      [50000, 0, 0, 0.0]),
                    self.person_group(13, '65to99_3_goout', 1,
                                      [60000, 0, 0, 0.0]),
                ],
                'areas': [
                    self.area_group(
                        0, 'ncroud', {
                            'before_SoE': [[0.0002] * 24, [0.0002] * 24],
                            'before_2_SoE': [[0.0002] * 24, [0.0002] * 24],
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
                            'before_2_SoE': [[0.002] * 24, [0.002] * 24],
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
                            'before_2_SoE': [[0.02] * 24, [0.02] * 24],
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
                        'Tokyo': 0.009,
                        'Chiba': 0.009,
                        'Kanagawa': 0.009,
                        'Saitama': 0.036,
                        'Gunma': 0.036,
                        'Ibaraki': 0.036
                    },
                    'before_2_SoE': {
                        'Tokyo': 0.01,
                        'Chiba': 0.01,
                        'Kanagawa': 0.01,
                        'Saitama': 0.04,
                        'Gunma': 0.04,
                        'Ibaraki': 0.04
                    },
                    '1week_SoE': {
                        'Tokyo': 0.009,
                        'Chiba': 0.009,
                        'Kanagawa': 0.009,
                        'Saitama': 0.036,
                        'Gunma': 0.036,
                        'Ibaraki': 0.036
                    },
                    '2week_SoE': {
                        'Tokyo': 0.006,
                        'Chiba': 0.006,
                        'Kanagawa': 0.006,
                        'Saitama': 0.024,
                        'Gunma': 0.024,
                        'Ibaraki': 0.024
                    },
                    '3week_SoE': {
                        'Tokyo': 0.006,
                        'Chiba': 0.006,
                        'Kanagawa': 0.006,
                        'Saitama': 0.024,
                        'Gunma': 0.024,
                        'Ibaraki': 0.024
                    },
                    '4week_SoE': {
                        'Tokyo': 0.006,
                        'Chiba': 0.006,
                        'Kanagawa': 0.006,
                        'Saitama': 0.024,
                        'Gunma': 0.024,
                        'Ibaraki': 0.024
                    },
                    '5__week_SoE': {
                        'Tokyo': 0.007,
                        'Chiba': 0.007,
                        'Kanagawa': 0.007,
                        'Saitama': 0.028,
                        'Gunma': 0.028,
                        'Ibaraki': 0.028
                    },
                    'after_SoE': {
                        'Tokyo': 0.009,
                        'Chiba': 0.009,
                        'Kanagawa': 0.009,
                        'Saitama': 0.036,
                        'Gunma': 0.036,
                        'Ibaraki': 0.036
                    },
                },
            },
            {
                'name':
                'Ibaraki',
                'p_remove':
                0.1,
                'peaple': [
                    self.person_group(2, '00to19_1', 1, [450000, 0, 0, 0.0]),
                    self.person_group(3, '20to44_1', 1, [240000, 0, 0, 0.0]),
                    self.person_group(4, '45to64_1', 1, [240000, 0, 0, 0.0]),
                    self.person_group(5, '65to99_1', 1, [720000, 0, 0, 0.0]),
                    self.person_group(6, '20to44_2_tele1', 1,
                                      [200000, 0, 0, 0.0]),
                    self.person_group(7, '20to44_2_tele2', 1,
                                      [280000, 0, 0, 0.0]),
                    self.person_group(8, '45to64_2_tele1', 1,
                                      [200000, 0, 0, 0.0]),
                    self.person_group(9, '45to64_2_tele2', 1,
                                      [280000, 0, 0, 0.0]),
                    self.person_group(10, '00to19_3_goout', 1,
                                      [50000, 0, 0, 0.0]),
                    self.person_group(11, '20to44_3_goout', 1,
                                      [80000, 0, 0, 0.0]),
                    self.person_group(12, '45to64_3_goout', 1,
                                      [80000, 0, 0, 0.0]),
                    self.person_group(13, '65to99_3_goout', 1,
                                      [80000, 0, 0, 0.0]),
                ],
                'areas': [
                    self.area_group(
                        0, 'ncroud', {
                            'before_SoE': [[0.0002] * 24, [0.0002] * 24],
                            'before_2_SoE': [[0.0002] * 24, [0.0002] * 24],
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
                            'before_2_SoE': [[0.002] * 24, [0.002] * 24],
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
                            'before_2_SoE': [[0.02] * 24, [0.02] * 24],
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
                        'Tokyo': 0.009,
                        'Chiba': 0.036,
                        'Kanagawa': 0.009,
                        'Saitama': 0.036,
                        'Gunma': 0.036,
                        'Tochigi': 0.036
                    },
                    'before_2_SoE': {
                        'Tokyo': 0.01,
                        'Chiba': 0.04,
                        'Kanagawa': 0.01,
                        'Saitama': 0.04,
                        'Gunma': 0.04,
                        'Tochigi': 0.04
                    },
                    '1week_SoE': {
                        'Tokyo': 0.009,
                        'Chiba': 0.036,
                        'Kanagawa': 0.009,
                        'Saitama': 0.036,
                        'Gunma': 0.036,
                        'Tochigi': 0.036
                    },
                    '2week_SoE': {
                        'Tokyo': 0.006,
                        'Chiba': 0.024,
                        'Kanagawa': 0.006,
                        'Saitama': 0.024,
                        'Gunma': 0.024,
                        'Tochigi': 0.024
                    },
                    '3week_SoE': {
                        'Tokyo': 0.006,
                        'Chiba': 0.024,
                        'Kanagawa': 0.006,
                        'Saitama': 0.024,
                        'Gunma': 0.024,
                        'Tochigi': 0.024
                    },
                    '4week_SoE': {
                        'Tokyo': 0.006,
                        'Chiba': 0.024,
                        'Kanagawa': 0.006,
                        'Saitama': 0.024,
                        'Gunma': 0.024,
                        'Tochigi': 0.024
                    },
                    '5__week_SoE': {
                        'Tokyo': 0.007,
                        'Chiba': 0.028,
                        'Kanagawa': 0.007,
                        'Saitama': 0.028,
                        'Gunma': 0.028,
                        'Tochigi': 0.028
                    },
                    'after_SoE': {
                        'Tokyo': 0.009,
                        'Chiba': 0.036,
                        'Kanagawa': 0.009,
                        'Saitama': 0.036,
                        'Gunma': 0.036,
                        'Tochigi': 0.036
                    },
                },
            },
        ]

        return self._create_citys(condition, smm)
