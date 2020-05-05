import numpy as np

import random
from tqdm import tqdm
from absl import flags

import constant as ct

import datetime
import models


class Initializer(object):
    def __init__(self, config):
        self.config

    def create_status(self):
        raise NotImplementedError

    def create_srange(self):
        raise NotImplementedError

    def create_citys(self):
        raise NotImplementedError

    def create_manager(self):
        raise NotImplementedError

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
