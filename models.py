import numpy as np

import random
from tqdm import tqdm
from absl import flags

import constant as ct


class Manager(object):
    def __init__(self, history, status, srange):
        self.history = history
        self.status = status
        self.srange = srange

    def calc_day(self, t):
        day = self.srange.get_day(t)
        self.status.calc_day(day)
        self.history.add(day, self.status)


class History(object):
    def __init__(self):
        self.h = []

    def set_init_record(self, status):
        self.add(Day.get_initial_day(), status)

    def add(self, day, status):
        record = {}
        record['day'] = day
        record['status'] = status
        self.h.append(record)


class Status(object):
    def __init__(self, citys):
        self.citys = citys

    def calc_day(self, day):
        records = [c.get_values(day) for c in self.citys]
        records = {c.name: c.get_values(day) for c in self.citys}

        records = self.calc_first(records, day)
        for t in range(24):
            records = self.calc_hour(records, day)
        records = self.calc_end(records, day)

    def calc_first(self, records, day):
        return records

    def calc_hour(self, records, day, hour):
        return records

    def calc_end(self, records, day):
        return records


class City(object):
    def __init__(self, name, peaple, areas, move_out):
        self.name = name
        self.peaple = peaple
        self.areas = areas
        self.move_out = move_out

    def get_values(self, day):
        values = {}
        values['move_out'], values['inner'] = self.sim_move_out(day)
        values['areas'] = [area.get_param(day) for area in self.areas]

    def sim_move_out(self, day):
        targets = self.get_targets()
        mo, inner = self.move_out.move(targets, day)
        return mo, inner

    def get_targets(self):
        targets = [p for p in self.peaple if p.condition != ct.const.REM]
        return targets


class Person(object):
    def __init__(self, id, group, condition, active_pattern):
        self.id = id
        self.group = group
        self.condition = condition
        self.active_pattern = active_pattern

    def get_values(self, day):
        return [self.id]


class Area(object):
    def __init__(self, name, group):
        self.name = name
        self.group = group

    def get_param(self, day):
        pass


class MoveOut(object):
    def __init__(self):
        self.pattern = {
            'grouptype': {
                'cityname': 0.1,
            },
        }

    def move(self, targets, day):
        t_np = np.array([t.get_values(day) for t in targets])
        num_mo = {}
        for k, v in self.get_pattern(day).items():
            m_num = t_np.size * v
            num_mo[k] = m_num

        new_t_np = np.random.choice(t_np, t_np.size, replace=False)
        splited = np.split(new_t_np, list(num_mo.values()))

        mo = {}
        for i, k in enumerate(num_mo.keys()):
            mo[k] = splited[i]

        inner = splited[-1]
        return mo, inner

    def get_pattern(self, day):
        if day.group not in self.pattern:
            return {'None': 0.0}
        return self.pattern[day.group]


class SimRange(object):
    def __init__(self, days, start_position, end_position):
        self.days = days
        self.start_position = start_position
        self.end_position = end_position

    def get_day(self, t):
        return self.days[t]


class Day(object):
    def __init__(self, date, group):
        self.date = date
        self.group = group

    @staticmethod
    def get_initial_day():
        from settings import Day_Groups
        return Day('initail day', Day_Groups['init_day'])
