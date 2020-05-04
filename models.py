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
        records = {c.name: c.get_values(day) for c in self.citys}

        records = self.calc_first(records)
        records = self.calc_hour(records)
        records = self.calc_end(records)

    def calc_first(self, records):
        for k in records.keys():
            temp_N = 0
            temp_I = 0
            temp_N += records[k]['inner'].shape[0]
            temp_I += Person.num_with_condition(records[k]['inner'],
                                                ct.const.INF)
            for v in records.values():
                if k in v['move_out']:
                    temp_N += v['move_out'][k].shape[0]
                    temp_I += Person.num_with_condition(
                        v['move_out'][k], ct.const.INF)

            records[k]['N_today'] = temp_N
            records[k]['I_today'] = temp_I
            records[k]['correction_NI'] = temp_I / temp_N

        return records

    def calc_hour(self, records):
        for k in records.keys():
            areas = records[k]['areas']
            p_inf = self.p_inf(records[k]['inner'][::, 4:], areas)
            v_inf = self.vector_chs(p_inf, *p_inf.shape)
            records[k]['inner_chs'] = v_inf
            for v in records.values():
                if k in v['move_out']:
                    p_inf = self.p_inf(v['move_out'][k][::, 4:], areas)
                    v_inf = self.vector_chs(p_inf, *p_inf.shape)
                    v['move_out']['{}_chs'.format(k)] = v_inf

        return records

    def calc_end(self, records):
        for k in records.keys():
            p_r = records[k]['p_remove']
            v_rem = self.vector_chs(p_r, records[k]['inner'].shape[0])
            records[k]['inner_chs'] = np.stack(
                [records[k]['inner_chs'], v_rem])
            for v in records.values():
                if k in v['move_out']:
                    v_rem = self.vector_chs(p_r, v['move_out'][k].shape[0])
                    v['move_out']['{}_chs'.format(k)] = np.stack(
                        [v['move_out']['{}_chs'.format(k)], v_rem])

        return records

    @staticmethod
    def vector_chs(p, *l):
        rand = np.random.rand(*l)
        v_chs = rand < p
        return v_chs

    @staticmethod
    def p_inf(peaple_pattern, areas):
        return np.sum(peaple_pattern * areas.T, axis=2)


class City(object):
    def __init__(self, name, peaple, areas, move_out, p_remove):
        self.name = name
        self.peaple = peaple
        self.areas = areas
        self.move_out = move_out
        self.p_remove = p_remove

    def get_values(self, day):
        values = {}
        values['move_out'], values['inner'] = self.sim_move_out(day)
        values['areas'] = np.array(
            [area.get_param(day) for area in self.areas])
        values['p_remove'] = self.p_remove
        return values

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
        from settings import Active_Pattern
        pattern = Active_Pattern.pattern(self.group, self.condition,
                                         self.pattern, day)
        person = np.zeros([4, pattern.shape[1]])
        person[0][0] = self.id
        person[1][0] = self.group
        person[2][0] = self.condition
        person[3][0] = self.active_pattern

        return np.vstack([person, pattern])

    @staticmethod
    def num_with_condition(values, target):
        n = np.count_nonzero(values[::, 2:3, 0] == target)
        return n


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
            m_num = t_np.shape[0] * v
            num_mo[k] = m_num

        new_t_np = np.random.choice(t_np, t_np.shape[0], replace=False)
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
