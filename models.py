import numpy as np
import constant as ct


class Manager(object):
    def __init__(self, status, srange, history=None):
        self.status = status
        self.srange = srange

        self.history = history
        if self.history is None:
            self.history = History()
            self.history.set_init_record(self.status)

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
        record['status'] = status.get_record_for_history()
        self.h.append(record)


class Status(object):
    def __init__(self, citys):
        self.citys = citys

    def calc_day(self, day):
        records = {c.name: c.get_values(day) for c in self.citys}

        records = self.calc_first(records)
        records = self.calc_hour(records)
        records = self.calc_end(records)
        self.save_records(records)

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
            areas = records[k]['areas'] * records[k]['correction_NI']
            p_inf = self.p_inf(records[k]['inner'][::, 3:], areas)
            v_inf = self.vector_chs(p_inf, *p_inf.shape)
            records[k]['inner_chs'] = v_inf
            for v in records.values():
                if k in v['move_out']:
                    p_inf = self.p_inf(v['move_out'][k][::, 3:], areas)
                    v_inf = self.vector_chs(p_inf, *p_inf.shape)
                    v['move_out']['{}_chs'.format(k)] = v_inf

        return records

    def calc_end(self, records):
        for k in records.keys():
            p_r = records[k]['p_remove']
            v_rem = self.vector_chs(p_r, records[k]['inner'].shape[0])
            records[k]['inner_chs'] = np.hstack(
                (records[k]['inner_chs'], v_rem.reshape(-1, 1)))
            for v in records.values():
                if k in v['move_out']:
                    v_rem = self.vector_chs(p_r, v['move_out'][k].shape[0])
                    v['move_out']['{}_chs'.format(k)] = np.hstack(
                        (v['move_out']['{}_chs'.format(k)],
                         v_rem.reshape(-1, 1)))

        return records

    def save_records(self, records):
        for c in self.citys:
            c.refrect_changes(records[c.name], [cc.name for cc in self.citys])

    def get_record_for_history(self):
        records = {}
        total = {}
        v_keys = None
        for c in self.citys:
            records[c.name] = c.get_records()
            v_keys = records[c.name].keys()

        for k in v_keys:
            total[k] = 0

        for v in records.values():
            for k in v.keys():
                total[k] += v[k]
        records['total'] = total

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
        # targets = [p for p in self.peaple if p.condition != ct.const.REM]
        targets = self.peaple
        return targets

    def refrect_changes(self, record, city_names):
        mo = record['move_out']
        r_list = [record['inner']]
        r_list.extend([mo[k] for k in city_names if k in mo])
        c_list = [record['inner_chs']]
        c_list.extend([mo['{}_chs'.format(k)] for k in city_names if k in mo])

        stacked_record = np.vstack(r_list)
        changes = np.vstack(c_list)
        i_chs = self.get_infected_changes(changes)
        rem_chs = self.get_removed_changes(changes)
        ids = stacked_record[::, :1]
        self.change_peaple(ids, i_chs, rem_chs)

    def change_peaple(self, ids, i_chs, rem_chs):
        for p in self.peaple:
            target = np.where(ids[:, 0, 0] == p.id)
            p.change(i_chs[target][0], rem_chs[target][0])

    def get_records(self):
        key_list = []
        record = {}
        for p in self.peaple:
            if p.condition not in key_list:
                key_list.append(p.condition)
                record[p.condition] = 0
            record[p.condition] += 1

        return record

    @staticmethod
    def get_infected_changes(changes):
        i_chs = changes[::, :-1]
        all_true = np.ones(i_chs.shape[1], dtype='bool')
        chs = np.dot(i_chs, all_true).reshape((-1, 1))
        return chs

    @staticmethod
    def get_removed_changes(changes):
        rem_chs = changes[::, -1:]
        return rem_chs


class Person(object):
    def __init__(self, id, group, condition, group_name='None'):
        self.id = id
        self.group = group
        self.condition = condition
        self.group_name = group_name

    def get_values(self, day):
        from settings import Active_Pattern
        pattern = Active_Pattern.pattern(self.group, self.condition, day)
        person = np.zeros([3, pattern.shape[1]])
        person[0][0] = self.id
        person[1][0] = self.group
        person[2][0] = self.condition

        return np.vstack([person, pattern])

    def change(self, infected, removed):
        if removed and self.condition == ct.const.INF:
            self.condition = ct.const.REM

        if infected and self.condition == ct.const.SUS:
            self.condition = ct.const.INF

    @staticmethod
    def num_with_condition(values, target):
        n = np.count_nonzero(values[::, 2:3, 0] == target)
        return n


class Area(object):
    def __init__(self, name, group, patterns):
        self.name = name
        self.group = group
        self.patterns = patterns

    def get_param(self, day):
        day_group_name = day.group.name
        idx = 1 if day.isHoliday else 0
        return self.patterns[day_group_name][idx]


class MoveOut(object):
    def __init__(self, pattern):
        self.pattern = pattern

    def move(self, targets, day):
        t_np = np.array([t.get_values(day) for t in targets])
        num_mo = {}
        for k, v in self.get_pattern(day).items():
            m_num = t_np.shape[0] * v
            num_mo[k] = m_num

        # new_t_np = np.random.choice(t_np, t_np.shape[0], replace=False)
        new_t_np = t_np[np.random.
                        choice(t_np.shape[0], t_np.shape[0], replace=False), :]
        splited = np.split(new_t_np, [int(v) for v in num_mo.values()])

        mo = {}
        for i, k in enumerate(num_mo.keys()):
            mo[k] = splited[i]

        inner = splited[-1]
        return mo, inner

    def get_pattern(self, day):
        return self.pattern[day.group.name]


class SimRange(object):
    def __init__(self, days, start_position, end_position):
        self.days = days
        self.start_position = start_position
        self.end_position = end_position

    def get_day(self, t):
        return self.days[t]


class Day(object):
    def __init__(self, date, group, holiday=None):
        self.date = date
        self.group = group
        self.holiday = holiday
        if self.holiday is None:
            self.holiday = self.isHoliday(self.date)

    @staticmethod
    def get_initial_day():
        from settings import Day_Groups
        return Day('initail day', Day_Groups['init_day'], False)

    @staticmethod
    def isHoliday(date):
        return False
