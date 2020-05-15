import dask.array as da
import dask
import constant as ct
from absl import flags
import dask.multiprocessing
dask.config.set(scheduler='processes')

# Simulation Parameters
FLAGS = flags.FLAGS

flags.DEFINE_integer('max_size_per_it', 10000000, 'max number of iter')


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

    def get_history(self):
        day_list = [h['day'].date for h in self.h]
        status_array_dic = {}
        for h in self.h:
            for k, v in h['status'].items():
                if k not in status_array_dic:
                    status_array_dic[k] = {}
                for i, j in v.items():
                    if i not in status_array_dic[k]:
                        status_array_dic[k][i] = []
                    status_array_dic[k][i].append(j)

        return day_list, status_array_dic


class Status(object):
    def __init__(self, citys):
        self.citys = citys

    def get_city(self, name):
        for c in self.citys:
            if c.name == name:
                return c

    def get_city_names(self):
        c_names = []
        for c in self.citys:
            c_names.append(c.name)
        return c_names

    def calc_day(self, day):
        records = {c.name: c.get_values(day) for c in self.citys}

        records = self.calc_first(records)

        max_size = FLAGS.max_size_per_it
        for city_name in records.keys():
            N_inner = records[city_name]['N_f_In']
            for start_idx in range(0, N_inner, max_size):
                end_idx = start_idx + max_size
                end_idx = end_idx if N_inner >= end_idx else -1

                c_records = self.calc_city_hour_inner(city_name, records,
                                                      start_idx, end_idx, day)
                c_records = self.calc_city_end_inner(city_name, c_records,
                                                     start_idx, end_idx)
                self.save_city_records_inner(city_name, c_records, start_idx,
                                             end_idx)
            for m_city_name in self.get_city_names():
                if m_city_name in records[city_name]['move_out']:
                    N_out = records[city_name]['move_out'][m_city_name].shape[0]
                    for start_idx in range(0, N_out, max_size):
                        end_idx = start_idx + max_size
                        end_idx = end_idx if N_out >= end_idx else -1
                        c_records = self.calc_city_hour_outer(
                            city_name, m_city_name, records, start_idx,
                            end_idx, day)
                        c_records = self.calc_city_end_outer(
                            city_name, m_city_name, c_records, start_idx,
                            end_idx)
                        self.save_city_records_outer(city_name, m_city_name,
                                                     c_records, start_idx,
                                                     end_idx)

    def calc_first(self, records):
        for k in records.keys():
            temp_N = 0
            temp_I = 0
            temp_N_out = 0
            temp_N += records[k]['inner'].shape[0]
            temp_I += Person.num_with_condition(records[k]['inner_c'],
                                                ct.const.INF)
            for v in records.values():
                if k in v['move_out']:
                    temp_N_out += v['move_out'][k].shape[0]
                    temp_I += Person.num_with_condition(
                        v['move_out']['{}_c'.format(k)], ct.const.INF)

            temp_N += temp_N_out
            records[k]['N_f_In'] = records[k]['inner'].shape[0]
            records[k]['N_f_Out'] = temp_N_out
            records[k]['N_today'] = temp_N
            records[k]['I_today'] = temp_I
            records[k]['correction_NI'] = temp_I / temp_N

        return records

    def calc_city_hour_inner(self, city_name, records, s_idx, e_idx, day):
        k = city_name
        areas = records[k]['areas'] * records[k]['correction_NI']
        records[k]['inner_chs'] = self.calc_hour_chs_iter(
            k, records[k]['inner'], areas, s_idx, e_idx, day)
        return records

    def calc_city_hour_outer(self, city_name, m_city_name, records, s_idx,
                             e_idx, day):
        k = city_name
        areas = records[m_city_name]['areas'] * records[m_city_name][
            'correction_NI']
        records[k]['move_out']['{}_chs'.format(
            m_city_name)] = self.calc_hour_chs_iter(
                city_name, records[k]['move_out'][m_city_name], areas, s_idx,
                e_idx, day)
        return records

    def calc_hour_chs_iter(self, city_name, person_ids, areas, s_idx, e_idx,
                           day):
        city = self.get_city(city_name)
        patterns = city.get_person_patterns(person_ids[s_idx:e_idx].compute(), day)
        p_inf = self.p_inf(patterns[::, 3:], areas)
        return self.vector_chs(p_inf, p_inf.shape)

    def calc_end(self, records):
        for k in records.keys():
            p_r = records[k]['p_remove']
            v_rem = self.vector_chs(p_r, records[k]['inner'].shape[0])
            records[k]['inner_chs'] = da.hstack(
                (records[k]['inner_chs'], v_rem.reshape(-1, 1)))
            for v in records.values():
                if k in v['move_out']:
                    v_rem = self.vector_chs(p_r, v['move_out'][k].shape[0])
                    v['move_out']['{}_chs'.format(k)] = da.hstack(
                        (v['move_out']['{}_chs'.format(k)],
                         v_rem.reshape(-1, 1)))

        return records

    def calc_city_end_inner(self, city_name, records, s_idx, e_idx):
        k = city_name
        p_r = records[k]['p_remove']
        v_rem = self.vector_chs(p_r, records[k]['inner'][s_idx:e_idx].shape[0])
        records[k]['inner_chs'] = da.hstack(
            (records[k]['inner_chs'], v_rem.reshape(-1, 1)))
        return records

    def calc_city_end_outer(self, city_name, m_city_name, records, s_idx,
                            e_idx):
        k = city_name
        p_r = records[m_city_name]['p_remove']
        v_rem = self.vector_chs(
            p_r, records[k]['move_out'][m_city_name][s_idx:e_idx].shape[0])
        records[k]['move_out']['{}_chs'.format(m_city_name)] = da.hstack(
            (records[k]['move_out']['{}_chs'.format(m_city_name)],
             v_rem.reshape(-1, 1)))
        return records

    def save_records(self, records):
        for c in self.citys:
            c.refrect_changes(records[c.name], [cc.name for cc in self.citys])

    def save_city_records_inner(self, city_name, records, s_idx, e_idx):
        c = self.get_city(city_name)
        c.refrect_changes_inner(records[c.name], s_idx, e_idx)

    def save_city_records_outer(self, city_name, m_city_name, records, s_idx,
                                e_idx):
        c = self.get_city(city_name)
        c.refrect_changes_outer(records[c.name], m_city_name, s_idx, e_idx)

    def get_record_for_history(self):
        records = {}
        total = {}
        for c in self.citys:
            records[c.name] = c.get_records()
            for k in records[c.name].keys():
                total[k] = 0

        for v in records.values():
            for k in v.keys():
                total[k] += v[k]
        records['total'] = total

        return records

    @staticmethod
    def vector_chs(p, l):
        rand = da.random.random(l)
        v_chs = rand < p
        v_chs = v_chs.rechunk('auto')
        return v_chs

    @staticmethod
    def p_inf(peaple_pattern, areas):
        return da.sum(peaple_pattern * areas.T, axis=2)


class City(object):
    def __init__(self, name, peaple, areas, move_out, p_remove):
        self.name = name
        self.peaple = peaple
        self.areas = areas
        self.move_out = move_out
        self.p_remove = p_remove

    def get_values(self, day):
        values = {}
        mo, inner = self.sim_move_out(day)
        values['move_out'] = mo
        values['inner'] = inner[:, 0]
        values['inner_c'] = inner[:, 1]

        values['areas'] = da.from_array(
            [area.get_param(day) for area in self.areas], chunks=(-1, -1))
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

        stacked_record = da.vstack(r_list)
        changes = da.vstack(c_list)
        i_chs = self.get_infected_changes(changes)
        rem_chs = self.get_removed_changes(changes)
        ids = stacked_record[::, 0].compute()
        self.change_peaple(ids, i_chs, rem_chs)

    def refrect_changes_inner(self, record, s_idx, e_idx):
        changes = record['inner_chs']
        i_chs = self.get_infected_changes(changes)
        rem_chs = self.get_removed_changes(changes)
        ids = record['inner'][s_idx:e_idx].compute()
        self.change_peaple(ids, i_chs, rem_chs)

    def refrect_changes_outer(self, record, city_name, s_idx, e_idx):
        changes = record['move_out']['{}_chs'.format(city_name)]
        i_chs = self.get_infected_changes(changes)
        rem_chs = self.get_removed_changes(changes)
        ids = record['move_out'][city_name][s_idx:e_idx].compute()
        self.change_peaple(ids, i_chs, rem_chs)

    def change_peaple(self, ids, i_chs, rem_chs):
        for i, idx in enumerate(ids):
            p = self.peaple[idx]
            p.change(i_chs[i][0], rem_chs[i][0])

    def get_records(self):
        key_list = []
        record = {}
        for p in self.peaple:
            if p.condition not in key_list:
                key_list.append(p.condition)
                record[p.condition] = 0
            record[p.condition] += 1

        return record

    def get_person_patterns(self, ids, day):
        pattern_list = []
        for idx in ids:
            p = self.peaple[idx]
            pattern_list.append(dask.delayed(p.get_values)(day))
        # return da.vstack(pattern_list)
        al = dask.delayed(list)(pattern_list)
        return da.vstack(al.compute())

    @staticmethod
    def get_infected_changes(changes):
        i_chs = changes[::, :-1]
        all_true = da.ones(i_chs.shape[1], dtype='bool', chunks=(-1))
        chs = da.dot(i_chs, all_true).reshape((-1, 1))
        return chs.compute()

    @staticmethod
    def get_removed_changes(changes):
        rem_chs = changes[::, -1:]
        return rem_chs.compute()


class Person(object):
    def __init__(self, id, group, condition, group_name='None'):
        self.id = id
        self.group = group
        self.condition = condition
        self.group_name = group_name

    def get_values(self, day):
        from settings import Active_Pattern
        pattern = Active_Pattern.pattern(self.group, self.condition, day)
        person = da.from_array(
            [[self.id] * pattern.shape[1], [self.group] * pattern.shape[1],
             [self.condition] * pattern.shape[1]], chunks=(-1, -1))
        values = da.vstack([person, pattern])
        return values.reshape(1, values.shape[0], values.shape[1])

    def get_id(self):
        return self.id

    def change(self, infected, removed):
        if removed and self.condition == ct.const.INF:
            self.condition = ct.const.REM

        if infected and self.condition == ct.const.SUS:
            self.condition = ct.const.INF

    @staticmethod
    def num_with_condition(values, target):
        n = da.count_nonzero(values == target)
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
        t_np = da.from_array([[t.id, t.condition] for t in targets], chunks=('auto', -1))
        num_mo = {}
        for k, v in self.get_pattern(day).items():
            m_num = t_np.shape[0] * v
            num_mo[k] = m_num

        new_t_np = t_np[da.random.
                        choice(t_np.shape[0], t_np.shape[0], replace=False), :]
        # splited = da.split(new_t_np, [int(v) for v in num_mo.values()])
        splited = []
        start_idx = 0
        for v in num_mo.values():
            i = start_idx + int(v)
            splited.append(new_t_np[start_idx:i, :])
            start_idx = i
        splited.append(new_t_np[start_idx:-1, :])

        mo = {}
        for i, k in enumerate(num_mo.keys()):
            mo[k] = splited[i][:, 0]
            mo['{}_c'.format(k)] = splited[i][:, 1]

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
