import numpy as np
import constant as ct
from absl import flags

from multiprocessing.managers import SharedMemoryManager
from multiprocessing import shared_memory
from multiprocessing import Pool

from settings import Active_Pattern

# Simulation Parameters
FLAGS = flags.FLAGS

flags.DEFINE_integer('max_size_per_it', 10000, 'max number of iter')
flags.DEFINE_integer('pool_size', 8, 'number of multi proceses')


class Manager(object):
    def __init__(self, status, srange, smm, history=None):
        self.status = status
        self.srange = srange

        self.smm = smm

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
        record['dist'] = status.get_record_for_dist()
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
    def __init__(self, citys, smm):
        self.citys = citys
        self.smm = smm

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
        pool_size = FLAGS.pool_size
        pool_obj = Pool(pool_size)

        total_args = []
        for city_name in records.keys():
            # make args for inner
            inner_args = create_args_for_inner(records, city_name, max_size,
                                               day)
            total_args.extend(inner_args)
            # pool_obj.map(calc_day_with_pool, inner_args)

            for m_city_name in self.get_city_names():
                if m_city_name in records[city_name]['moved_idx']:
                    # make args for outer
                    outer_args = create_args_for_outer(records, city_name,
                                                       m_city_name, max_size,
                                                       day)
                    total_args.extend(outer_args)
                    # pool_obj.map(calc_day_with_pool, outer_args)

        pool_obj.map(calc_day_with_pool, total_args)

    def calc_first(self, records):
        for k in records.keys():
            temp_N = 0
            temp_I = 0
            temp_N_out = 0
            temp_I += Person.num_with_condition(
                records[k]['targets'][0]
                [records[k]['moved_idx']['inner'][0]][:, 2], ct.const.INF)
            for v in records.values():
                if k in v['moved_idx']:
                    temp_N_out += v['moved_idx'][k][0].shape[0]
                    temp_I += Person.num_with_condition(
                        v['targets'][0][v['moved_idx'][k][0]][:, 2], ct.const.INF)

            records[k]['N_f_In'] = records[k]['moved_idx']['inner'][0].shape[0]
            records[k]['N_f_Out'] = temp_N_out
            temp_N = records[k]['N_f_In'] + temp_N_out
            records[k]['N_today'] = temp_N
            records[k]['I_today'] = temp_I
            records[k]['correction_NI'] = temp_I / temp_N
            records[k]['areas'][
                0][:] = records[k]['areas'][0][:] * records[k]['correction_NI']

        return records

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

    def get_record_for_dist(self):
        records = {}
        for c in self.citys:
            records[c.name] = c.get_dist()

        return records


class City(object):
    def __init__(self, name, peaple, areas, move_out, p_remove, smm):
        self.name = name
        self.peaple = peaple
        self.areas = areas
        self.move_out = move_out
        self.p_remove = p_remove
        self.smm = smm

    def get_values(self, day):
        values = {}
        moved_target, move_ids = self.sim_move_out(day)
        values['targets'] = (moved_target, self.peaple[1])
        values['moved_idx'] = self.create_moved_index(moved_target, move_ids)
        values['areas'] = create_numpy_shm(
            np.array([area.get_param(day) for area in self.areas]), self.smm)
        values['p_remove'] = self.get_p_remove(day)
        return values

    def create_moved_index(self, moved_target, move_ids):
        moved_indexs = {}
        for k, v in move_ids.items():
            moved_indexs[k] = np.where(moved_target[:, 3] == v)[0]

        moved_indexs_shm = {}
        for k, v in moved_indexs.items():
            moved_indexs_shm[k] = create_numpy_shm(v, self.smm)

        return moved_indexs_shm

    def sim_move_out(self, day):
        targets = self.get_targets()
        moved_target, move_ids = self.move_out.move(targets, day, self.smm)
        return moved_target, move_ids

    def get_p_remove(self, day):
        day_group_name = day.group.name
        isholiday = day.holiday
        if type(self.p_remove) is float:
            return self.p_remove
        elif type(self.p_remove) is dict:
            return self.p_remove[day_group_name][1 if isholiday else 0]

    def get_targets(self):
        # targets = [p for p in self.peaple if p.condition != ct.const.REM]
        targets = self.peaple[0]
        return targets

    def get_records(self):
        key_list = [0, 1, 2]
        record = {}
        for k in key_list:
            record[k] = np.count_nonzero(self.peaple[0][:, 2] == k)
        return record

    def get_dist(self):
        from settings import Peaple_Groups
        key_list = [0, 1, 2]
        record = {}

        for pg, pg_i in Peaple_Groups.items():
            indexs = np.where(self.peaple[0][:, 1] == pg_i)
            if indexs[0].shape[0] != 0:
                record[pg] = {}
                for k in key_list:
                    record[pg][k] = np.count_nonzero(
                        self.peaple[0][indexs][:, 2] == k)
        return record


class Person(object):
    def __init__(self, id, group, condition, group_name='None'):
        self.id = id
        self.group = group
        self.condition = condition
        self.group_name = group_name

    def get_values(self, day):
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
        n = np.count_nonzero(values == target)
        return n

    @staticmethod
    def get_valuesday_np(day, array, idx):
        group = array[idx][1]
        condition = array[idx][2]
        pattern = Active_Pattern.pattern(group, condition, day)
        return pattern

    @staticmethod
    def change_np(infected, removed, array, idx):
        condition = array[idx][2]
        if removed and condition == ct.const.INF:
            array[idx][2] = ct.const.REM

        if infected and condition == ct.const.SUS:
            array[idx][2] = ct.const.INF


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

    def move(self, targets, day, smm):
        num_mo = {}
        num_inner = targets.shape[0]
        move_ids = {'inner': 0}
        idx = 1
        for k, v in self.get_pattern(day).items():
            m_num = int(targets.shape[0] * v)
            num_mo[k] = m_num
            num_inner -= m_num
            move_ids[k] = idx
            idx += 1

        i_array = np.ones(num_inner, dtype=int) * 0
        m_list = [i_array]
        for k, v in num_mo.items():
            m_list.append(np.ones(v, dtype=int) * move_ids[k])

        m_array = np.hstack(m_list)
        m_array = np.random.choice(m_array, m_array.shape[0], replace=False)
        targets[:, 3] = m_array

        return targets, move_ids

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


def create_numpy_shm(narray, smm):
    shm = smm.SharedMemory(size=narray.nbytes)
    shm_array = np.ndarray(narray.shape, dtype=narray.dtype, buffer=shm.buf)
    shm_array[:] = narray[:]
    shm_name = shm.name
    return shm_array, shm_name, shm


def load_numpy_shm(shm_name, shape, dtype):
    shm = shared_memory.SharedMemory(name=shm_name)
    shm_array = np.ndarray(shape, dtype=dtype, buffer=shm.buf)
    return shm_array, shm


def create_args_for_inner(records, c_name, split_size, day):
    total_num = records[c_name]['N_f_In']

    args_list = []

    base_args = []
    base_args.append(records[c_name]['areas'][1])
    base_args.append(records[c_name]['areas'][0].shape)
    base_args.append(records[c_name]['targets'][1])
    base_args.append(records[c_name]['targets'][0].shape)
    base_args.append(records[c_name]['moved_idx']['inner'][1])
    base_args.append(records[c_name]['moved_idx']['inner'][0].shape)

    base_args.append(day.group.name)
    base_args.append(day.holiday)

    base_args.append(records[c_name]['p_remove'])

    for start_idx in range(0, total_num, split_size):
        end_idx = start_idx + split_size
        end_idx = end_idx if total_num >= end_idx else -1
        args = [start_idx, end_idx]
        args.extend(base_args)

        args_list.append(tuple(args))

    return args_list


def create_args_for_outer(records, c_name, to_c_name, split_size, day):
    total_num = records[c_name]['moved_idx'][to_c_name][0].shape[0]

    args_list = []

    base_args = []
    base_args.append(records[to_c_name]['areas'][1])
    base_args.append(records[to_c_name]['areas'][0].shape)
    base_args.append(records[c_name]['targets'][1])
    base_args.append(records[c_name]['targets'][0].shape)
    base_args.append(records[c_name]['moved_idx'][to_c_name][1])
    base_args.append(records[c_name]['moved_idx'][to_c_name][0].shape)

    base_args.append(day.group.name)
    base_args.append(day.holiday)

    base_args.append(records[to_c_name]['p_remove'])

    for start_idx in range(0, total_num, split_size):
        end_idx = start_idx + split_size
        end_idx = end_idx if total_num >= end_idx else -1
        args = [start_idx, end_idx]
        args.extend(base_args)

        args_list.append(tuple(args))

    return args_list


def calc_day_with_pool(args):
    s_idx = args[0]
    e_idx = args[1]
    areas_name = args[2]
    areas_shape = args[3]
    targets_name = args[4]
    targets_shape = args[5]
    moved_idxs_name = args[6]
    moved_idxs_shape = args[7]
    day_group_name = args[8]
    day_isholiday = args[9]
    p_remove = args[10]

    areas, areas_shm = load_numpy_shm(areas_name, areas_shape, np.float64)
    targets, targets_shm = load_numpy_shm(targets_name, targets_shape,
                                          np.int32)
    moved_idxs, moved_idxs_shm = load_numpy_shm(moved_idxs_name,
                                                moved_idxs_shape, np.int64)

    patterns_list = []
    for t in targets[moved_idxs][s_idx:e_idx]:
        patterns_list.append(
            Active_Pattern.pattern(t[1], t[2], day_group_name, day_isholiday))

    patterns_array = np.array(patterns_list)
    p_inf = pattern_infection(patterns_array, areas)

    inf_changes = create_changes(p_inf, *p_inf.shape)
    inf_changes = calc_is_infection(inf_changes)
    rem_changes = create_changes(p_remove, moved_idxs[s_idx:e_idx].shape[0])

    for i, t in enumerate(targets[moved_idxs][s_idx:e_idx]):
        condition = t[2]
        if condition == ct.const.SUS and inf_changes[i][0]:
            targets[t[0], 2] = ct.const.INF
        if condition == ct.const.INF and rem_changes[i]:
            targets[t[0], 2] = ct.const.REM


def create_changes(p, *shape):
    rand = np.random.rand(*shape)
    v_chs = rand < p
    return v_chs


def pattern_infection(peaple_pattern, areas):
    return np.sum(peaple_pattern * areas.T, axis=2)


def calc_is_infection(changes):
    all_true = np.ones(changes.shape[1], dtype='bool')
    chs = np.dot(changes, all_true).reshape((-1, 1))
    return chs
