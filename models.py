import numpy as np

import random
from tqdm import tqdm
from absl import flags


class Manager(object):
    def __init__(self, history, status, srange):
        self.history = history
        self.status = status
        self.srange = srange

    def calc_day(self, t):
        day = self.srange.get_day(t)


class History(object):
    def __init__(self):
        self.history = []

    def set_init_record(self, status):
        self.add(Day.get_initial_day(), status)

    def add(self, day, status):
        record = {}
        record['day'] = day
        record['status'] = status
        self.history.append(record)


class Status(object):
    def __init__(self, citys):
        self.citys = citys


class City(object):
    def __init__(self, name, peaple, areas, move_out):
        self.name = name
        self.peaple = peaple
        self.areas = areas
        self.move_out = move_out


class Person(object):
    def __init__(self, id, group, condition, active_pattern):
        self.id = id
        self.group = group
        self.condition = condition
        self.active_pattern = active_pattern


class Area(object):
    def __init__(self, name, group):
        self.name = name
        self.group = group


class MoveOut(object):
    def __init__(self):
        self.pattern = {}


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
