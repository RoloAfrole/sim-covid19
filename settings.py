import numpy as np


class Group(object):
    def __init__(self, id):
        self.id = id


class Group_Day(Group):
    def __init__(self, id, name='None'):
        super(Group_Day, self).__init__(id)
        self.name = name


Day_Groups = {
    'init_day': Group_Day(0, 'init_day'),
    'before_SoE': Group_Day(1, 'before_SoE'),
    '1week_SoE': Group_Day(2, '1week_SoE'),
    '2week_SoE': Group_Day(3, '2week_SoE'),
    '3week_SoE': Group_Day(4, '3week_SoE'),
    '4week_SoE': Group_Day(5, '4week_SoE'),
    '5__week_SoE': Group_Day(6, '5__week_SoE'),
    'after_SoE': Group_Day(7, 'after_SoE'),
}


class Active_Pattern(object):
    @staticmethod
    def pattern(group, condition, day):
        return np.empty([24, 3])
