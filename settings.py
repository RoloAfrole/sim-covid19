import numpy as np

class Group(object):
    def __init__(self, id):
        self.id = id


class Group_Day(Group):
    def __init__(self, id):
        super(Group_Day, self).__init__(id)
        pass


Day_Groups = {
    'init_day': Group_Day(0),
}


class Active_Pattern(object):
    @staticmethod
    def pattern(group, condition, pattern, day):
        return np.empty([24, 3])
