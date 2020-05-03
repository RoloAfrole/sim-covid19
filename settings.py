

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