from tqdm import tqdm
from absl import flags


class Conductor(object):
    def __init__(self, config, initializer):
        self.config = config
        self.manager = initializer.create_manager()

    def set_condition(self, condition):
        pass

    def sim(self, start=None, end=None):
        if start is None:
            start = self.manager.srange.start_position

        if end is None:
            end = self.manager.srange.end_position

        if start < 0 or end > len(self.manager.srange.days):
            raise ValueError('out of range')

        with tqdm(range(start, end), leave=False) as pb_day:
            for day in pb_day:
                self.manager.calc_day(day)

    def plot(self):
        pass
