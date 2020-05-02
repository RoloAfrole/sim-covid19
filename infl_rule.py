import random
from tqdm import tqdm
from absl import flags

# Parameters
flags.DEFINE_integer('total_population', 100000000,
                     'total number of population')
flags.DEFINE_integer('init_infl', 10, 'inital number of influencers')
flags.DEFINE_integer('init_segregated', 0,
                     'inital number of segregated influencers')

flags.DEFINE_float('rate_red', 0.50, 'coefficient rate of infled in red zone')
flags.DEFINE_float('rate_green', 0.20,
                   'coefficient rate of infled in green zone')
flags.DEFINE_float('rate_blue', 0.05,
                   'coefficient rate of infled in blue zone')

flags.DEFINE_integer('red_per_day', 8, 'hour per day in red zone')
flags.DEFINE_integer('green_per_day', 8, 'hour per day in green zone')
flags.DEFINE_integer('blue_per_day', 8, 'hour per day in blue zone')

flags.DEFINE_float('prob_segregate', 0.01, 'probability of segregation')

flags.DEFINE_integer('sim_range', 90, 'day of sim')

flags.DEFINE_bool('sim_per_hour', False, 'sim each hour or each day')
flags.DEFINE_integer('delay_infl', 0, 'delay of having infl')

FLAGS = flags.FLAGS


class Status(object):
    def __init__(self, init_pop, init_infl, init_seg, rate_red, rate_green,
                 rate_blue, red_per_day, green_per_day, blue_per_day):
        self.pop = init_pop
        self.infl = init_infl
        self.seg = init_seg

        self.rate_red = rate_red
        self.rate_green = rate_green
        self.rate_blue = rate_blue

        self.red_per_day = red_per_day
        self.green_per_day = green_per_day
        self.blue_per_day = blue_per_day

        self.h_pop = [init_pop]
        self.h_infl = [init_infl]
        self.h_seg = [init_seg]

        self.h_zones = [[0, self.get_zone_p]]

    def update_pop(self, pop, infl, seg):
        self.h_pop.append(pop)
        self.h_infl.append(infl)
        self.h_seg.append(seg)

        self.pop = pop
        self.infl = infl
        self.seg = seg

    def get_zone_p(self):
        return [[self.rate_red, self.red_per_day],
                [self.rate_green, self.green_per_day],
                [self.rate_blue, self.blue_per_day]]

    def update_zone_p(self, new_zones):
        self.h_zones.append([len(self.h_pop), new_zones])

        self.rate_red = new_zones[0][0]
        self.rate_green = new_zones[1][0]
        self.rate_blue = new_zones[2][0]

        self.red_per_day = new_zones[0][1]
        self.green_per_day = new_zones[1][1]
        self.blue_per_day = new_zones[2][1]


def initalize_status(config):
    status = Status(init_pop=config.total_population - config.init_infl,
                    init_infl=config.init_infl,
                    init_seg=config.init_segregated,
                    rate_red=config.rate_red,
                    rate_green=config.rate_green,
                    rate_blue=config.rate_blue,
                    red_per_day=config.red_per_day,
                    green_per_day=config.green_per_day,
                    blue_per_day=config.blue_per_day)
    return status


class Status_Detail(Status):
    def __init__(self, init_pop, init_infl, init_seg, rate_red, rate_green,
                 rate_blue, red_per_day, green_per_day, blue_per_day):
        super(Status, self).__init__(init_pop, init_infl, init_seg, rate_red,
                                     rate_green, rate_blue, red_per_day,
                                     green_per_day, blue_per_day)


def sim_infl(status, sim_range):
    with tqdm(range(sim_range), leave=False) as pb_day:
        pb_day.set_description('Day')
        for day in pb_day:
            if status.pop <= 0:
                break
            infl_day(status)

    return status


def infl_day(status):
    infl_prob = status.infl / (status.pop + status.infl)
    if FLAGS.delay_infl > 0 and len(status.h_infl) >= FLAGS.delay_infl:
        delayed_infl = status.h_infl[-FLAGS.delay_infl] - (
            status.seg - status.h_seg[-FLAGS.delay_infl])
        infl_prob = delayed_infl / (status.h_pop[-FLAGS.delay_infl] +
                                    delayed_infl)
    tmp_pop = status.pop
    sum_infled = 0

    for zone_p in status.get_zone_p():
        if FLAGS.sim_per_hour:
            infled_num = infl_zone(infl_prob, zone_p[0], tmp_pop, zone_p[1])
        else:
            infled_num = infl_zone_w_rate(infl_prob, zone_p[0], tmp_pop,
                                          zone_p[1])
        tmp_pop -= infled_num
        sum_infled += infled_num

    segregeted_num = segregate_infl(status.infl)

    new_pop = status.pop - sum_infled
    new_infl = status.infl + sum_infled - segregeted_num
    # if new_infl <= 0:
    #     new_infl = 1
    new_seg = status.seg + segregeted_num

    status.update_pop(new_pop, new_infl, new_seg)

    return status


def infl_zone_w_rate(infl_prob, zone_rate, pop, zone_hour):
    init_pop = pop
    tmp_pop = init_pop
    infled_num = 0
    sum_infled_num = 0
    infled_num = infl_hour(infl_prob, zone_rate * zone_hour, tmp_pop)

    sum_infled_num += infled_num

    return sum_infled_num


def infl_zone(infl_prob, zone_rate, pop, zone_hour):
    init_pop = pop
    tmp_pop = init_pop
    infled_num = 0
    sum_infled_num = 0
    for i in range(zone_hour):
        infled_num = infl_hour(infl_prob, zone_rate, tmp_pop)
        tmp_pop -= infled_num

        sum_infled_num += infled_num

    return sum_infled_num


def infl_hour(infl_prob, zone_rate, pop):
    zone_prob = infl_prob * zone_rate
    infled_num = 0
    for i in range(pop):
        if random.random() < zone_prob:
            infled_num += 1

    return infled_num


def segregate_infl(infl):
    prob = FLAGS.prob_segregate
    segregated_num = 0
    for i in range(infl):
        if random.random() < prob:
            segregated_num += 1

    return segregated_num
