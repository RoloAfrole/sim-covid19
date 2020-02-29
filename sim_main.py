import infl_rule as rule
import util
from datetime import datetime
from absl import app
from absl import flags

flags.DEFINE_string('f', '', 'kernel')

# Simulation Parameters
FLAGS = flags.FLAGS

FLAGS.sim_range = 2

FLAGS.total_population = 120000000
FLAGS.init_infl = 10
FLAGS.init_segregated = 0

FLAGS.rate_red = 0.1
FLAGS.rate_green = 0.01
FLAGS.rate_blue = 0.001

FLAGS.red_per_day = 8
FLAGS.green_per_day = 8
FLAGS.blue_per_day = 8

FLAGS.prob_segregate = 0.1

mode = False


def sim(argv):
    if mode:
        calc(FLAGS)
    else:
        basename = './data/base_30day'
        plot(basename)


def calc(config=FLAGS):
    basename = datetime.now().strftime('%Y%m%d%H%M')
    status = rule.initalize_status(config=FLAGS)

    results = rule.sim_infl(status, sim_range=FLAGS.sim_range)

    util.plot_sim(results, basename)
    util.save_status(results, basename)


def plot(basename):
    status = util.load_status(basename)
    util.plot_sim(status, basename)


if __name__ == '__main__':
    app.run(sim)
