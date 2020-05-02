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
        # basename = './data/202003010352_wn'
        # basename = './data/202003010555_wc'
        names = []
        # names.append('./data/30d_002p')
        # names.append('./data/30d_0018p')
        names.append(('./data/30-60_0h',
                      ['0 hour in crowded zone',
                       '0 hour in crowded zone'], ['o', 'o']))
        names.append(('./data/30-60_1h',
                      ['1 hour in crowded zone',
                       '1 hour in crowded zone'], ['v', 'v']))
        names.append(('./data/30-60_2h',
                      ['2 hour in crowded zone',
                       '2 hour in crowded zone'], ['s', 's']))
        names.append(('./data/30-60_4h',
                      ['4 hour in crowded zone',
                       '4 hour in crowded zone'], ['+', '+']))
        names.append(('./data/30-60_6h',
                      ['6 hour in crowded zone',
                       '6 hour in crowded zone'], ['x', 'x']))
        names.append(('./data/30-60_8h',
                      ['8 hour in crowded zone',
                       '8 hour in crowded zone'], ['1', '1']))
        plots(names)


def calc(config=FLAGS):
    basename = datetime.now().strftime('%Y%m%d%H%M')
    status = rule.initalize_status(config=FLAGS)

    results = rule.sim_infl(status, sim_range=FLAGS.sim_range)

    util.plot_sim(results, basename)
    util.save_status(results, basename)


def plot(basename):
    status = util.load_status(basename)
    util.plot_sim(status, basename, title=None, ylimit=[0, 2000])
    # util.save_as_csv(status, basename)


def plots(names):
    items = []
    for item in names:
        status = util.load_status(item[0])
        items.append((status, item[1], item[2]))
    util.plot_sims(items,
                   infected=True,
                   segregated=False,
                   filename=None,
                   title=None,
                   ylimit=[0, 2000],
                   xlimit=[25, 62])


if __name__ == '__main__':
    app.run(sim)
