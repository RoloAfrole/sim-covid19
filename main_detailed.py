from conductor import Conductor
from initializer import Default_Izer
from datetime import datetime
from absl import app
from absl import flags
import util

flags.DEFINE_string('f', '', 'kernel')

# Simulation Parameters
FLAGS = flags.FLAGS

FLAGS.max_size_per_it = 1000000

mode = True


def sim(argv):
    if mode:
        calc(FLAGS)
    else:
        names = './202005071758'
        plots(names)


def calc(config=FLAGS):
    basename = datetime.now().strftime('%Y%m%d%H%M')
    initzr = Default_Izer(config)
    conductor = Conductor(config, initzr)
    conductor.sim()
    print(conductor.manager.history.h[-1]['status'])
    # util.plot_sim(results, basename)
    util.save_status(conductor.manager.history, basename)


def plots(basename):
    history = util.load_status(basename)
    util.plot_history(history,
                      susceptible=False,
                      infected=True,
                      removed=True,
                      filename=None,
                      save=True,
                      title=None,
                      ylimit=[0, 2000],
                      xlimit=[0, 160])
    # util.save_as_csv(status, basename)


if __name__ == '__main__':
    try:
        app.run(sim)
    except SystemExit:
        pass
