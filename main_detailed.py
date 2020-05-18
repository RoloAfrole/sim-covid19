from conductor import Conductor
from initializer import Default_Izer, Detailed_Izer, Detailed_TL_Izer, Detailed_TL_GO_Izer
from datetime import datetime
from absl import app
from absl import flags
import util

flags.DEFINE_string('f', '', 'kernel')

# Simulation Parameters
FLAGS = flags.FLAGS

FLAGS.max_size_per_it = 1000000
FLAGS.pool_size = 16

mode = True
# mode = False


def sim(argv):
    if mode:
        # calc(Detailed_Izer, FLAGS,
        #      'det_' + datetime.now().strftime('%Y%m%d%H%M'))
        # calc(Detailed_TL_Izer, FLAGS,
        #      'det_TL_' + datetime.now().strftime('%Y%m%d%H%M'))
        calc(Detailed_TL_GO_Izer, FLAGS,
             'det_TL_GO_' + datetime.now().strftime('%Y%m%d%H%M'))
    else:
        names = '202005171711'
        plots(names)


def calc(izer, config=FLAGS, name=None):
    basename = name
    if name is None:
        basename = datetime.now().strftime('%Y%m%d%H%M')
    initzr = izer(config)
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
