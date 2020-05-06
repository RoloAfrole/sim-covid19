from conductor import Conductor
from initializer import Default_Izer
from datetime import datetime
from absl import app
from absl import flags

flags.DEFINE_string('f', '', 'kernel')

# Simulation Parameters
FLAGS = flags.FLAGS

mode = False


def sim(argv):
    if mode:
        calc(FLAGS)
    else:
        # plots(names)
        pass


def calc(config=FLAGS):
    # basename = datetime.now().strftime('%Y%m%d%H%M')
    initzr = Default_Izer(config)
    conductor = Conductor(config, initzr)
    conductor.sim()
    # util.plot_sim(results, basename)
    # util.save_status(results, basename)


if __name__ == '__main__':
    app.run(sim)
