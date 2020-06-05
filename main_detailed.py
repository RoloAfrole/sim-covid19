from conductor import Conductor
from initializer import Detailed_Izer, Detailed_TL_Izer, Detailed_TL_GO_Izer
from initializer import Detailed_I10_Izer, Detailed_I10_TL_Izer, Detailed_I10_TL_GO_Izer
from initializer import Izer_kanto, Izer_TL_kanto, Izer_TL_GO_kanto, Izer_TL_GO_LI_kanto
from initializer import Izer_TL_GO_LI_kanto_with_dist

from datetime import datetime
from absl import app
from absl import flags
import util

flags.DEFINE_string('f', '', 'kernel')

# Simulation Parameters
FLAGS = flags.FLAGS

FLAGS.max_size_per_it = 1000000
FLAGS.pool_size = 14

FLAGS.dist_day = '2020/05/24'
FLAGS.dist_file = 'base_4-7_5-24_w4576'
# FLAGS.dist_day = '2020/04/06'
# FLAGS.dist_file = 'base_3-1_4-6_w1167'

mode = True
# mode = False


def sim(argv):
    if mode:
        # calc(Detailed_Izer, FLAGS,
        #      'det_' + datetime.now().strftime('%Y%m%d%H%M'))
        # calc(Detailed_TL_Izer, FLAGS,
        #      'det_TL_' + datetime.now().strftime('%Y%m%d%H%M'))
        # calc(Detailed_TL_GO_Izer, FLAGS,
        #      'det_TL_GO_' + datetime.now().strftime('%Y%m%d%H%M'))
        # calc(Detailed_I10_Izer, FLAGS,
        #      'det_I10_' + datetime.now().strftime('%Y%m%d%H%M'))
        # calc(Detailed_I10_TL_Izer, FLAGS,
        #      'det_I10_TL_' + datetime.now().strftime('%Y%m%d%H%M'))
        # calc(Detailed_I10_TL_GO_Izer, FLAGS,
        #      'det_I10_TL_GO_' + datetime.now().strftime('%Y%m%d%H%M'))
        # calc(Izer_kanto, FLAGS,
        #      'kanto_' + datetime.now().strftime('%Y%m%d%H%M'))
        # calc(Izer_TL_kanto, FLAGS,
        #      'kanto_TL_' + datetime.now().strftime('%Y%m%d%H%M'))
        # calc(Izer_TL_GO_kanto, FLAGS,
        #      'kanto_TL_GO_' + datetime.now().strftime('%Y%m%d%H%M'))
        # calc(Izer_TL_GO_LI_kanto, FLAGS,
        #      'kanto_TL_GO_LI_for1200_' + datetime.now().strftime('%Y%m%d%H%M'))
        # calc(Izer_TL_GO_LI_kanto_with_dist, FLAGS,
        #      'kanto_after_' + datetime.now().strftime('%Y%m%d%H%M'))
        calc(Izer_TL_GO_LI_kanto_with_dist, FLAGS,
             'kanto_after_SOE_start_5-24_w4576_C2_' + datetime.now().strftime('%Y%m%d%H%M'))
    else:
        names = 'kanto_after_SOE_start_5-24_w4576_C1_202006051141'
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
                      save=False,
                      title=None,
                      ylimit=[0, 6000],
                      xlimit=[0, 50]
                      )
    # util.save_as_csv(status, basename)


if __name__ == '__main__':
    try:
        app.run(sim)
    except SystemExit:
        pass
