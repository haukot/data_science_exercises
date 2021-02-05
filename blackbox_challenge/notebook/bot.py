import interface as bbox
from numpy import *


def get_action_by_state(state, verbose=1):
    # if verbose:
        # for i in xrange(n_features):
        #     print "state[%d] = %f" %  (i, state[i])

        # print "score = {}, time = {}".format(bbox.get_score(), bbox.get_time())

    # action_to_do = int((random.rand() * 4))
    action_to_do = 0
    return action_to_do


n_features = n_actions = -1


def prepare_bbox():
    global n_features, n_actions

    if bbox.is_level_loaded():
        bbox.reset_level()
    else:
        bbox.load_level("../levels/train_level.data", verbose=1)
        n_features = bbox.get_num_of_features()
        n_actions = bbox.get_num_of_actions()

def state_str(state):
    line = ""
    for i in xrange(n_features):
        line += "%f," %  (state[i])

    return line


def run_bbox(verbose=False):
    has_next = 1

    prepare_bbox()

    a = ""
    for i in xrange(n_features):
        a += "%i," % i
    a += "%i,reward\n" % (n_features + 1)

    # f = file('one_and_two', 'w')

    steps = 0
    while has_next:
        state = bbox.get_state()
        action = 1 if steps % 2 == 0 else 2 #get_action_by_state(state)
        score = bbox.get_score()
        has_next = bbox.do_action(action)

        action_score = bbox.get_score() - score
        a += state_str(state)
        a += "%f,%f\n" % (action, action_score)
        steps = steps + 1

    # f.write(a)
    print steps
    bbox.finish(verbose=1)


if __name__ == "__main__":
    run_bbox(verbose=0)
