import interface as bbox
import numpy as np

from rltools.FARLBasic import *
from rltools.kNNSCIPY import kNNQ
from rltools.ActionSelection import *
import cPickle

n_features = n_actions = -1

def prepare_bbox():
	global n_features, n_actions

	if bbox.is_level_loaded():
		bbox.reset_level()
	else:
		bbox.load_level("../levels/train_level.data", verbose=1)
		n_features = bbox.get_num_of_features()
		n_actions = bbox.get_num_of_actions()


def run_bbox(verbose=False):
	has_next = 1

	prepare_bbox()
        global Q
        nels = 16
        input_ranges = [[-100, 100] for j in range(0,nels)]
        n_elements = [3 for j in range(0,nels)]
        Q = kNNQ(nactions=n_actions,input_ranges=input_ranges,nelemns=n_elements,npoints=False,k=4,alpha=0.3,lm=0.95)
        SelectAction = e_greedy_selection(epsilon=0.0, Q=Q, nactions=n_actions)
        s                = bbox.get_state()
        r                = 0
        gamma = 1.0
        # selects an action using the epsilon greedy selection strategy
        a,v   = SelectAction(s[:nels])
        steps = 0
        counter = 0

	while has_next:
                # do the selected action and get the next car state
                score = bbox.get_score()
		has_next = bbox.do_action(a)
                state = bbox.get_state()
                # select action prime
                ap,vp     = SelectAction(state[:nels])
                reward = bbox.get_score() - score
                target_value = reward + gamma * vp * (has_next)
                Q.Update(s[:nels],a,target_value)
                s = state
                a = ap
                steps = steps + 1
                if steps / 1000 > counter:
                        counter = steps / 1000
                        print score, a

	bbox.finish(verbose=1)


if __name__ == "__main__":
	run_bbox()
