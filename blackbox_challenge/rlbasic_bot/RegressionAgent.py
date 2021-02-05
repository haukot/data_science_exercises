import numpy as np

class RegressionAgent:
    def __init__(self, n_actions, n_features):
	coefs = np.loadtxt("reg_coefs.txt").reshape(n_actions, n_features + 1)
	self.reg_coefs = coefs[:,:-1]
	self.free_coefs = coefs[:,-1]
        self.n_actions = n_actions

    def get_action_by_state(self, state):
	best_act = -1
	best_val = -1e9

	for act in xrange(self.n_actions):
		val = self.calc_reg_for_action(act, state)
		if val > best_val:
			best_val = val
			best_act = act

	return best_act

    def calc_reg_for_action(self, action, state):
	return np.dot(self.reg_coefs[action], state) + self.free_coefs[action]
