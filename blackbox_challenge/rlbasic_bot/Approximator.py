import math
# # Regressor - from https://github.com/dabbler0/wigo/blob/master/src/regressor.coffee
# Keeps a running record of thetas; given new input/output maps,
# performs gradient descent linear regression on given basis functions.
class Approximator:
    def __init__(self, bases):
        self.thetas = [0 for basis in bases]
        self.bases = bases

    # Get the predicted output for the given input using
    # basis functions and current thetas.
    #
    # Returns sum(theta[i] * basis[i]).
    def estimate(self, state):
        output = 0
        for i, basis in enumerate(self.bases):
            output += self.thetas[i] * basis(state)
        return output

    def get(self):
        return self.thetas

   # Given an input/output map, do another gradient descent iteration to improve
   # thetas.
    def update(self, state, Qcost, rate):
        for i, basis in enumerate(self.bases):
            # penalty = (gradient + self.alpha * self.thetas[i] / len(self.thetas))
            # self.thetas[i] -= self.rate * penalty * basis(state)
            self.thetas[i] += rate * Qcost * basis(state) # wi <- wi + nbFi(s,a)
