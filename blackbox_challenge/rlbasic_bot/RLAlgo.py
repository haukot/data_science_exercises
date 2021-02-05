from Approximator import Approximator
from RegressionAgent import RegressionAgent
import math
from numpy import *

def e_greedy_selection(fun, actions_list, epsilon):
    #selects an action using Epsilon-greedy strategy
    if (random.rand()>epsilon):
        a = fun()
        # a = self.GetBestAction(s)
    else:
        # selects a random action based on a uniform distribution
        a = random.randint(0,len(actions_list))
    return a

def GetBestAction(s, actions_list, get_Q):
    best = [];
    max = float('-inf');
    for action in actions_list:
        estimate = get_Q(s, action)
        if estimate >= max:
            max = estimate
            best.append(action)

    if len(best) == 0:
        print max, get_Q(s, 0)
    idx = random.randint(0, len(best))
    return [best[idx], max]

def normalize(value, mean, max_to_min):
    return (value - mean) / max_to_min

def basis(i):
    # mean, min, max
    info = [
        [-5.08459598063e-07 , -0.776983 , 11.335174],
        [9.57765950199e-07 , -1.414294 , 1.572714],
        [-7.22122132531e-07 , -1.306795 , 1.484734],
        [2.33570520865e-08 , -2.089417 , 2.090534],
        [-8.9839877696e-09 , -2.360041 , 2.359522],
        [-7.01993716533e-07 , -0.656761 , 6.499477],
        [-4.1454152429e-07 , -0.660002 , 8.538275],
        [-3.59242902979e-07 , -0.610312 , 4.636621],
        [-2.19618243059e-07 , -0.615479 , 4.639163],
        [-4.45759002292e-07 , -0.806575 , 1.698915],
        [-6.85536442991e-07 , -0.812909 , 1.654963],
        [-1.68346818077e-06 , -2.610441 , 3.674306],
        [-1.42240309291e-07 , -2.611374 , 3.335361],
        [-2.02409316299e-06 , -7.275107 , 4.326519],
        [-1.33645535565e-06 , -7.113534 , 3.692792],
        [-1.58942570836e-06 , -15.129109 , 4.965273],
        [-1.56764382608e-06 , -15.067246 , 4.042665],
        [-3.71856906964e-07 , -16.516737 , 5.657391],
        [-2.25219716892e-06 , -22.059725 , 4.346263],
        [-5.98374302831e-07 , -7.095282 , 4.926612],
        [1.84282475446e-07 , -24.740515 , 4.050561],
        [8.25402173614e-07 , -5.190787 , 3.917299],
        [1.57904528178e-06 , -6.979709 , 3.348381],
        [-1.31849255117e-08 , -2.244521 , 3.549175],
        [3.21500247244e-08 , -2.246065 , 3.490342],
        [-5.48867262145e-07 , -3.938857 , 3.709533],
        [-8.08558947487e-08 , -4.04109 , 3.746598],
        [3.74582337796e-07 , -5.043235 , 3.325117],
        [-9.13219792753e-08 , -5.156866 , 3.360399],
        [2.48902032791e-08 , -6.032743 , 2.920661],
        [5.32653102878e-07 , -6.190408 , 2.938002],
        [7.13877599975e-08 , -6.694309 , 2.369039],
        [5.65649556994e-07 , -7.100039 , 2.432636],
        [2.79079165593e-07 , -5.329312 , 1.447598],
        [-1.65146966598e-07 , -6.741731 , 1.72552],
        [0.0367171019371 , -1.1 , 1.1]
    ]
    mean = info[i][0]
    min = info[i][1]
    max = info[i][2]

    return lambda s: normalize(s[i], mean, max - min)


def QEpisode(getInitialState, doAction, getReward, actions_list,
             maxsteps=100, gamma=0.1, epsilon=0.1, alpha=0.1):
    # do one episode with sarsa learning
    # maxstepts: the maximum number of steps per episode
    # Q: the current QTable
    # alpha: the current learning rate
    # gamma: the current discount factor
    # epsilon: probablity of a random action
    # statelist: the list of states
    # actionlist: the list of actions

    s                = getInitialState()
    steps            = 0
    counter = 0
    total_reward     = 0
    r                = 0

    bases = [(lambda i: basis(i))(i) for i in range(0,len(s))]
    apprxs = [Approximator(bases) for a in actions_list]

    get_Q = lambda s, a: apprxs[a].estimate(s)
    set_Q = lambda s, a, Qcost, rate: apprxs[a].update(s, Qcost, rate)

    # selects an action using the epsilon greedy selection strategy
    selectAction = lambda s: (lambda: GetBestAction(s, actions_list, get_Q)[0])
    a   = e_greedy_selection(selectAction(s), actions_list, epsilon)
    start_epsilon = epsilon

    reg_agent = RegressionAgent(4, 36)

    for i in range(1,maxsteps+1):
        # do the selected action and get the next car state
        isfinal,sp     = doAction(a)

        # observe the reward at state xp and the final state flag
        new_total    = getReward()
        r = new_total - total_reward
        total_reward = new_total

        next_a, next_max_Q = GetBestAction(sp, actions_list, get_Q)

        # select action prime
        ap = -1
        if steps > 200000:
            ap = e_greedy_selection(selectAction(s), actions_list, epsilon)
        else:
            ap = reg_agent.get_action_by_state(sp)

        old_Q = get_Q(s, a)
        difference = r + gamma * next_max_Q * (not isfinal) - old_Q

        rate = 0.6
        epsilon = 0.05
        set_Q(s, a, difference, rate)

        #update the current variables
        s = sp
        a = ap

        #increment the step counter.
        steps = steps + 1
        if steps / 1000 > counter:
            counter = steps / 1000
            print steps, total_reward, rate, epsilon
            # print apprxs[0].get()

        # if reachs the goal breaks the episode
        if isfinal==True:
            break

    return total_reward,steps

def SARSAEpisode(getInitialState, doAction, getReward, actions_list,
                 maxsteps=100, gamma=0.1, epsilon=0.1):
    # do one episode with sarsa learning
    # maxstepts: the maximum number of steps per episode
    # Q: the current QTable
    # alpha: the current learning rate
    # gamma: the current discount factor
    # epsilon: probablity of a random action
    # statelist: the list of states
    # actionlist: the list of actions

    s                = getInitialState()
    steps            = 0
    counter = 0
    total_reward     = 0
    r                = 0

    bases = [(lambda i: (lambda x: x[i]))(i) for i in range(0,len(s))]
    apprxs = [Approximator(bases) for a in actions_list]

    get_Q = lambda s, a: apprxs[a].estimate(s)
    set_Q = lambda s, a, Qcost: apprxs[a].update(s, Qcost)

    # selects an action using the epsilon greedy selection strategy
    selectAction = lambda: GetBestAction(s, actions_list, get_Q)
    a   = e_greedy_selection(selectAction, actions_list, epsilon)

    for i in range(1,maxsteps+1):
        # do the selected action and get the next car state
        isfinal,sp     = doAction(a)

        # observe the reward at state xp and the final state flag
        new_total    = getReward()
        r = new_total - total_reward
        total_reward = new_total

        # select action prime
        ap   = e_greedy_selection(selectAction, actions_list, epsilon)

        old_Q = get_Q(s, a)
        new_Q = r*1000 + gamma * get_Q(sp, ap) * (not isfinal) - old_Q
        if math.isnan(new_Q):
            print new_Q, r, old_Q
        set_Q(s, a, new_Q)

        #update the current variables
        s = sp
        a = ap

        #increment the step counter.
        steps = steps + 1
        if steps / 1000 > counter:
            counter = steps / 1000
            print total_reward

        # if reachs the goal breaks the episode
        if isfinal==True:
            break

    return total_reward,steps
