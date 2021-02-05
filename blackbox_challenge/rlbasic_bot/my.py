# -*- coding: cp1252 -*-
import numpy as np
from RLBasic import *

class CartPole(RLBase):
    def BuildActionList(self):
        return np.array([0,1,2,3])

    def BuildStateList(self):
        # this is typically a combinatorial discretization of the input space

        # x = arange(-4,4,0.1)
        # I=size(x)
        # states =[]
        # index=0
        # for i in range(I):
        #     for j in range(J):
        #         for k in range(K):
        #             for l in range(L):
        #                 states.append([x1[i],x2[j],x3[k],x4[l]])

        # return array(states)
        return []


    def GetReward(self, s ):
        action_score = bbox.get_score() - score
        return action_score

    def DoAction(self, action, x ):
        score = bbox.get_score()
        bbox.do_action(action)
        return bbox.get_state()

    def GetInitialState(self):
        return  bbox.get_state()

def CartPoleDemo(bbox_x):
    global bbox, score;
    bbox = bbox_x
    CP  = CartPole(0.3,1.0,0.001)
    maxsteps = 100000
    grafica  = False


    for i in range(maxsteps):

        total_reward,steps  = CP.SARSAEpisode( maxsteps, grafica )
        #total_reward,steps  = CP.QLearningEpisode( maxsteps, grafica )

        # print 'Espisode: ',i,'  Steps:',steps,'  Reward:',str(total_reward),' epsilon: ',str(CP.epsilon)

        CP.epsilon = CP.epsilon * 0.99
    print("GOOD", bbox.get_score())
