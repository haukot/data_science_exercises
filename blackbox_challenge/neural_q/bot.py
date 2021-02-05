import interface as bbox
import numpy as np

import random

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.advanced_activations import LeakyReLU, ELU
from keras.optimizers import RMSprop

from keras.callbacks import TensorBoard


from datetime import datetime

n_features = n_actions = -1


def prepare_bbox():
	global n_features, n_actions

	if bbox.is_level_loaded():
		bbox.reset_level()
	else:
		bbox.load_level("../levels/train_level.data", verbose=0)
		n_features = bbox.get_num_of_features()
		n_actions = bbox.get_num_of_actions()

def run_bbox(verbose=False):
        model = Sequential()
        model.add(Dense(164, init='lecun_uniform', input_shape=(56,)))
        # model.add(Activation('relu'))
        model.add(ELU(alpha=0.01))
        #model.add(Dropout(0.2)) I'm not using dropout, but maybe you wanna give it a try?

        model.add(Dense(150, init='lecun_uniform'))
        # model.add(Activation('relu'))
        model.add(ELU(alpha=0.01))
        #model.add(Dropout(0.2))

        model.add(Dense(4, init='lecun_uniform'))
        model.add(Activation('linear')) #linear output so we can have range of real-valued outputs

        rms = RMSprop()
        model.compile(loss='mse', optimizer=rms)

        # model.load_weights('latest_weights')
        gamma = 0.975
        epsilon = 0.0
        batchSize = 64
        buffer = 128
        replay = []
        #stores tuples of (S, A, R, S')
        h = 0
        steps = 0
        has_next = 1
        filepath = datetime.now().strftime("%d.%m.%y_%I:%M") + "_weights_steps_" + str(steps)

        last_actions = [0 for i in range(0,20)]

	prepare_bbox()
        state = bbox.get_state()
        has_next = 1

        all_state = np.append(state, last_actions).reshape(1,56)
        #while game still in progress
        while(has_next):
            #We are in state S
            #Let's run our Q function on S to get Q values for all possible actions
            qval = model.predict(all_state, batch_size=1)
            if (random.random() < epsilon): #choose random action
                action = np.random.randint(0,4)
            else: #choose best action from Q(s,a) values
                action = (np.argmax(qval))
            #Take action, observe new state S'

            score = bbox.get_score()
            has_next = bbox.do_action(action)
            new_state = bbox.get_state()
            #Observe reward
            reward = bbox.get_score() - score
            # save old actions as state
            last_actions_old = list(last_actions)
            last_actions.pop()
            last_actions.insert(0, action)

            steps = steps + 1
            if steps % 500 == 0:
                    print "SCORE", score, steps, epsilon
            if steps % 20000 == 0:
                    filepath = datetime.now().strftime("%d.%m.%y_%I:%M") + "_weights_steps_" + str(steps)
                    model.save_weights(filepath, overwrite=False)
                    print "save"


            all_state_old = np.append(state, last_actions_old).reshape(1,56)
            all_state = np.append(new_state, last_actions).reshape(1,56)
            #Experience replay storage
            if (len(replay) < buffer): #if buffer not filled, add to it
                replay.append((all_state_old, action, reward, all_state))
            else: #if buffer full, overwrite old values
                if (h < (buffer-1)):
                    h += 1
                else:
                    h = 0
                replay[h] = (all_state_old, action, reward, all_state)
                #randomly sample our experience replay memory
                minibatch = random.sample(replay, batchSize)
                X_train = []
                y_train = []
                for memory in minibatch:
                    #Get max_Q(S',a)
                    old_state, action, reward, new_mem_state = memory
                    old_qval = model.predict(old_state, batch_size=1)
                    newQ = model.predict(new_mem_state, batch_size=1)
                    maxQ = np.max(newQ)
                    y = np.zeros((1,4))
                    y[:] = old_qval[:]
                    if reward == -1: #non-terminal state
                        update = (reward + (gamma * maxQ))
                    else: #terminal state
                        update = reward
                    y[0][action] = update
                    X_train.append(old_state.reshape(56,))
                    y_train.append(y.reshape(4,))

                X_train = np.array(X_train)
                y_train = np.array(y_train)
                # tensor_board_callback = TensorBoard(log_dir='./logs', histogram_freq=0)
                model.fit(X_train, y_train, batch_size=batchSize, nb_epoch=1, verbose=0, callbacks=[])
            state = new_state
            # clear_output(wait=True)
            if epsilon > 0.1: #decrement epsilon over time
                    epsilon = 1/((steps + 1)/100000.0)

        filepath = datetime.now().strftime("%d.%m.%y_%I:%M") + "_weights"
        model.save_weights(filepath, overwrite=False)
	bbox.finish(verbose=1)


if __name__ == "__main__":
	run_bbox()
