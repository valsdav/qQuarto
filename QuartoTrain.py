
# coding: utf-8

# In[1]:


import argparse
import random
from collections import deque

import json
import os.path
import time

from keras import initializers
from keras.initializers import normal, identity
from keras.models import model_from_json
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Conv2D
from keras.optimizers import SGD , Adam
import tensorflow as tf

# before training init writer (for tensorboard log) / model
writer = tf.summary.FileWriter("./logs")



# In[2]:


from Quarto import *


# In[3]:
GAMMA = 0.99 # decay rate of past observations
OFFSET = 0  #number of oservation already done
OBSERVATION = 32 + OFFSET # timesteps to observe before training
EXPLORE = 3000000. # frames over which to anneal epsilon
FINAL_EPSILON = 0.0001 # final value of epsilon
INITIAL_EPSILON = 0.1 # starting value of epsilon
REPLAY_MEMORY = 50000 # number of previous transitions to remember
BATCH = 32 # size of minibatch
LEARNING_RATE = 1e-4

#We go to training mode
OBSERVE = OBSERVATION

epsilon = INITIAL_EPSILON
if OFFSET:
    for o in range(OFFSET):
        epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / EXPLORE


# In[4]:


from keras.utils import plot_model

def build_model():
    print("Now we build the model")
    model = Sequential()
    model.add(Conv2D(32, (2,3), strides=(1, 1), input_shape=(4,16,4)))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (2,3), strides=(1, 1)))
    model.add(Activation('relu'))
    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dense(1))

    adam = Adam(lr=LEARNING_RATE)
    model.compile(loss='mse',optimizer=adam)

    plot_model(model,show_shapes=True, to_file='model.png')
    print("We finish building the model")
    return model


model = build_model()
print ("Now we load weight")
if os.path.isfile("model.h5"):
    model.load_weights("model.h5")
    adam = Adam(lr=LEARNING_RATE)
    model.compile(loss='mse',optimizer=adam)
    print ("Weight load successfully")

# In[5]:


def select_actions(status, rand=True):
    #getting the available actions
    actions= status.get_available_actions()
    image = None
    if (len(actions) == 0):
        return [], None
    #choose an action epsilon greedy
    if rand and random.random() <= epsilon:
       # print("----------Random Action----------")
        a_index = random.randrange(len(actions))
        image = status.get_action_image(actions[a_index])
        show_action_image(image)
    else:
        a_images_t = status.get_action_images(actions)
        a_index = np.argmax(model.predict_on_batch(a_images_t))
        image = a_images_t[a_index]
    return actions[a_index] , image


# In[ ]:

losses = []
replay_memory = deque()
#Initial status
s_t0 = get_initial_status()
a_t0,im_t0 = select_actions(s_t0)
t = OFFSET
while(t< 150):
    loss = 0
    Q_sa = 0
    Q_sa_target = 0
    r_t = 0
    deltat_train = 0
    deltat_prepare_train = 0
    n_images_train = 0

    #We reduced the epsilon gradually
    if epsilon > FINAL_EPSILON and t > OBSERVE:
        epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / EXPLORE

    # Now we need to get the answer action and final status
    s_t1, win, win_r = s_t0.get_new_status(a_t0)
    if win:
        #Increasing of 100 the reward
        r_t = win_r +100
        #saving win status
        replay_memory.append((s_t0, a_t0, im_t0, r_t, s_t1, win))
        #restarting cleaning status
        s_t0 = get_initial_status()
        a_t0, im_t0 = select_actions(s_t0)
        #print("WIN! Reward: {}".format(win_r))
        #show(s_t1[0])

    else:
        #continue with the answer from enemy
        a_t1, im_t1 = select_actions(s_t1)

        if(len(a_t1) == 0):
            #print("No more moves")
            # reset the game because we are stucks
            s_t0 = get_initial_status()
            a_t0, im_t0 = select_actions(s_t0)
            continue

        #get final state
        s_t2, win_e, win_e_r = s_t1.get_new_status(a_t1)
        r_t = s_t0.get_transition_reward(a_t0, a_t1)
        if win_e:
            #reward is decreased of 100
            r_t -= 100
            # this will be analyzed in the next turn
            #print("WIN of ENEMY! Reward: {}".format(r_t))

        # we save the state in the replay memory
        replay_memory.append((s_t0, a_t0, im_t0, r_t, s_t2, win))
        # to the next state
        s_t0 = s_t1
        a_t0 = a_t1
        im_t0 = im_t1


    if len(replay_memory) > REPLAY_MEMORY:
            replay_memory.popleft()

    #only train if done observing
    if t > OBSERVE:
        ttrain1 = time.time()
        #sample a minibatch to train on
        minibatch = random.sample(replay_memory, BATCH)
        inputs = np.zeros((BATCH, 4,16,4))
        targets = np.zeros((BATCH))
        #Now we do the experience replay
        for i in range(0, len(minibatch)):
            ttrain_j1 = time.time()
            state_t = minibatch[i][0]
            action_t = minibatch[i][1]
            image_t = minibatch[i][2]
            reward_t = minibatch[i][3]
            state_t1 = minibatch[i][4]
            terminal = minibatch[i][5]
            # Saving the action image of the first action as input
            inputs[i:i + 1] = image_t

            # Getting all the possible action in the final state
            Q_sa_max = 0
            if terminal:
                Q_sa_max = reward_t
                ttrain_j2 = time.time()
                ttrain_j3 = ttrain_j2
            else:
                final_actions= state_t1.get_available_actions()
                if (len(final_actions)==0):
                    #it is the final action
                    Q_sa_max = reward_t
                    ttrain_j2 = time.time()
                    ttrain_j3 = ttrain_j2
                else:
                    n_images_train+=len(final_actions)
                    a_images_t1 = state_t1.get_action_images(final_actions)
                    ttrain_j2 = time.time()
                    Q_sa = np.max(model.predict_on_batch(a_images_t1))
                    ttrain_j3 = time.time()
                    #Qs = [ model.predict(a_im) for a_im in a_images_t1]
                    Q_sa_max = reward_t + GAMMA * Q_sa
            #Saving target
            targets[i:i+1] = Q_sa_max
            Q_sa_target = Q_sa_max
            deltat_prepare_train += ttrain_j2 - ttrain_j1
            deltat_train += ttrain_j3 - ttrain_j2
        #backpropagation training
        ttrain2 = time.time()
        loss += model.train_on_batch(inputs, targets)
        summary = tf.Summary(value=[
                tf.Summary.Value(tag="summary_tag", simple_value=t),])
        writer.add_summary(summary)
        ttrain3 = time.time()
        deltat_train += ttrain3- ttrain2

    #going to the next epoch. The action_t0 and state_t0 are already setted
    t+=1
     # save progress every 10000 iterations
    if t % 1000 == 0:
        print("Now we save model")
        model.save_weights("model.h5", overwrite=True)
        #with open("model.json", "w") as outfile:
        #    json.dump(model.to_json(), outfile)
        plt.plot([i for i in range(len(losses))], losses)
        plt.savefig("losses{}.png".format(t%10000))
    # print info
    if (t> OBSERVE and t % 10 == 0) or (t< OBSERVE and t % 100 == 0):
        state = ""
        if t <= OBSERVE:
            state = "observe"
        elif t > OBSERVE and t <= OBSERVE + EXPLORE:
            state = "explore"
        else:
            state = "train"

        print("Epoch {0:d} \t| State {1} \t| Epsilon {2:>5.4f} \t| N.images {3:>4d} \t| TP {4:>4.2f} \t| TT {5:>4.2f} \t|  Reward {6:>4d} \t| Q_sa {7:>6.3f} \t| Q_sa_target {8:>6.3f} \t| Loss {9:>8.3f}".
             format(t,state,epsilon,n_images_train, deltat_prepare_train,deltat_train, r_t ,Q_sa, Q_sa_target, loss))
        losses.append(loss)
