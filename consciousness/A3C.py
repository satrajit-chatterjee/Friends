#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
from keras import backend as Backend
from keras.layers import *
from keras.models import *
import numpy as np
import gym
import time
import threading
import random


# In[2]:


#Constants
L_RATE = 0.01
RUN_TIME = 30
THREAD_DELAY = 0.001
MIN_BATCH = 32
GAMMA = 0.99
N_STEP = 8
THREADS = 8
OPTIMIZERS = 2
GAMMA_N  = GAMMA ** N_STEP
ENTROPY_CONST = 0.01
VALUE_CONST = 0.5
POLICY_LOSS = 1
EP_START = 1
EP_END  = 0.15
EP_STEPS = 75000
ENV_NAME = "CartPole-v0"
NONE_STATE = []


# In[3]:


#Creating the class envirnoment
class Environment(threading.Thread):
    def __init__(self, render = False, eps_start = EP_START, eps_end = EP_END, eps_steps = EP_STEPS):
        threading.Thread.__init__(self) 
        self.agent = Agent(EP_START, EP_END, EP_STEPS)
        self.render = render
        self.env = gym.make(ENV_NAME)
        self.stop_signal = False
        self.count = 0
    
    def run(self):
        while not self.stop_signal:
            self.count += 1
            self.runEpisode()
            
    def runEpisode(self):
        state = self.env.reset()
        Reward = 0
        while True:
            time.sleep(THREAD_DELAY) #To allow concurrency
            action = self.agent.act(state)
            next_state, reward, done, info = self.env.step(action)
            
            if done:
                next_state = None
            
            self.agent.train(state, action, reward, next_state)
            
            next_state = state
            Reward += reward
            
            if done or self.stop_signal:
                break
        
        if self.count % 100 == 0:
            print("This is iteration ", self.count)
            print("Total Reward : ", Reward)
    
    def stop(self):
        self.stop_signal = True


# In[4]:


#Creating an optimizer class
class Optimizer(threading.Thread):
    def __init__(self):
        threading.Thread.__init__(self)
        self.stop_signal = False
        
    def run(self):
        while not self.stop_signal:
            self.g_net.optimize() #Note: g_net is global here 
            
    def stop(self):
        self.stop_signal = True


# In[ ]:


#Creating the network
class A3CNet:
    def __init__(self, env_name, name = "Skynet"):
        #Creating a training queue
        self.train_queue = [[], [], [], [], []] #In the form of [state, action, reward, next_state]
        self.lock_queue = threading.Lock()
        
        #Creating the session
        self.session = tf.Session()
        Backend.set_session(self.session)
        Backend.manual_variable_initialization(True)
        
        #Setting up the environment variables
        self.env = gym.make(env_name)
        self.state_size = self.env.observation_space.shape[0]
        self.action_size = self.env.action_space.n
        NONE_STATE = np.zeros(self.state_size)
            
        #Building the graph
        self.model = self._build_model(policy_layers = [16], value_layers = [8])
        self.graph = self._build_graph(self.model) 
        self.default_graph = tf.get_default_graph()
        #self.default_graph.finalize()
        
        #Initializing all the variables
        self.init = tf.global_variables_initializer()
        self.session.run(self.init)
        
    def _build_model(self, policy_layers = [64, 24, 8], value_layers = [24, 8]):
        input_state = Input(batch_shape = (None, self.state_size))
        
        #Creating the policy network
        for i in range(len(policy_layers)):
            if i == 0:
                middle_layer_p = Dense(policy_layers[i], activation = "relu")(input_state)
            else:
                middle_layer_p = Dense(policy_layers[i], activation = "relu")(middle_layer_p)
        
        actions = Dense(self.action_size, activation = "softmax")(middle_layer_p)
        
        #Creating the value function
        for i in range(len(value_layers)):
            if i == 0:
                middle_layer_v = Dense(value_layers[i], activation = "relu")(input_state)
            else:
                middle_layer_v = Dense(value_layers[i], activation = "relu")(middle_layer_v)

        value = Dense(1, activation = "linear")(middle_layer_v)
        
        model = Model(inputs = [input_state], outputs = [actions, value])
        model._make_predict_function()
        return model
    
    def _build_graph(self, model):
        state_t  = tf.placeholder(dtype = tf.float32, shape = [None, self.state_size])
        action_t = tf.placeholder(dtype = tf.float32, shape = [None, self.action_size])
        reward_t = tf.placeholder(dtype = tf.float32, shape = [None, 1]) #Discounted n step reward
        
        action_preds, value = model(state_t)
        
        #Formula for policy gradient = -log(action_preds * actions)
        neg_log_prob = tf.log(tf.reduce_sum(-1 * action_preds * action_t + 1e-10, axis = 1, keep_dims = True))
        
        #Formula for advantage function -> Reward(s, a) - V(s)
        advantage = reward_t - value
        
        #Finding the policy loss
        #Back prop into the advantage function is stopped here as the actor is being trained here, not critic 
        policy_loss = neg_log_prob * tf.stop_gradient(advantage)
        
        #The value loss equals the advantage function squared
        value_loss = VALUE_CONST * tf.square(advantage)
        
        #Finding the entropy of the policy values to maximize exploration
        entropy = ENTROPY_CONST * -1 * tf.reduce_sum(policy_loss * tf.log(policy_loss + 1e-10), axis = 1, keep_dims = True)
        
        #Finding the total loss
        loss_total = tf.reduce_mean(entropy + policy_loss + value_loss)
        optimizer = tf.train.AdamOptimizer(learning_rate = L_RATE)
        minimize = optimizer.minimize(loss_total)
        
        return state_t, action_t, reward_t, minimize 
    
    def optimize(self):
        if len(self.training_queue[0]) < MIN_BATCH:
            time.sleep(0)
            return
        
        #Incase some threads pass without waiting
        with self.lock_queue:
            if len(self.training_queue[0]) < MIN_BATCH: 
                return
            
            states, actions, rewards, next_states = self.training_queue
            self.training_queue = [[], [], [], []]
        
        #Stacking the input data
        states = np.vstack(states)
        actions = np.vstack(actions)
        rewards = np.vstack(rewards)
        next_states = np.vstack(next_states)
    
        
        values = self.predict_v(next_states)
        rewards = rewards + GAMMA_N * values * next_states
        state_t, action_t, reward_t, minimize = self.graph
        self.sess.run([minimize], feed_dict = {
            state_t: states, 
            action_t: actions, 
            rewards_t: rewards
        })
    
    def train_push(self, state, action, reward, next_state):
        with self.lock_queue:
            self.train_queue[0].append(state)
            self.train_queue[1].append(action)
            self.train_queue[2].append(reward)
            if next_state is None:
                self.train_queue[3].append(NONE_STATE)
                self.train_queue[4].append(0)
            else:
                self.train_queue[3].append(next_state)
                self.train_queue[4].append(1)
    
    def predict(self, state):
        with self.default_graph.as_default():
            predictions, values = self.model.predict(state)
            return predictions, values
        
    def predict_p(self, state):
        with self.default_graph.as_default():
            predictions, _ = self.model.predict(state)
            return predictions
    
    def predict_v(self, state):
        with self.default_graph.as_default():
            _, values = self.model.predict(state)
            return values


# In[ ]:


#Creating a class for agent
class Agent:
    def __init__(self, eps_start, eps_end, eps_steps):
        self.eps_start = eps_start
        self.eps_end   = eps_end
        self.eps_steps = eps_steps

        self.memory = [] # used for n_step return
        self.R = 0.

    def getEpsilon(self):
        if(frames >= self.eps_steps):
            return self.eps_end
        else:
            return self.eps_start + frames * (self.eps_end - self.eps_start) / self.eps_steps# linearly interpolate

    def act(self, s):
        eps = self.getEpsilon()
        global frames; frames = frames + 1

        if random.random() < eps:
            return random.randint(0, NUM_ACTIONS-1)

        else:
            s = np.array([s])
            p = brain.predict_p(s)[0]

            # a = np.argmax(p)
            a = np.random.choice(NUM_ACTIONS, p=p)

            return a

    def train(self, s, a, r, s_):
        def get_sample(memory, n):
            s, a, _, _  = memory[0]
            _, _, _, s_ = memory[n-1]

            return s, a, self.R, s_

        a_cats = np.zeros(NUM_ACTIONS)# turn action into one-hot representation
        a_cats[a] = 1 

        self.memory.append( (s, a_cats, r, s_) )

        self.R = ( self.R + r * GAMMA_N ) / GAMMA

        if s_ is None:
            while len(self.memory) > 0:
                n = len(self.memory)
                s, a, r, s_ = get_sample(self.memory, n)
                brain.train_push(s, a, r, s_)

                self.R = ( self.R - self.memory[0][2] ) / GAMMA
                self.memory.pop(0)

        self.R = 0

        if len(self.memory) >= N_STEP:
            s, a, r, s_ = get_sample(self.memory, N_STEP)
            brain.train_push(s, a, r, s_)

            self.R = self.R - self.memory[0][2]
            self.memory.pop(0)


# In[ ]:


#Writing the main game code 
frames = 0
#-- main
env_test = Environment(render=True, eps_start=0., eps_end=0.)
NUM_STATE = env_test.env.observation_space.shape[0]
NUM_ACTIONS = env_test.env.action_space.n
NONE_STATE = np.zeros(NUM_STATE)

brain = A3CNet(env_name = "CartPole-v0")# brain is global in A3C

envs = [Environment() for i in range(THREADS)]
opts = [Optimizer() for i in range(OPTIMIZERS)]

for o in opts:
    o.start()

for e in envs:
    e.start()

time.sleep(RUN_TIME)

for e in envs:
    e.stop()
for e in envs:
    e.join()

for o in opts:
    o.stop()
for o in opts:
    o.join()

print("Training finished")
env_test.run()


# In[ ]:




