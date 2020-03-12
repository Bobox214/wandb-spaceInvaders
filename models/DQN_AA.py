import numpy as np
from collections import deque
import keras
import random
import tensorflow as tf
import logging
from helpers import *

class Model:
    """
        Implements the first DQN model published in Nature.
    """
    def __init__(self,env,config,eval):
        self.env    = env
        self.eval   = eval
        self.config = config
        self.memory = ExperienceBuffer(size=config.experience_buffer_size)
        self.action_size = env.action_space.n
        self.learning_rate = config.learning_rate
        self.epsilon_start = config.epsilon_start if not eval else 0
        self.epsilon_end   = config.epsilon_end   if not eval else 0
        self.epsilon       = config.epsilon_start if not eval else 0
        self.modelCopyRate = 1000
        self.gamma = 0.99
        self.model        = self._build_model()
        self.target_model = self._build_model()
        self.loss = 0
        self.predictQ = 0
        self.nextCopy = self.modelCopyRate

    def _build_model(self):
        X = I = keras.layers.Input(self.env.observation_space.shape, name='frames')
        X = keras.layers.Lambda(lambda x: x / 255.0)(I)
        X = keras.layers.Conv2D(32,kernel_size=8,strides=4, activation='relu')(X)
        X = keras.layers.Conv2D(64,kernel_size=4,strides=2, activation='relu')(X)
        X = keras.layers.Conv2D(64,kernel_size=3,strides=1, activation='relu')(X)
        X = keras.layers.Flatten()(X)
        X = keras.layers.Dense(512, activation='relu')(X)
        X = keras.layers.Dense(self.action_size)(X)
        O = X

        model = keras.models.Model(inputs=[I], outputs=O)
        model.compile(loss='mse',optimizer=keras.optimizers.Adam(lr=self.config.learning_rate))

        return model

    def inlineInfo(self):
        return f"Epsilon: {self.epsilon:0.2f} Loss: {self.loss:.2f} Q: {self.predictQ:.2f}"

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train(self,frameIdx):
        if self.eval or len(self.memory) < self.config.min_experience_size:
            return
        states,actions,rewards,next_states,dones = self.memory.sample(self.config.batch_size)

        targets = self.model.predict(states)
        next_Q = self.target_model.predict(next_states)
        next_Q[dones==1] = 0
        targets[range(self.config.batch_size),actions] = rewards + self.gamma*np.max(next_Q,axis=1) 
        self.predictQ = np.mean(targets)

        self.loss = self.model.train_on_batch(states,targets) 

        self.epsilon = max(self.epsilon_end,self.epsilon_start-frameIdx/self.config.epsilon_decay_last_frame)

        if frameIdx>self.nextCopy:
            self.target_model.set_weights(self.model.get_weights())
            self.nextCopy += self.modelCopyRate

    def act(self, state):
        if np.random.random() < self.epsilon:
            return self.env.action_space.sample()
        else:
            np_state = np.expand_dims(state,0)
            return np.argmax(self.model.predict(np_state)[0])

    def play(self,state):
        np_state = np.expand_dims(state,0)
        Qs = self.model.predict(np_state)[0]
        action = np.argmax(Qs)
        return action,Qs

    def save(self, name):
        self.model.save(name)

    def load(self, name):
        self.model.load_weights(name)
        self.target_model.load_weights(name)
