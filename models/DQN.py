import numpy as np
from collections import deque
import keras
import random
import tensorflow as tf
import logging

class Model:
    """
        Implements the first DQN model published in Nature.
        ref: 100 ep 1m33 855f/s
    """
    def __init__(self,env,config,eval):
        self.env    = env
        self.eval   = eval
        self.config = config
        self.memory = deque(maxlen=config.experience_buffer_size)
        self.action_size = env.action_space.n
        self.learning_rate = config.learning_rate
        self.epsilon_endFrame = 1000000
        self.epsilon_min = 0.1
        self.epsilon = 1.0 if not eval else self.epsilon_min
        self.modelCopyRate = 10000
        self.gamma = 0.99
        self.model        = self._build_model()
        self.target_model = self._build_model()
        self.loss = 0
        self.predictQ = 0
        self.nextCopy = self.modelCopyRate

    def _build_model(self):
        I = keras.layers.Input(self.env.observation_space.shape, name='frames')
        X = keras.layers.Lambda(lambda x: x / 255.0)(I)
        X = keras.layers.Conv2D(32,kernel_size=8,strides=4, activation='relu')(X)
        X = keras.layers.Conv2D(64,kernel_size=4,strides=2, activation='relu')(X)
        X = keras.layers.Conv2D(64,kernel_size=3,strides=1, activation='relu')(X)
        X = keras.layers.Flatten()(X)
        X = keras.layers.Dense(512, activation='relu')(X)
        X = keras.layers.Dense(self.action_size)(X)

        J = keras.layers.Input((self.action_size,), name='mask')

        O = keras.layers.Multiply()([X,J])

        model = keras.models.Model(inputs=[I,J], outputs=O)
        model.compile(loss='mse',optimizer=keras.optimizers.RMSprop(lr=0.00025, rho=0.95, epsilon=0.01))

        return model

    def inlineInfo(self):
        return f"Epsilon: {self.epsilon:0.2f} Loss: {self.loss:.2f} Q: {self.predictQ:.2f}"

    def remember(self, state, action, reward, next_state, done):
        #logging.info(f"APP {action} -> {reward} : {done}")
        oneHotAction = [(i==action) for i in range(self.action_size)]
        self.memory.append((state, oneHotAction, reward, next_state, done))

    def train(self,frameIdx):
        if self.eval or len(self.memory) < self.config.min_experience_size:
            return
        indices = np.random.choice(len(self.memory),self.config.batch_size,replace=False)
        states,actions,rewards,next_states,dones = map(np.array,zip(*(self.memory[idx] for idx in indices)))

        next_Q = self.target_model.predict([next_states,np.ones(actions.shape)])
        next_Q[dones] = 0
        targets = rewards + self.gamma*np.max(next_Q,axis=1) 
        self.predictQ = np.mean(targets)

        self.loss = self.model.train_on_batch([states,actions],actions*targets[:,None]) 

        if frameIdx<self.epsilon_endFrame:
            self.epsilon = (1-self.epsilon_min)*(self.epsilon_endFrame-frameIdx)/self.epsilon_endFrame+self.epsilon_min

        if frameIdx>self.nextCopy:
            self.target_model.set_weights(self.model.get_weights())
            self.nextCopy += self.modelCopyRate

    def act(self, state):
        if np.random.random() < self.epsilon:
            return self.env.action_space.sample()
        else:
            np_state = np.expand_dims(state,0)
            return np.argmax(self.model.predict([np_state,np.ones((1,self.action_size))])[0])

    def play(self,state):
        np_state = np.expand_dims(state,0)
        Qs = self.model.predict([np_state,np.ones((1,self.action_size))])[0]
        action = np.argmax(Qs)
        return action,Qs

    def save(self, name):
        self.model.save(name)

    def load(self, name):
        self.model.load_weights(name)
        self.target_model.load_weights(name)
