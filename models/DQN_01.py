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
        self.epsilon_decay = 0.99
        self.epsilon_min = 0.02
        self.epsilon = 1.0 if not eval else self.epsilon_min
        self.gamma = 0.95
        self.model        = self._build_model()
        self.target_model = self._build_model()
        self.modelName = 'DQN_01'
        self.loss = 0
        self.predictQ = 0

    def _build_model(self):
        model = keras.models.Sequential()
        model.add(keras.layers.Conv2D(32,kernel_size=8,strides=4, activation='relu', input_shape=self.env.observation_space.shape))
        model.add(keras.layers.Conv2D(64,kernel_size=4,strides=2, activation='relu'))
        model.add(keras.layers.Conv2D(64,kernel_size=3,strides=1, activation='relu'))
        model.add(keras.layers.Flatten())
        model.add(keras.layers.Dense(512, activation='relu'))
        model.add(keras.layers.Dense(self.action_size))
        model.compile(loss='mse', optimizer=keras.optimizers.Nadam(lr=self.learning_rate))
        return model

    def inlineInfo(self):
        return f"Epsilon: {self.epsilon:0.2f} Loss: {self.loss:.2f} Q: {self.predictQ:.2f}"

    def remember(self, state, action, reward, next_state, done):
        #logging.info(f"APP {action} -> {reward} : {done}")
        self.memory.append((state, action, reward, next_state, done))

    def train(self):
        if self.eval or len(self.memory) < self.config.min_experience_size:
            return
        indices = np.random.choice(len(self.memory),self.config.batch_size,replace=False)
        states,actions,rewards,next_states,dones = zip(*(self.memory[idx] for idx in indices))
        states_a = np.array(states)
        targets_a = self.model.predict(states_a)
        self.predictQ = np.mean(Q_futures_a)
        next_states_a = np.array(next_states)
        dones_a = np.array(dones)
        Q_futures_a = np.amax(self.target_model.predict(next_states_a),1)
        Q_futures_a[dones_a] = 0
        for i in range(self.config.batch_size):
            targets_a[i][actions[i]] = rewards[i]+Q_futures_a[i]*self.gamma

        self.loss = self.model.train_on_batch(states_a,targets_a) 
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        TAU = 0.3
        model_weights = self.model.get_weights()
        target_model_weights = self.target_model.get_weights()
        for i in range(len(model_weights)):
            target_model_weights[i] = TAU * model_weights[i] + (1 - TAU) * target_model_weights[i]
        self.target_model.set_weights(target_model_weights)

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
