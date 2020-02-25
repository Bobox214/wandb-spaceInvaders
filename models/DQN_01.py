import numpy as np
from collections import deque
import keras
import random
import tensorflow as tf

class Model:
    """
        Implements the first DQN model published in Nature.
    """
    def __init__(self,env,config,eval):
        self.env    = env
        self.eval   = eval
        self.config = config
        self.memory = deque(maxlen=20000)
        self.action_size = env.action_space.n
        self.learning_rate = config.learning_rate
        self.epsilon_decay = 0.99
        self.epsilon_min = 0.02
        self.epsilon = 1.0 if not eval else self.epsilon_min
        self.gamma = 0.95
        self.model = self._build_model()
        self.modelName = 'DQN_01'

    def _build_model(self):
        inputDim = self.env.observation_space.shape[1]
        model = keras.models.Sequential()
        model.add(keras.layers.Conv2D(32,kernel_size=8,strides=4, activation='relu'))
        model.add(keras.layers.Conv2D(64,kernel_size=4,strides=2, activation='relu'))
        model.add(keras.layers.Conv2D(64,kernel_size=3,strides=1, activation='relu'))
        model.add(keras.layers.Flatten())
        model.add(keras.layers.Dense(512, activation='relu'))
        model.add(keras.layers.Dense(self.action_size))
        model.compile(loss='mse', optimizer=keras.optimizers.Nadam(lr=self.learning_rate))
        return model

    def inlineInfo(self):
        return f"Epsilon: {self.epsilon:0.2f}"

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train(self):
        if self.eval or len(self.memory) < self.config.batch_size:
            return
        minibatch = random.sample(self.memory, self.config.batch_size)
        for state, action, reward, next_state, done in minibatch:
            np_state      = np.expand_dims(state,0)
            np_next_state = np.expand_dims(next_state,0)
            target = self.model.predict(np_state)
            if done:
                target[0][action] = reward
            else:
                Q_future = max(self.model.predict(np_next_state)[0])
                target[0][action] = reward+Q_future*self.gamma
            self.model.fit(np_state,target,epochs=1,verbose=0) 
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def act(self, state):
        if np.random.random() < self.epsilon:
            return self.env.action_space.sample()
        else:
            np_state = np.expand_dims(state,0)
            return np.argmax(self.model.predict(np_state)[0])

    def play(self,state):
        np_state = np.expand_dims(state,0)
        return np.argmax(self.model.predict(np_state)[0])

    def save(self, name):
        self.model.save(name)

    def load(self, name):
        self.model.load_weights(name)
