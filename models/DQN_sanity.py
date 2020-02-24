import numpy as np
from collections import deque
import keras
import random

class Model:
    def __init__(self,env,config,eval):
        self.env    = env
        self.eval   = eval
        self.config = config
        self.memory = deque(maxlen=20000)
        self.action_size = env.action_space.n
        self.learning_rate = config.learning_rate
        self.epsilon_decay = 0.999
        self.epsilon_min = 0.02
        self.epsilon = 1.0 if not eval else epsilon_min
        self.gamma = 0.95
        self.model = self._build_model()
        self.modelName = 'DQN_sanity'

    def _build_model(self):
        inputDim = self.env.observation_space.shape[1]
        model = keras.models.Sequential()
        model.add(keras.layers.Dense(8, activation='relu', input_dim=inputDim))
        model.add(keras.layers.Dense(16, activation='relu'))
        model.add(keras.layers.Dense(self.action_size, activation='linear'))
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
            target = self.model.predict(state)
            if done:
                target[0][action] = reward
            else:
                Q_future = max(self.model.predict(next_state)[0])
                target[0][action] = reward+Q_future*self.gamma
            self.model.fit(state,target,epochs=1,verbose=0) 
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def act(self, state):
        if np.random.random() < self.epsilon:
            return self.env.action_space.sample()
        else:
            return np.argmax(self.model.predict(state)[0])

    def play(self,state):
        return np.argmax(self.model.predict(state)[0])

    def save(self, name):
        self.model.save(name)

    def load(self, name):
        self.model.load_weights(name)
