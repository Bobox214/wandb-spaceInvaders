from collections import deque
import keras
import random

class Model:
    def __init__(self, state_size, action_size,config):
        self.config = config
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=20000)
        self.learning_rate = config.learning_rate
        self.epsilon = 1.0
        self.epsilon_decay = 0.9
        self.epsilon_min = 0.05
        self.gamma = 0.95
        self.model = self._build_model()

    def _build_model(self):
        model = keras.models.Sequential()
        model.add(keras.layers.Dense(8, activation='relu', input_dim=self.state_size))
        model.add(keras.layers.Dense(16, activation='relu'))
        model.add(keras.layers.Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=keras.optimizers.Nadam(lr=self.learning_rate))
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train(self):
        if len(self.memory) > self.config.batch_size:
          minibatch = random.sample(self.memory, self.config.batch_size)
          for state, action, reward, next_state, done in minibatch:
              target = reward
          if self.epsilon > self.epsilon_min:
              self.epsilon *= self.epsilon_decay

    def act(self, state):
        return random.randrange(self.action_size)

    def save(self, name):
        self.model.save(name)

    def load(self, name):
        self.model.load_weights(name)
