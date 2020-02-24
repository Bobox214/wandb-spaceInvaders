import gym
from gym import logger as gymlogger
from gym.wrappers import Monitor
gymlogger.set_level(30)

import numpy as np
import random
import math
import glob
import io
import os
import cv2
import base64
import tensorflow as tf
import matplotlib
import matplotlib.pyplot as plt
#%matplotlib inline
from collections import deque
from datetime import datetime
import keras

#from IPython.display import HTML
#from IPython import display as ipythondisplay
from pyvirtualdisplay import Display
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf

# import wandb
import wandb

# Preprocessing - crop images, convert them to 1D black and white image tensors
#   Image dimensions - (210, 160, 3)
#   Output dimensions - (88, 80, 1)
color = np.array([210, 164, 74]).mean()

def preprocess_frame(obs):
    # Crop and resize
    img = obs[25:201:2, ::2]

    # Convert to greyscale
    img = img.mean(axis=2)

    # Improve contrast
    img[img==color] = 0

    # Normalzie image
    img = (img - 128) / 128 - 1

    # Reshape to 80*80*1
    img = img.reshape(88,80)

    return img 

## Initialize gym environment and explore game screens

env = gym.make("SpaceInvaders-v0")
state_size = env.observation_space.shape[0]
action_size = env.action_space.n
print("Actions available(%d): %r"%(env.action_space.n, env.env.get_action_meanings()))

observation = env.reset()
# Game Screen
for i in range(11):
  if i > 9:
    plt.imshow(observation)
    plt.show()
  observation, _, _, _ = env.step(1)

# Preprocessed Game Screen
obs_preprocessed = preprocess_frame(observation).reshape(88,80)
plt.imshow(obs_preprocessed)
plt.show()

# Evaluation

# **** Caution: Do not modify this cell ****
# initialize total reward across episodes
cumulative_reward = 0
episode = 0

def evaluate(episodic_reward):
  '''
  Takes in the reward for an episode, calculates the cumulative_avg_reward
    and logs it in wandb. If episode > 100, stops logging scores to wandb.
    Called after playing each episode. See example below.

  Arguments:
    episodic_reward - reward received after playing current episode
  '''
  global episode
  global cumulative_reward
  episode += 1
  print("Episode: %d"%(episode))

  # your models will be evaluated on 100-episode average reward
  # therefore, we stop logging after 100 episodes
  if (episode > 100):
    print("Scores from episodes > 100 won't be logged in wandb.")
    return

  # log total reward received in this episode to wandb
  wandb.log({'episodic_reward': episodic_reward})

  # add reward from this episode to cumulative_reward
  cumulative_reward += episodic_reward

  # calculate the cumulative_avg_reward
  # this is the metric your models will be evaluated on
  cumulative_avg_reward = cumulative_reward/episode

  # log cumulative_avg_reward over all episodes played so far
  wandb.log({'cumulative_avg_reward': cumulative_avg_reward})

### Play a random game, log reward and gameplay video in wandb
# In this section we'll show you how to save a model in Weights & Biases. This is necessary in order for us to evaluate your model.

# Let's train a very basic model to start.

class DQN:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=20000)
        self.learning_rate = wandb.config.learning_rate
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

    def train(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
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


# Here we adapt the code from the earlier section where we took random actions. We replace these random actions with the outputs from a basic model.


# initialize environment
env = gym.make('SpaceInvaders-v0')

# initialize a new wandb run
wandb.init(project="qualcomm")

# define hyperparameters
wandb.config.episodes = 100
wandb.config.batch_size = 32
wandb.config.learning_rate = 0.003
input_shape = (None, 88, 80, 1)
agent = DQN(state_size, action_size)

# record gameplay video
display = Display(visible=0, size=(1400, 900))
display.start()

# run for 100 episodes
for i in range(wandb.config.episodes):
  # Set reward received in this episode = 0 at the start of the episode
  episodic_reward = 0

  # record a video of the game using wrapper
  env = gym.wrappers.Monitor(env, './video', force=True)

  # play a random game
  state = env.reset()
  state = preprocess_frame(state)

  done = False
  while not done:
    # get prediction for next action from model
    # ****TODO: replace this with your model's prediction****
    action = agent.act(state)

    # perform the action and fetch next state, reward
    next_state, reward, done, _ = env.step(action)
    next_state = preprocess_frame(next_state)
    agent.remember(state, action, reward, next_state, done)
    state = next_state

    episodic_reward += reward

  # call evaluation function - takes in reward received after playing an episode
  # calculates the cumulative_avg_reward over 100 episodes & logs it in wandb
  evaluate(episodic_reward)

  if len(agent.memory) > wandb.config.batch_size:
    agent.train(wandb.config.batch_size)

  # your models will be evaluated on 100-episode average reward
  # therefore, we stop logging after 100 episodes
  if (i == 99):
    # ---- Save the model in Weights & Biases ----
    agent.save(os.path.join(wandb.run.dir, "model.h5"))
    break

  record_video = False
  env.close()

  # render gameplay video
  if (i %50 == 0):
    mp4list = glob.glob('video/*.mp4')
    if len(mp4list) > 0:
      print(len(mp4list))
      mp4 = mp4list[-1]
      video = io.open(mp4, 'r+b').read()
      encoded = base64.b64encode(video)

      # log gameplay video in wandb
      wandb.log({"gameplays": wandb.Video(mp4, fps=4, format="gif")})

      ## display gameplay video
      #ipythondisplay.display(HTML(data='''<video alt="" autoplay
      #            loop controls style="height: 400px;">
      #            <source src="data:video/mp4;base64,{0}" type="video/mp4" />
      #        </video>'''.format(encoded.decode('ascii'))))

# Load the model
agent.load(os.path.join(wandb.run.dir, "model.h5")) 
