import gym
from gym import logger as gymlogger
from gym.wrappers import Monitor
gymlogger.set_level(30)

import numpy as np
import importlib
import math
import glob
import io
import os,sys
import cv2
import base64
import tensorflow as tf
from collections import deque
from datetime import datetime

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 


# Add argument parsing
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--wandb",action="store_true",help="Log the run in wandb")
parser.add_argument("-e","--episodes",type=int,default=100,help="Number of episodes to play")
parser.add_argument("-i","--input",help="Fullname of the file containing save state of the model to be loaded.")
parser.add_argument("-m","--model",default="DQN_simple",help="Name of the model to be used.")
args = parser.parse_args()

# Logging
import logging
logging.basicConfig(format='[%(levelname)s] %(message)s',level=logging.INFO)


if args.wandb:
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
logging.debug("Actions available(%d): %r"%(env.action_space.n, env.env.get_action_meanings()))

# Evaluation

cumulative_reward = 0
rewards = deque([0],maxlen=50)
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
  rewards.append(episodic_reward)
  logging.info("Episode: %4d - %d - %d"%(episode,episodic_reward,sum(rewards)/len(rewards)))

  if args.wandb:
    # your models will be evaluated on 100-episode average reward
    # therefore, we stop logging after 100 episodes
    if (episode > 100):
      #print("Scores from episodes > 100 won't be logged in wandb.")
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

# Signal handling
# Save on exit
import signal

def saveModel():
  # Save the model
  saveName = os.path.join("h5",f"{args.model}_{os.getpid()}.h5")
  agent.save(saveName)
  logging.info(f"Saving model to {saveName}")

def signal_handler(sig,f):
  saveModel()
  sys.exit(11)
signal.signal(signal.SIGINT,signal_handler)

# initialize environment
env = gym.make('SpaceInvaders-v0')

if args.wandb:
  # initialize a new wandb run
  wandb.init(project="qualcomm")

  # define hyperparameters
  wandb.config.episodes = args.episodes
  wandb.config.batch_size = 32
  wandb.config.learning_rate = 0.003
  config = wandb.config
else:
  class Config:pass
  config = Config()
  config.episodes = args.episodes
  config.batch_size = 32
  config.learning_rate = 0.003

modelModule = importlib.import_module(f"models.{args.model}")
agent = modelModule.Model(state_size, action_size,config)
if args.input is not None:
  agent.load(args.input)
  logging.info(f"Model for {args.model} loaded from '{args.input}'")

logging.info(f"Running {args.episodes} of model '{args.model}'")
# record a video of the game using wrapper
env = gym.wrappers.Monitor(env, './video', force=True)

for i in range(config.episodes):
  # Set reward received in this episode = 0 at the start of the episode
  episodic_reward = 0

  # play a random game
  state = env.reset()
  state = preprocess_frame(state)

  done = False
  while not done:
    # get prediction for next action from model
    action = agent.act(state)

    # perform the action and fetch next state, reward
    next_state, reward, done, _ = env.step(action)
    next_state = preprocess_frame(next_state)
    agent.remember(state, action, reward, next_state, done)
    state = next_state

    episodic_reward += reward

  # call evaluation function - takes in reward received after playing an episode
  # calculates the cumulative_avg_reward over args.episodes & logs it in wandb
  evaluate(episodic_reward)

  agent.train()

  # your models will be evaluated on 100-episode average reward
  # therefore, we stop logging after 100 episodes
  if args.wandb and i == 99:
    # ---- Save the model in Weights & Biases ----
    logging.warn("Stopping wandb run after 100 episodes")
    agent.save(os.path.join(wandb.run.dir, "model.h5"))
    break

  env.close()

  # render gameplay video
  if args.wandb and i %99 == 0:
    mp4list = glob.glob('video/*.mp4')
    if len(mp4list) > 0:
      mp4 = mp4list[-1]
      video = io.open(mp4, 'r+b').read()
      encoded = base64.b64encode(video)

      # log gameplay video in wandb
      wandb.log({"gameplays": wandb.Video(mp4, fps=4, format="gif")})

saveModel()
## Load the model
#agent.load(os.path.join(wandb.run.dir, "model.h5")) 
