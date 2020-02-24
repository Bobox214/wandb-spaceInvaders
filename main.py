import gym
from gym import logger as gymlogger
from gym.wrappers import Monitor
gymlogger.set_level(30)

import time
import numpy as np
import wandb
import importlib
import math
import glob
import io
import os,sys
import cv2
import base64
from collections import deque

# Add argument parsing
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--wandb",action="store_true",help="Log the run in wandb")
parser.add_argument("-e","--episodes",type=int,default=100,help="Number of episodes to play")
parser.add_argument("-i","--input",help="Fullname of the file containing save state of the model to be loaded.")
parser.add_argument("-m","--model",default="DQN_sanity",help="Name of the model to be used.")
args = parser.parse_args()

if args.wandb and args.input is None:
      logging.fatal("When logging to WanDB with --wandb, an input file for weights is required")
      sys.exit(12)

if args.wandb and args.episodes != 100:
      logging.fatal("When logging to WanDB with --wandb, number of episodes must be 100")
      sys.exit(12)

# Logging
import logging
logging.basicConfig(format='[%(levelname)s] %(message)s',level=logging.INFO)

# Tensorflow loading and configuration
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf
devices = tf.config.list_physical_devices()
logging.info("Tensorflow available devices\n\t"+"\n\t".join(d.name for d in devices))

class ImageProcessWrapper(gym.ObservationWrapper):
  # Preprocessing - crop images, convert them to 1D black and white image tensors
  #   Image dimensions - (210, 160, 3)
  #   Output dimensions - (88, 80, 1)
  color = np.array([210, 164, 74]).mean()
  def __init__(self,env=None):
    super().__init__(env)
    self.observation_space = gym.spaces.Box(low=0,high=255,shape=(88,80,1),dtype=np.uint8)
  def observation(self,obs):
    return self.process(obs)
  @staticmethod
  def process(frame):
    # Crop and resize
    img = frame[25:201:2, ::2]
    # Convert to greyscale
    img = img.mean(axis=2)
    # Improve contrast
    img[img==ImageProcessWrapper.color] = 0
    # Normalize
    img = (img - 128) / 128 - 1
    # Reshape
    img = img.reshape(88,80)
    return img 


# Evaluation

cumulative_reward = 0
rewards = deque([0],maxlen=50)
episode = 0
frame   = 0

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
  lastMeanScore = int(sum(rewards)/len(rewards))
  speed = frame/(time.time()-startTime)
  logging.info(f"Episode: {episode:4d} LastMeanScore: {lastMeanScore:4d} Speed: {speed:.3f}f/s "+agent.inlineInfo())

  if args.wandb:
    if (episode > 100):
      raise SystemError("Should have been restricted to 100 episodes")

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

def finalize():
  # Save the model
  saveName = os.path.join("h5",f"{agent.modelName}_{os.getpid()}.h5")
  agent.save(saveName)
  logging.info(f"Saving model to {saveName}")
  # Speed
  logging.info(f"Run {episode} episodes at  {frame/(time.time()-startTime):.3f}f/s")
  # Record a video
  recordLastRun(env)

def signal_handler(sig,f):
  finalize()
  sys.exit(11)
signal.signal(signal.SIGINT,signal_handler)

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

## Initialize gym environment and explore game screens
env = gym.make("SpaceInvaders-v0")
env = ImageProcessWrapper(env)

#
# Create model and load weights if requested
#
module = importlib.import_module(f"models.{args.model}")
agent = module.Model(env,config,eval=args.wandb)
if args.input is not None:
  agent.load(args.input)
  logging.info(f"Model for {args.model} loaded from '{args.input}'")
logging.info(f"Running {args.episodes} episodes of model '{agent.modelName}'")

recorded = False
def recordLastRun(env):
  global recorded
  if recorded: return
  recorded = True
  logging.info("Recording a run in video.")
  env = gym.wrappers.Monitor(env, './video', force=True)
  state = env.reset()
  done = False
  while not done:
    action  = agent.play(state)
    next_state, _, done, _ = env.step(action)
    state = next_state


startTime = time.time()
for i in range(config.episodes):
  # Set reward received in this episode = 0 at the start of the episode
  episodic_reward = 0

  state = env.reset()

  done = False
  while not done:
    frame += 1
    # get prediction for next action from model
    action = agent.act(state)

    # perform the action and fetch next state, reward
    next_state, reward, done, _ = env.step(action)
    agent.remember(state, action, reward, next_state, done)
    state = next_state

    episodic_reward += reward

  # call evaluation function - takes in reward received after playing an episode
  # calculates the cumulative_avg_reward over args.episodes & logs it in wandb
  evaluate(episodic_reward)

  agent.train()

finalize()

env.close()

if args.wandb:
  # ---- Save the model in Weights & Biases ----
  agent.save(os.path.join(wandb.run.dir, "model.h5"))
  # render gameplay video
  mp4list = glob.glob('video/*.mp4')
  if len(mp4list) > 0:
    mp4 = mp4list[-1]
    print("MP4",mp4)
    # log gameplay video in wandb
    wandb.log({"gameplays": wandb.Video(mp4, fps=4, format="gif")})

## Load the model
#agent.load(os.path.join(wandb.run.dir, "model.h5")) 