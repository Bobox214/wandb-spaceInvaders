import gym
from gym import logger as gymlogger
from gym.wrappers import Monitor
gymlogger.set_level(30)

import resource
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
import logging
from PIL import Image

from wrappers import *

# Add argument parsing
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--wandb",action="store_true",help="Log the run in wandb")
parser.add_argument("-e","--episodes",type=int,default=100,help="Number of episodes to play")
parser.add_argument("-i","--input",help="Fullname of the file containing save state of the model to be loaded.")
parser.add_argument("-m","--model",default="DQN",help="Name of the model to be used.")
parser.add_argument("-n","--name",help="Name used for storing logs.")
parser.add_argument("--gpu",action="store_true",help="Run on the GPU, else only the CPU.")
args = parser.parse_args()

if args.wandb and args.input is None:
      logging.fatal("When logging to WanDB with --wandb, an input file for weights is required")
      sys.exit(12)

if args.wandb and args.episodes != 100:
      logging.fatal("When logging to WanDB with --wandb, number of episodes must be 100")
      sys.exit(12)

if args.name is not None:
    baseName = args.name
else:
    baseName = args.model
curTimeS = time.strftime("%Y%m%d_%H%M%S")
saveBaseName = os.path.join("log",f"{baseName}_{curTimeS}")

# Logging
logging.basicConfig(format='[%(levelname)s] %(message)s',level=logging.INFO)
logF = logging.FileHandler(saveBaseName+'.log')
logF.setLevel(logging.INFO)
logF.setFormatter(logging.Formatter('[%(asctime)s][%(levelname)s] %(message)s'))
logging.getLogger('').addHandler(logF)
logging.info(f"Logging to file '{saveBaseName}.log'")

# Tensorflow loading and configuration
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf
if not args.gpu:
  # Remove all GPUs
  tf.config.set_visible_devices([],'GPU')
devices = tf.config.list_logical_devices()
logging.info("Tensorflow visible devices\n\t"+"\n\t".join(d.name for d in devices))
#tf.debugging.set_log_device_placement(True)

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
  if episode==1 or episode%10==0:
    mean = sum(rewards)//len(rewards)
    speed = frame/(time.time()-startTime)
    mem = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/(1024*1024)
    logging.info(f"Episode: {episode:4} Frame: {frame//1000:5}k Score: {episodic_reward:4} Mean50: {mean:4} Speed: {speed:.3f}f/s Mem: {mem:2.2f}GB "+agent.inlineInfo())

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
  saveName = saveBaseName+'.h5'
  agent.save(saveName)
  logging.info(f"Saving model to {saveName}")
  recordLastRun(env)
  # Speed
  duration = time.time()-startTime
  logging.info(f"Run {episode} episodes in {duration:.0f}s at {frame/duration:.3f}f/s")
  # Memory consumption
  mem = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
  logging.info(f"Memory peak: {mem/(1024*1024):0.2f}GB")
  # Record a video

def signal_handler(sig,f):
  finalize()
  sys.exit(11)
signal.signal(signal.SIGINT,signal_handler)

class Config:pass
config = Config()
config.episodes = args.episodes
config.batch_size = 32
config.min_experience_size = 50000
config.experience_buffer_size = 1000000
config.learning_rate = 0.003
config.device = 'gpu' if args.gpu else 'cpu'

if args.wandb:
  # initialize a new wandb run
  wandb.init(project="qualcomm")
  # define hyperparameters
  for k,v in config.__dict__.items():
      setattr(wandb.config,k,v)

recorded = False
def recordLastRun(env):
  global recorded
  if recorded: return
  recorded = True
  logging.info("Recording a run in video.")
  env = gym.wrappers.Monitor(env, './video', force=True)
  state = env.reset()
  done = False
  rewards = 0
  frame = 0
  while not done:
    action,Qs  = agent.play(state)
    next_state, reward, done, _ = env.step(action)
    state = next_state
    rewards += reward
    if frame%100==0:
      logging.info(f"F:{frame//100} Qs: {Qs} Action: {action}")
    frame += 1
  logging.info(f"\tScore: {rewards}")
  # Find video and move it to saveBaseName
  mp4list = glob.glob('video/*.mp4')
  if len(mp4list) > 0:
    mp4 = mp4list[-1]
    os.system(f'mv {mp4} {saveBaseName}.mp4')

## Initialize gym environment
env = gym.make("SpaceInvaders-v0")
env = MaxAndSkipWrapper(env)
env = ImageProcessWrapper(env)
env = FrameStackWrapper(env)
env = ClipRewardWrapper(env)

#
# Create model and load weights if requested
#
module = importlib.import_module(f"models.{args.model}")
agent = module.Model(env,config,eval=args.wandb)
if args.input is not None:
  agent.load(args.input)
  logging.info(f"Model for {args.model} loaded from '{args.input}'")
logging.info(f"Running {args.episodes} episodes of model '{args.model}'")

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

    episodic_reward += int(reward)

  # call evaluation function - takes in reward received after playing an episode
  # calculates the cumulative_avg_reward over args.episodes & logs it in wandb
  evaluate(episodic_reward)

  agent.train(frame)

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
