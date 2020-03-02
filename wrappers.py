from collections import deque
import gym
import numpy as np
import cv2

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
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Reshape
    img = np.expand_dims(img,2)
    return img 

class FrameStackWrapper(gym.Wrapper):
  # Stack last 4 Frames as observation.
  # This should allow the NN to see moving sprites
  def __init__(self,env):
    super().__init__(env)
    self.frames = deque([],maxlen=4)
    oldShape = env.observation_space.shape
    self.observation_space = gym.spaces.Box(
      low=0,high=255
    , shape=oldShape[:-1] + (oldShape[-1]*4,)
    , dtype=env.observation_space.dtype
    )
  def reset(self):
    obs = self.env.reset()
    for _ in range(4):
      self.frames.append(obs)
    return self._get_obs()
  
  def step(self,action):
      obs,reward,done,info = self.env.step(action)
      self.frames.append(obs)
      return self._get_obs(),reward,done,info

  def _get_obs(self):
    return np.concatenate(self.frames,-1)
