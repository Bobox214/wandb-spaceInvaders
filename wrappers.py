from collections import deque
import gym
import numpy as np
import cv2

class ImageProcessWrapper(gym.ObservationWrapper):
  # Preprocessing - crop images, convert them to 1D black and white image tensors
  #   Image dimensions - (210, 160, 3)
  #   Output dimensions - (88, 80, 1)
  def __init__(self,env=None):
    super().__init__(env)
    self.observation_space = gym.spaces.Box(low=0,high=255,shape=(88,64,1),dtype=np.uint8)
  def observation(self,obs):
    return self.process(obs)
  @staticmethod
  def process(frame):
    # Crop , resize, convert to gray and kept 3D
    # Crop and resize
    img = frame[25:200,15:145]
    img = cv2.resize(img,(64,88),interpolation=cv2.INTER_AREA)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
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

class FireResetWrapper(gym.Wrapper):
    # For breakout test, need to start firing
    def __init__(self,env):
        super().__init__(env)
    def reset(self,**kwargs):
        self.env.reset(**kwargs)
        obs,_,done,_ = self.env.step(1)
        if done:
            self.env.reset(**kwargs)
        obs,_,done,_ = self.env.step(2)
        if done:
            self.env.reset(**kwargs)
        return obs

            
class MaxAndSkipWrapper(gym.Wrapper):
    # Apply each step 3 times with the same action
    # return obs as the max over the three obs
    def __init__(self,env):
        super().__init__(env)
        self._obs_buffer = np.zeros( (3,) + env.observation_space.shape , dtype=env.observation_space.dtype)
    def step(self,action):
        total_reward = 0
        done = False
        for i in range(3):
            if not done:
                obs,reward,done,info=self.env.step(action)
                total_reward += reward
            self._obs_buffer[i] = obs
        max_obs = self._obs_buffer.max(axis=0)
        return max_obs,total_reward,done,info

