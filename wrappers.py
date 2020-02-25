import gym
import numpy as np

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
    img = img.reshape(88,80,1)
    return img 

