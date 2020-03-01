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
    # Normalize
    img = img/256
    # Reshape
    img = np.expand_dims(img,2)
    return img 

