from collections import deque
import gym
import numpy as np
import cv2

class ImageProcessWrapper(gym.ObservationWrapper):
    # Preprocessing image 
    # Crop , resize, convert to gray and kept 3D
    def __init__(self,env=None):
        super().__init__(env)
        self.observation_space = gym.spaces.Box(low=0,high=255,shape=(88,64,1),dtype=np.uint8)
    def observation(self,obs):
        # Crop
        obs = obs[25:200,15:145]
        # Keep Y part of image
        obs = obs[:, :, 0] * 0.299 + obs[:, :, 1] * 0.587 + obs[:, :, 2] * 0.114
        # Resize
        obs = cv2.resize(obs,(64,88),interpolation=cv2.INTER_AREA)
        # make 3D
        obs = np.expand_dims(obs,2)
        return obs.astype(dtype=np.uint8)

class FrameStackWrapper(gym.ObservationWrapper):
    # Return a stack the last n observations.
    def __init__(self, env, n_steps=4):
        super().__init__(env)
        old_space = env.observation_space
        self.observation_space = gym.spaces.Box(
            old_space.low .repeat(n_steps, axis=2)
        ,   old_space.high.repeat(n_steps, axis=2)
        ,   dtype=np.uint8
        )

    def reset(self):
        self.buffer = np.zeros_like(self.observation_space.low, dtype=np.uint8)
        return self.observation(self.env.reset())

    def observation(self, observation):
        self.buffer[:,:,:-1] = self.buffer[:,:,1:]
        self.buffer[:,:,-1:] = observation
        return self.buffer

class MaxAndSkipWrapper(gym.Wrapper):
    # Apply each step 'skip' times with the same action
    # return obs as the max over the last 2 obs, to avoid clipping
    def __init__(self,env,skip=4):
        super().__init__(env)
        self._skip = skip
    def step(self,action):
        total_reward = 0
        done = False
        for _ in range(self._skip):
            obs,reward,done,info=self.env.step(action)
            total_reward += reward
            self._obs_buffer[0] = self._obs_buffer[1]
            self._obs_buffer[1] = obs
            if done:
                break
        max_obs = np.max(self._obs_buffer,axis=0)
        return max_obs,total_reward,done,info
    def reset(self):
        self._obs_buffer = np.zeros( (2,)+self.env.observation_space.shape,dtype=np.uint8)
        obs = self.env.reset()
        self._obs_buffer[1] = obs
        return obs

class NormalizeWrapper(gym.ObservationWrapper):
    def observation(self, obs):
        return obs.astype(np.float32) / 255.0

class AsFloatWrapper(gym.ObservationWrapper):
    def observation(self, obs):
        return obs.astype(np.float32)


class ClipRewardWrapper(gym.Wrapper):
    # Reduce rewards to +1 0 or -1
    def step(self,action):
        obs,reward,done,info=self.env.step(action)
        return obs,np.sign(reward),done,info

class ScaleRewardWrapper(gym.Wrapper):
    def __init__(self,env,scale=30):
        super().__init__(env)
        self.scale = scale
    # Reduce rewards to +1 0 or -1
    def step(self,action):
        obs,reward,done,info=self.env.step(action)
        return obs,reward/self.scale,done,info

class NoopResetWrapper(gym.Wrapper):
    # Apply a random number of NoOp action after each reset.
    # Nobody seems to know why it helps convergence.
    # But it seems to work
    def __init__(self,env,noop_max=30):
        super().__init__(env)
        self.noop_max = noop_max
        assert env.unwrapped.get_action_meanings()[0] == 'NOOP'
    def reset(self,**kwargs):
        self.env.reset(**kwargs)
        noops = np.random.randint(1,self.noop_max+1)
        obs = None
        for _ in range(noops):
            obs,_,done,_ = self.env.step(0)
            if done:
                obs = self.env.reset(**kwargs)
        return obs

class LossLifeResetWrapper(gym.Wrapper):
    # Return done on each loss of life.
    # Only perform a real reset when there was a real done
    # Reduce reward by lossCost when losing one life
    def __init__(self,env,lossCost=0):
        super().__init__(env)
        self.lives = 0
        self.realDone = True
        self.lossCost = lossCost
    def step(self,action):
        obs,reward,done,info = self.env.step(action)
        self.realDone = done
        lives = self.env.unwrapped.ale.lives()
        if lives < self.lives and lives>0:
            done = True
        reward = reward-lives*self.lossCost
        self.lives = lives
        return obs,reward,done,info
    def reset(self,**kwargs):
        if self.realDone:
            obs = super().reset(**kwargs)
        else:
            obs,_,_,_ = self.env.step(0)
        self.lives = self.env.unwrapped.ale.lives()
        return obs