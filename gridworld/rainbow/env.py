# -*- coding: utf-8 -*-
from collections import deque
import random
import cv2
import torch
import gym
from gym_minigrid import wrappers as wrappers
from core.enviornment import PaddingWrapper, MoveDirActionWrapper
from core.enviornment import MinAtarEnv

class Minigrid2Image(gym.ObservationWrapper):
    def __init__(self, env, args):
        gym.ObservationWrapper.__init__(self, env)
        self.device = args.device
        self.observation_space = env.observation_space.spaces['image']

    @property
    def steps(self):
        return self.unwrapped.step_count

    def observation(self, observation):
        return torch.tensor(observation['image'].transpose([2, 0, 1]), device=self.device, dtype=torch.float32)

    # Uses loss of life as terminal signal
    def train(self):
        self.training = True

    # Uses standard terminal signal
    def eval(self):
        self.training = False

class Numpy2Torch(gym.ObservationWrapper):
    def __init__(self, env, args):
        gym.ObservationWrapper.__init__(self, env)
        self.device = args.device

    def observation(self, observation):
        return torch.tensor(observation.transpose([2, 0, 1]), device=self.device, dtype=torch.float32)

def get_env(args):
    if "MiniGrid" in args.env:
        env = gym.make(args.env)
        if args.action == "move_dir":
            env = MoveDirActionWrapper(env)
        env = Minigrid2Image(PaddingWrapper(wrappers.FullyObsWrapper(env)), args)
        #env.unwrapped.max_steps=1e10
        return env
    elif "Minatar" in args.env:
        env = Numpy2Torch(MinAtarEnv(args.env.replace("Minatar-", ""), 0.1), args)
        return env