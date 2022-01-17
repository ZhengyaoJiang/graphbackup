from collections import namedtuple
from gym_minigrid.wrappers import *
from gym_minigrid.minigrid import COLOR_TO_IDX, OBJECT_TO_IDX, STATE_TO_IDX, DIR_TO_VEC
from gym_minigrid.minigrid import *
from minatar import Environment
import gym
from src.rlpyt.rlpyt.spaces.int_box import IntBox
from gym.utils import seeding
from gym import register
import numpy as np
from enum import IntEnum
import torch
from typing import List, Tuple
from gym import error, spaces, utils
import itertools as itt
import gym
import numpy as np
from src.rlpyt.rlpyt.envs.base import Env, EnvStep
from src.rlpyt.rlpyt.samplers.collections import TrajInfo

EnvInfo = namedtuple("EnvInfo", ["game_score", "traj_done"])

class PaddingWrapper(gym.core.ObservationWrapper):
    """
    pad the observation into a square field
    """
    def __init__(self, env, width=16):
        super().__init__(env)

        self.raw_obs_shape = env.observation_space['image'].shape

        self.observation_space.spaces["image"] = spaces.Box(
            low=0,
            high=255,
            shape=(width, width, self.raw_obs_shape[2]),
            dtype='uint8'
        )
        self.height_pad = width - self.raw_obs_shape[0]
        self.width_pad = width - self.raw_obs_shape[1]

    def observation(self, observation):
        img = observation["image"]
        img = np.pad(img, ((self.height_pad, 0), (self.width_pad, 0), (0, 0)))
        return dict(image=img)


class MiniGridEnv(Env):
    def __init__(self, game, seed, id=0):
        env = gym.make(game)
        env = ReseedWrapper(env)
        env = FullyObsWrapper(env)
        env = PaddingWrapper(env)
        self.env = env
        self.id = id
        obs_shape = env.observation_space.spaces["image"].shape
        action_shape = env.action_space.n
        self._action_space = IntBox(low=0, high=action_shape)
        self._observation_space = IntBox(low=0, high=255, shape=(1, 3, obs_shape[0], obs_shape[1]),
                                         dtype="uint8")
        self.seed(seed, id)

    def seed(self, seed=None, id=0):
        self.env.seed(seed)

    def reset(self):
        return self.env.reset()["image"].transpose([2, 0, 1])[None, :]

    def step(self, action):
        s, r, d, _ = self.env.step(action)
        return EnvStep(s["image"].transpose([2, 0, 1])[None, :], r, d, EnvInfo(r, d))


class Actions(IntEnum):
    # Turn left, turn right, move forward
    left = 0
    right = 1
    forward = 2
    # Pick up an object
    pickup = 3
    # Drop an object
    drop = 4
    # Toggle/activate an object
    toggle = 5
    # Done completing task
    done = 6

class Spec():
    def __init__(self, id):
        self.id = id


class MinAtarEnv(Env):
    def __init__(self, game, seed, sticky_action_prob=0.1, id=0):
        super().__init__()
        self.maximum_steps = 5000
        self.env = Environment(game.replace("Minatar-", ""), sticky_action_prob)
        self.spec = Spec("minatar")
        self.channels = [f"channel{i}" for i in range(len(self.env.env.channels))]
        self._action_space = IntBox(low=0, high=self.env.num_actions())
        obs_shp = self.env.state_shape()
        self._observation_space = IntBox(low=0, high=255, shape=(1, obs_shp[2], obs_shp[0], obs_shp[1]),
                                         dtype="uint8")
        self.steps = 0
        self.id = id
        self.seed(seed, id)

    def seed(self, seed, id=0):
        _, seed1 = seeding.np_random(seed)
        if id > 0:
            seed = seed*100 + id
        self.np_random, _ = seeding.np_random(seed)
        # Derive a random seed. This gets passed as a uint, but gets
        # checked as an int elsewhere, so we need to keep it below
        # 2**31.
        seed2 = seeding.hash_seed(seed1 + 1) % 2**31

        self.env.random.seed(seed2)

    def reset(self):
        self.env.reset()
        self.steps = 0
        return self.env.state().transpose([2, 0, 1])[None, :].astype(np.uint8)

    def render(self, mode='human'):
        self.env.display_state()

    def step(self, action):
        r, done = self.env.act(action)
        self.steps += 1
        if self.steps >= self.maximum_steps:
            done = True
        return EnvStep(self.env.state().transpose([2, 0, 1])[None, :].astype(np.uint8),
                       r, done, EnvInfo(r, done))

    # Uses loss of life as terminal signal
    def train(self):
        self.training = True

    # Uses standard terminal signal
    def eval(self):
        self.training = False