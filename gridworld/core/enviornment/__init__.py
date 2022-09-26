import gym
from .minatar import MinAtarEnv
from .minigrid import *

gym.envs.register(
     id='random-v0',
     entry_point='nlrl.enviornment.random:RandomEnv',
     max_episode_steps=20,
)