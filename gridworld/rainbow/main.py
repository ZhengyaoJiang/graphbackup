# -*- coding: utf-8 -*-
from __future__ import division
import argparse
import bz2
from datetime import datetime
import os
import pickle

import atari_py
import numpy as np
import torch
from tqdm import trange

from rainbow.agent import Agent
from rainbow.env import get_env
from rainbow.memory import ReplayMemory
from rainbow.test import test
from core.util import PositionRecorder
import wandb
import pandas as pd


# Note that hyperparameters may originally be reported in ATARI game frames instead of agent steps
parser = argparse.ArgumentParser(description='Rainbow')
parser.add_argument('--id', type=str, default='default', help='Experiment ID')
parser.add_argument('--exp_group', type=str, default='test', help='exp_group')
parser.add_argument('--seed', type=int, default=123, help='Random seed')
parser.add_argument('--disable-cuda', action='store_true', help='Disable CUDA')
parser.add_argument('--env', type=str, default='space_invaders', help='ATARI and gridworld game')
parser.add_argument('--T-max', type=int, default=int(50e6), metavar='STEPS', help='Number of training steps (4x number of frames)')
parser.add_argument('--max-episode-length', type=int, default=int(108e3), metavar='LENGTH', help='Max episode length in game frames (0 to disable)')
parser.add_argument('--history-length', type=int, default=4, metavar='T', help='Number of consecutive states processed')
parser.add_argument('--architecture', type=str, default='canonical', choices=['canonical', 'data-efficient', 'gridworld'], metavar='ARCH', help='Network architecture')
parser.add_argument('--hidden-size', type=int, default=512, metavar='SIZE', help='Network hidden size')
parser.add_argument('--noisy-std', type=float, default=0.1, metavar='σ', help='Initial standard deviation of noisy linear layers')
parser.add_argument('--atoms', type=int, default=51, metavar='C', help='Discretised size of value distribution')
parser.add_argument('--V-min', type=float, default=-10, metavar='V', help='Minimum of value distribution support')
parser.add_argument('--V-max', type=float, default=10, metavar='V', help='Maximum of value distribution support')
parser.add_argument('--model', type=str, metavar='PARAMS', help='Pretrained model (state dict)')
parser.add_argument('--memory-capacity', type=int, default=int(1e6), metavar='CAPACITY', help='Experience replay memory capacity')
parser.add_argument('--replay-frequency', type=int, default=4, metavar='k', help='Frequency of sampling from memory')
parser.add_argument('--priority-exponent', type=float, default=0.5, metavar='ω', help='Prioritised experience replay exponent (originally denoted α)')
parser.add_argument('--priority-weight', type=float, default=0.4, metavar='β', help='Initial prioritised experience replay importance sampling weight')
parser.add_argument('--multi-step', type=int, default=3, metavar='n', help='Number of steps for multi-step return')
parser.add_argument('--discount', type=float, default=0.99, metavar='γ', help='Discount factor')
parser.add_argument('--target-update', type=int, default=int(8e3), metavar='τ', help='Number of steps after which to update target network')
parser.add_argument('--reward-clip', type=int, default=1, metavar='VALUE', help='Reward clipping (0 to disable)')
parser.add_argument('--learning-rate', type=float, default=0.0000625, metavar='η', help='Learning rate')
parser.add_argument('--adam-eps', type=float, default=1.5e-4, metavar='ε', help='Adam epsilon')
parser.add_argument('--batch-size', type=int, default=32, metavar='SIZE', help='Batch size')
parser.add_argument('--norm-clip', type=float, default=10, metavar='NORM', help='Max L2 norm for gradient clipping')
parser.add_argument('--learn-start', type=int, default=int(20e3), metavar='STEPS', help='Number of steps before starting training')
parser.add_argument('--evaluate', action='store_true', help='Evaluate only')
parser.add_argument('--evaluation-interval', type=int, default=100000, metavar='STEPS', help='Number of training steps between evaluations')
parser.add_argument('--evaluation-episodes', type=int, default=10, metavar='N', help='Number of evaluation episodes to average over')
# TODO: Note that DeepMind's evaluation method is running the latest agent for 500K frames ever every 1M steps
parser.add_argument('--evaluation-size', type=int, default=500, metavar='N', help='Number of transitions to use for validating Q')
parser.add_argument('--render', action='store_true', help='Display screen (testing only)')
parser.add_argument('--enable-cudnn', action='store_true', help='Enable cuDNN (faster but nondeterministic)')
parser.add_argument('--checkpoint-interval', default=0, help='How often to checkpoint the model, defaults to 0 (never checkpoint)')
parser.add_argument('--memory', help='Path to save/load the memory from')
parser.add_argument('--disable-bzip-memory', action='store_true', help='Don\'t zip the memory file. Not recommended (zipping is a bit slower and much, much smaller)')


parser.add_argument("--exploration", default="epsilon_greedy", type=str,
                    choices=["epsilon_greedy", "AVAHash_epsilon", "AVAHash_UCB", "noisy_net"])
parser.add_argument("--epsilon", default=0.02, type=float)
parser.add_argument("--disable_noisy", action="store_true")
parser.add_argument("--c", default=2.0, type=float)

parser.add_argument("--disable_steps", action="store_true")
parser.add_argument("--disable_dist", action="store_true")
parser.add_argument("--disable_duelling", action="store_true")
parser.add_argument("--disable_double", action="store_true")
parser.add_argument("--offline", action="store_true")
parser.add_argument("--debug", action="store_true")

parser.add_argument("--action", type=str, default="raw", choices=["raw", "move_dir"])


# Setup
args = parser.parse_args()
args.no_op = True

terms = args.id.split("-")
group = terms[0] + "-" + terms[1]

results_dir = os.path.expandvars(os.path.expanduser(os.path.join('~/logs/ava', args.id)))
if not os.path.exists(results_dir):
    os.makedirs(results_dir)
wandb.init(config=args, project="AVA", name=args.id, group=group, dir=results_dir, job_type=args.exp_group)


print(' ' * 26 + 'Options')
for k, v in vars(args).items():
    print(' ' * 26 + k + ': ' + str(v))
results_dir = os.path.expandvars(os.path.expanduser(os.path.join('~/logs/ava', args.id)))
if not os.path.exists(results_dir):
    os.makedirs(results_dir)
metrics = {'steps': [], 'rewards': [], 'Qs': [], 'best_avg_reward': -float('inf')}
np.random.seed(np.random.randint(1, 10000))
torch.manual_seed(np.random.randint(1, 10000))
if torch.cuda.is_available() and not args.disable_cuda:
    args.device = torch.device('cuda')
    torch.cuda.manual_seed(np.random.randint(1, 10000))
    torch.backends.cudnn.enabled = args.enable_cudnn
else:
    args.device = torch.device('cpu')

# Environment
env = get_env(args)
env.train()
action_space = env.action_space.n

if "MiniGrid" in args.env:
    image_size = 32
    nb_features = 3
    args.architecture = "gridworld"
elif "Minatar" in args.env:
    image_size = 10
    nb_features = env.observation_space.shape[-1]
    args.architecture = "minatar"
else:
    image_size = 84
    nb_features = 1

if "MiniGrid" in args.env:
    args.history_length = 1
    env_type = "gridworld"
    position_recorder = PositionRecorder([env.grid.width, env.grid.height])
elif "Minatar" in args.env:
    args.history_length = 1
    env_type = "minatar"
else:
    env_type = "atari"

# Simple ISO 8601 timestamped logger
def log(s):
    print('[' + str(datetime.now().strftime('%Y-%m-%dT%H:%M:%S')) + '] ' + s)


def load_memory(memory_path, disable_bzip):
    if disable_bzip:
        with open(memory_path, 'rb') as pickle_file:
            return pickle.load(pickle_file)
    else:
        with bz2.open(memory_path, 'rb') as zipped_pickle_file:
            return pickle.load(zipped_pickle_file)


def save_memory(memory, memory_path, disable_bzip):
    if disable_bzip:
        with open(memory_path, 'wb') as pickle_file:
            pickle.dump(memory, pickle_file)
    else:
        with bz2.open(memory_path, 'wb') as zipped_pickle_file:
            pickle.dump(memory, zipped_pickle_file)



# Agent
dqn = Agent(args, env.action_space.n, nb_features)

# If a model is provided, and evaluate is false, presumably we want to resume, so try to load memory
if args.model is not None and not args.evaluate:
    if not args.memory:
        raise ValueError('Cannot resume training without memory save path. Aborting...')
    elif not os.path.exists(args.memory):
        raise ValueError('Could not find memory file at {path}. Aborting...'.format(path=args.memory))

    mem = load_memory(args.memory, args.disable_bzip_memory)

else:
    mem = ReplayMemory(args, args.memory_capacity, image_size, nb_features, env_type)

priority_weight_increase = (1 - args.priority_weight) / (args.T_max - args.learn_start)


# Construct validation memory
val_mem = ReplayMemory(args, args.evaluation_size, image_size, nb_features, env_type)
T, done = 0, True
while T < args.evaluation_size:
    if done:
        state = env.reset()

    next_state, _, done, *_ = env.step(np.random.randint(0, action_space))
    val_mem.append(state, -1, 0.0, done, env.steps)
    state = next_state
    T += 1

if args.evaluate:
    dqn.eval()    # Set DQN (online network) to evaluation mode
    avg_reward, avg_Q = test(args, 0, dqn, val_mem, metrics, results_dir, evaluate=True)    # Test
    print('Avg. reward: ' + str(avg_reward) + ' | Avg. Q: ' + str(avg_Q))
else:
    record = dict(step=[], mean_episode_return=[])
    # Training loop
    dqn.train()
    T, done = 0, True
    for T in trange(1, args.T_max + 1):
        if done:
            if args.seed > -1:
                env.seed(args.seed)
            else:
                env.seed(np.random.randint(99999))
            state = env.reset()
        if env_type == "gridworld":
            position_recorder.record(env.agent_pos[0], env.agent_pos[1])

        if T % args.replay_frequency == 0:
            dqn.reset_noise()    # Draw a new set of noisy weights

        if args.debug:
            env.render()

        if T >= args.learn_start:
            action = dqn.act_with_exploration(state, env.steps, args.epsilon)
        else:
            action = dqn.act_with_exploration(state, env.steps, 1.0)    # Choose an action greedily (with noisy weights)
        next_state, reward, done, *_ = env.step(action)    # Step
        if args.reward_clip > 0:
            if args.disable_steps and env_type == "gridworld":
                reward = float(reward>0)
            if env_type == "atari":
                reward = max(min(reward, args.reward_clip), -args.reward_clip)    # Clip rewards

        if T <= args.learn_start or not args.offline:
            mem.append(state, action, reward, done, env.steps)    # Append transition to memory
        #if done:
        #    print(end="")

        # Train and test
        if T >= args.learn_start:
            mem.priority_weight = min(mem.priority_weight + priority_weight_increase, 1)    # Anneal importance sampling weight β to 1

            if T % args.replay_frequency == 0:
                dqn.learn(mem, T)    # Train with n-step distributional double-Q learning

            if T % args.evaluation_interval == 0:
                dqn.eval()    # Set DQN (online network) to evaluation mode
                avg_reward, avg_Q = test(args, T, dqn, val_mem, metrics, results_dir)    # Test
                log('T = ' + str(T) + ' / ' + str(args.T_max) + ' | Avg. reward: ' + str(avg_reward) + ' | Avg. Q: ' + str(avg_Q))
                wandb.log(dict(episode_return=avg_reward, avg_Q=avg_Q), step=T)

                record["step"].append(T)
                record["mean_episode_return"].append(avg_reward)
                dqn.train()    # Set DQN (online network) back to training mode

                if env_type == "gridworld":
                    density = position_recorder.get_density()
                    print(density)
                    wandb.log(dict(position_heatmap=wandb.Image(density*255)), step=T)

                # If memory path provided, save it
                if args.memory is not None:
                    save_memory(mem, args.memory, args.disable_bzip_memory)

            # Update target network
            if T % args.target_update == 0:
                dqn.update_target_net()

            # Checkpoint the network
            if (args.checkpoint_interval != 0) and (T % args.checkpoint_interval == 0):
                dqn.save(results_dir, 'checkpoint.pth')

        state = next_state
    pd.DataFrame(record).to_csv(f"{results_dir}/logs.csv")

env.close()
