import gym
import numpy as np
from gym_minigrid import wrappers as wrappers
from core.enviornment import MinAtarEnv
try:
    from core.enviornment import ProcGen
except:
    print("no procgen")
from core.enviornment import MoveDirActionWrapper
from rainbow.agent import Agent
import torch
import pickle
import os
import glob
import cv2

def get_gym_env(args, image_only=True):
    args.game = args.env
    if "MiniGrid" in args.env:
        env = gym.make(args.env)
        #env.unwrapped.max_steps *= 100
        if args.action == "move_dir":
            env = MoveDirActionWrapper(env)
        if image_only:
            env = Minigrid2Image(wrappers.FullyObsWrapper(env))
        else:
            env = wrappers.FullyObsWrapper(gym.make(args.env))
        return env, env.unwrapped.max_steps
    elif "Minatar" in args.env:
        env = MinAtarEnv(args.env.replace("Minatar-", ""), 0.1)
        return env, env.maximum_steps
    elif "procgen" in args.env:
        env = ProcGen(args.env, debug=args.debug)
        env.env._max_episode_steps = args.max_episode_length
        return env, args.max_episode_length
        return env, args.max_episode_length

def get_nb_actions(env):
    return env.action_space.n

def hashing(image, method):
    if method == "exact":
        return np.array(image, dtype=np.uint8).tobytes()
    elif method == "jpg":
        image = np.array(image, dtype=np.uint8)
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 20]
        result, encimg = cv2.imencode('.jpg', image, encode_param)
        return encimg.tobytes()
    elif method == "downsample":
        image = np.array(image, dtype=np.uint8)
        grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        icon = cv2.resize(grey, (16, 16), interpolation=cv2.INTER_LINEAR)
        discrete = icon // 32
        return np.array(discrete, dtype=np.uint8).tobytes()


class Minigrid2Image(gym.ObservationWrapper):
    def __init__(self, env):
        gym.ObservationWrapper.__init__(self, env)
        self.observation_space = env.observation_space.spaces['image']

    def observation(self, observation):
        return observation['image']

def test(args, Q, s2i, seed, all_seeds=False, rounds=3):
    rewards = []
    env_test, max_steps = get_gym_env(args)
    if "MiniGrid" not in args.env and "Minatar" not in args.env:
        env_test.training = False
    for i in range(rounds):
        steps = 0
        if seed > -1 and not all_seeds:
            env_test.seed(seed)
        else:
            env_test.seed(np.random.randint(99999))
        state = env_test.reset()
        rAll = 0

        while True:
            steps += 1
            #if args.full_obs == 0:
            #    state = env_test.env.env.gen_obs()['image']


            if isinstance(Q, np.ndarray):
                if "MiniGrid" in args.env:
                    state = pad(state)
                s_idx = s2i.get_index(state)
                #if args.debug and seed>5:
                #    env_test.render()
                #    if s_idx in Q:
                #        print(Q[s_idx])

                # Reset environment and get first new observation
                # Break equal tie
                if not (s_idx is None):
                    Qmax = np.max(Q[s_idx])
                    p = (Q[s_idx] == Qmax).astype(float) / np.sum(Q[s_idx] == Qmax).astype(float)
                else:
                    p = np.ones(get_nb_actions(env_test)) / get_nb_actions(env_test)
                a = np.random.choice(np.arange(get_nb_actions(env_test)), p=p)
            else:
                if "MiniGrid" in args.env:
                    state = pad(state)
                Q.eval()

                if isinstance(state, np.ndarray):
                    s = torch.tensor(state, device=args.device, dtype=torch.float32)
                else:
                    s = state.to(args.device)
                if isinstance(Q, Agent):
                    s = s.permute(2, 0, 1)
                    a = Q.act_with_exploration(s, steps, epsilon=0.001, test=True)
                else:
                    s = torch.tensor(state, device=args.device, dtype=torch.float32)
                    a = Q(dict(frame=s[None, None, :]))["action"][0]

            #if i == 0:
            #    render_index_img(vector2index_img(state))
            # Get new state and reward from environment
            state, r, d, *_ = env_test.step(int(a))

            rAll += r
            if d == True or steps == max_steps-1:
                break
        rewards.append(rAll)
    return rewards

def pad(img, width=32):
    img = np.pad(img, ((width-img.shape[0], 0), (width-img.shape[1], 0), (0, 0)))
    return img

def get_action_lists(env_name):
    if "MiniGrid" in env_name:
        return ["left", "right", "forward", "pickup", "drop", "toggle", "done"]

def format_action_value_string(env_name, values):
    actions = get_action_lists(env_name)
    string = ""
    for action, value in zip(actions, values):
        string += f"{action}: {value:.4f}, "
    return string

def get_example_states():
    """
    return: dict[Name -> State]
    """
    result = {}
    source_dir = os.path.expandvars(os.path.expanduser(f"~/project/explore/emptyroomstates/*.pkl"))
    for filename in glob.glob(source_dir):
        with open(filename, 'rb') as f:
            state = pickle.load(f)
            name = filename.split("/")[-1].replace(".pkl","")
            result[name] = pad(state)
    return result


class PositionRecorder():
    def __init__(self, size):
        self.map = np.zeros(size)

    def record(self, x, y):
        self.map[y, x] += 1

    def get_density(self):
        return self.map / self.map.sum()

# Copyright (c) Facebook, Inc. and its affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import collections
import timeit

class Timings:
    """Not thread-safe."""

    def __init__(self):
        self._means = collections.defaultdict(int)
        self._vars = collections.defaultdict(int)
        self._counts = collections.defaultdict(int)
        self.reset()

    def reset(self):
        self.last_time = timeit.default_timer()

    def time(self, name):
        """Save an update for event `name`.
        Nerd alarm: We could just store a
            collections.defaultdict(list)
        and compute means and standard deviations at the end. But thanks to the
        clever math in Sutton-Barto
        (http://www.incompleteideas.net/book/first/ebook/node19.html) and
        https://math.stackexchange.com/a/103025/5051 we can update both the
        means and the stds online. O(1) FTW!
        """
        now = timeit.default_timer()
        x = now - self.last_time
        self.last_time = now

        n = self._counts[name]

        mean = self._means[name] + (x - self._means[name]) / (n + 1)
        var = (
            n * self._vars[name] + n * (self._means[name] - mean) ** 2 + (x - mean) ** 2
        ) / (n + 1)

        self._means[name] = mean
        self._vars[name] = var
        self._counts[name] += 1

    def means(self):
        return self._means

    def vars(self):
        return self._vars

    def stds(self):
        return {k: v ** 0.5 for k, v in self._vars.items()}

    def summary2(self, prefix=""):
        """
        used for uneven count case
        """
        means = self.means()
        counts = np.array([self._counts[k] for k in means.keys()])
        total = sum(np.array(list(means.values()))*counts)

        result = prefix
        for k in sorted(means, key=means.get, reverse=True):
            result += f"\n    %s: %.6fs (%.2f%%) " % (
                k,
                self._counts[k] * means[k],
                100 * self._counts[k] * means[k] / total,
            )
        result += "\nTotal: %.6fs" % (total)
        return result

    def summary(self, prefix=""):
        means = self.means()
        stds = self.stds()
        total = sum(means.values())

        result = prefix
        for k in sorted(means, key=means.get, reverse=True):
            result += f"\n    %s: %.6fms +- %.6fms (%.2f%%) " % (
                k,
                1000 * means[k],
                1000 * stds[k],
                100 * means[k] / total,
            )
        result += "\nTotal: %.6fms" % (1000 * total)
        return result