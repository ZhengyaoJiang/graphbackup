import numpy as np
import torch
import networkx as nx
import json
import matplotlib.pyplot as plt
import matplotlib as pylab
from random import choices
from src.rlpyt.rlpyt.utils.tensor import select_at_indexes, valid_mean
from src.rlpyt.rlpyt.samplers.collectors import (DecorrelatingStartCollector,
    BaseEvalCollector)
from src.rlpyt.rlpyt.agents.base import AgentInputs
from src.rlpyt.rlpyt.utils.buffer import (torchify_buffer, numpify_buffer, buffer_from_example,
    buffer_method)
from typing import Optional, Dict, Tuple, List, Union


def hashing(image, method):
    if method == "exact":
        return np.array(image, dtype=np.uint8).tobytes()
    else:
        raise NotImplementedError()

class State2Index:
    def __init__(self, hashing_method="exact"):
        self.data = dict()
        self.hashing_method = hashing_method
        self.max = 0
        self.states = []
        self.last_key = ""
        self.stay_count = 0

    def get_index(self, key):
        if not isinstance(key, tuple):
            if isinstance(key, torch.Tensor):
                np_state = key.cpu().numpy()
            elif isinstance(key, np.ndarray):
                np_state = key
            else:
                raise ValueError()
            key = hashing(np_state, self.hashing_method)
            key = (key, 0)
        if key in self.data:
            return self.data[key]
        else:
            return None

    def get_states(self, indexs):
        return [self.states[i] for i in indexs]

    def append_state(self, state, increase_stay=True):
        if isinstance(state, torch.Tensor):
            np_state = state.cpu().numpy()
        else:
            np_state = state
        key = hashing(np_state, self.hashing_method)

        if self.last_key != key:
            self.stay_count = 0
        elif increase_stay:
            self.stay_count += 1
        self.last_key = key
        key = (key, self.stay_count)

        if key not in self.data:
            self.data[key] = len(self.data)
            self.states.append(state)
            self.max += 1
        return key

class TransitionFreq():
    def __init__(self):
        self.freq = {}
        self.transition_count = 0

    def append(self, s, a, r, d, s1):
        self.transition_count += 1
        s = s
        a = int(a)
        s1 = s1
        if s not in self.freq:
            self.freq[s] = {}
        if a not in self.freq[s]:
            self.freq[s][a] = {}
        if (r, d, s1) not in self.freq[s][a]:
            self.freq[s][a][(r, d, s1)] = 0
        self.freq[s][a][(r, d, s1)] += 1

    def save_graph(self, path="."):
        edges = []
        for s, subdict in self.freq.items():
            for a, subsubdict in subdict.items():
                for r, d, s1 in subsubdict.keys():
                    edges.append((s,s1))
        with open(path+'/edges.json', 'w') as f:
            json.dump(edges, f)

        #graph = nx.Graph(edges)
        #plt.savefig(path+"/graphvis.png", bbox_inches="tight")


class CpuResetGraphCollector(DecorrelatingStartCollector):
    """Collector which executes ``agent.step()`` in the sampling loop (i.e.
    use in CPU or serial samplers.)

    It immediately resets any environment which finishes an episode.  This is
    typically indicated by the environment returning ``done=True``.  But this
    collector defers to the ``done`` signal only after looking for
    ``env_info["traj_done"]``, so that RL episodes can end without a call to
    ``env_reset()`` (e.g. used for episodic lives in the Atari env).  The
    agent gets reset based solely on ``done``.
    """

    mid_batch_reset = True

    def __init__(self,
                 rank,
                 envs,
                 samples_np,
                 batch_T,
                 TrajInfoCls,
                 agent=None,  # Present or not, depending on collector class.
                 sync=None,
                 step_buffer_np=None,
                 global_B=1,
                 env_ranks=None # TODO: double check
                 ):
        super(CpuResetGraphCollector, self).__init__(rank, envs, samples_np, batch_T, TrajInfoCls,
                                                     agent=agent, sync=sync, step_buffer_np=step_buffer_np,
                                                     global_B=global_B)
        self.transition_freq = TransitionFreq()
        self.s2i = State2Index()

    def collect_batch(self, agent_inputs, traj_infos, itr):
        # Numpy arrays can be written to from numpy arrays or torch tensors
        # (whereas torch tensors can only be written to from torch tensors).
        agent_buf, env_buf = self.samples_np.agent, self.samples_np.env
        completed_infos = list()
        observation, action, reward = agent_inputs
        obs_pyt, act_pyt, rew_pyt = torchify_buffer(agent_inputs)
        agent_buf.prev_action[0] = action  # Leading prev_action.
        env_buf.prev_reward[0] = reward
        self.agent.sample_mode(itr)

        o_key = self.s2i.append_state(agent_inputs.observation[0], increase_stay=False)
        s_idx = self.s2i.get_index(o_key)

        for t in range(self.batch_T):
            env_buf.observation[t] = observation
            # Agent inputs and outputs are torch tensors.
            act_pyt, agent_info = self.agent.step(obs_pyt, act_pyt, rew_pyt)
            action = numpify_buffer(act_pyt)
            # Environment inputs and outputs are numpy arrays.
            b = 0 # index
            env = self.envs[b]
            o, r, d, env_info = env.step(action[b])

            #env.render()
            o_key = self.s2i.append_state(o)
            s1_idx = self.s2i.get_index(o_key)
            self.transition_freq.append(s_idx, action[b], r, d, s1_idx)
            traj_infos[b].step(observation[b], action[b], r, d, agent_info[b],
                               env_info, s_idx)
            s_idx = s1_idx

            if getattr(env_info, "traj_done", d):
                completed_infos.append(traj_infos[b].terminate(o))
                traj_infos[b] = self.TrajInfoCls()
                o = env.reset()

                o_key = self.s2i.append_state(o)
                s_idx = self.s2i.get_index(o_key)
                #print("end game")
            if d:
                self.agent.reset_one(idx=b)
                #print("end life")
            observation[b] = o
            reward[b] = r
            env_buf.done[t, b] = d
            if env_info:
                env_buf.env_info[t, b] = env_info
            agent_buf.action[t] = action
            env_buf.reward[t] = reward
            if agent_info:
                agent_buf.agent_info[t] = agent_info

        if "bootstrap_value" in agent_buf:
            # agent.value() should not advance rnn state.
            agent_buf.bootstrap_value[:] = self.agent.value(obs_pyt, act_pyt, rew_pyt)

        return AgentInputs(observation, action, reward), traj_infos, completed_infos


def q2v(q, policy="greedy", epsilon=0.02):
    if policy == "greedy":
        return np.max(q)
    elif policy == "epsilon_greedy":
        return np.max(q)*(1-epsilon) + np.sum(epsilon/torch.sum(q.shape)*q)

def graph_limited_backup(agent, freq, states, s2i, discount, breath, depth, double, dist, one_step_backup, source_indexes):
    targets = []
    sa_all = []
    target_states = []

    for source_idx in source_indexes:
        new_s = {source_idx}
        sa = []
        target_states.append(source_idx)
        for step in range(depth):
            new_trans = set()
            for s in new_s:
                if s in freq.freq:
                    for action in freq.freq[s].keys():
                        for r, d, next_state in freq.freq[s][action]:  # loop through different possibilities
                            new_trans.add((s, action, r, d, next_state))
            target_states.extend([t[-1] for t in new_trans])
            if len(new_trans) > breath:
                new_trans = list(new_trans)
                counts = [freq.freq[t[0]][t[1]][t[2:]] for t in new_trans]
                prob = counts
                new_trans = choices(new_trans, weights=prob, k=breath)
                sa.extend([t[:2] for t in new_trans])
                new_s = set([t[-1] for t in new_trans])
            else:
                sa.extend([t[:2] for t in new_trans])
                new_s = set([t[-1] for t in new_trans])
        sa_all.append((source_idx, sa))

    target_states = list(set(target_states))
    states_array = torch.tensor(np.stack(s2i.get_states(target_states))).to(states)
    qs = agent.target(states_array, None, None).cpu()
    i2q = {s: qs[i].cpu() for i, s in enumerate(target_states)}
    if double:
        q_online = agent(states_array, None, None).cpu().detach()
        if dist:
            z = torch.linspace(agent.V_min, agent.V_max, agent.n_atoms)
            q_online = torch.tensordot(q_online, z, dims=1)
        i2max = {s: torch.argmax(q_online[i]) for i, s in enumerate(target_states)}
    else:
        i2max = None

    for source_idx, sa, in sa_all:
        i2q_temp = {}
        for state, action in reversed(sa):
            if state not in i2q_temp:
                i2q_temp[state] = torch.clone(i2q[state])
            v = 0
            overall_count = 0
            for r, d, next_state in freq.freq[state][action]:  # loop though different possibilities
                count = freq.freq[state][action][(r, d, next_state)]
                overall_count += count
                if next_state not in i2q_temp:
                    i2q_temp[next_state] = torch.clone(i2q[next_state])
                idx = i2max[next_state] if double else torch.zeros(0)
                v += count * one_step_backup(i2q_temp[next_state], idx, torch.tensor(r), torch.tensor(d))
            i2q_temp[state][action] = v / overall_count
        targets.append(i2q_temp[source_idx])
    return torch.stack(targets)



def graph_mixed_backup(agent, freq, states, actions, s2i, discount, breath,
                       depth, double, dist, one_step_backup, source_indexes):
    targets = []
    sa_all = []
    target_states = []

    for n, source_idx in enumerate(source_indexes):
        new_s = {source_idx}
        #sa = [(source_idx, actions[n])]
        sa = []
        target_states.append(source_idx)
        for step in range(depth):
            new_trans = set()
            for s in new_s:
                if s in freq.freq:
                    for action in freq.freq[s].keys():
                        for r, d, next_state in freq.freq[s][action]:  # loop though different possibilities
                            new_trans.add((int(s), int(action), float(r), bool(d), int(next_state)))
            target_states.extend([t[-1] for t in new_trans])
            if step == 0: # only expand source action in source state
                new_trans = set()
                for r, d, next_state in freq.freq[source_idx][actions[n]]:  # loop though different possibilities
                    new_trans.add((int(source_idx), int(actions[n]), float(r), bool(d), int(next_state)))

            if len(new_trans) > breath:
                new_trans = list(new_trans)
                counts = [freq.freq[t[0]][t[1]][t[2:]] for t in new_trans]
                prob = counts
                new_trans = choices(new_trans, weights=prob, k=breath)
                sa.extend([t[:2] for t in new_trans])
                new_s = set([t[-1] for t in new_trans])
            else:
                sa.extend([t[:2] for t in new_trans])
                new_s = set([t[-1] for t in new_trans])
        sa_all.append((source_idx, sa))

    target_states = list(set(target_states))
    states_array = torch.tensor(np.stack(s2i.get_states(target_states))).to(states)
    qs = agent.target(states_array, None, None)
    i2q = {s: qs[i].cpu() for i, s in enumerate(target_states)}
    if double:
        q_online = agent(states_array, None, None)
        if dist:
            z = torch.linspace(agent.V_min, agent.V_max, agent.n_atoms)
            q_online = torch.tensordot(q_online, z, dims=1)
        i2max = {s: torch.argmax(q_online[i]) for i, s in enumerate(target_states)}
    else:
        i2max = None

    actions = [int(a) for a in actions]

    freq_map, rewards, dones, next_states, counts = format_freq_dict(freq.freq)
    result = mixed_backup_with_graph(sa_all, freq_map, rewards, dones, next_states, counts,
                                     i2q, i2max, actions, double, dist, discount)
    return result


def format_freq_dict(freq_dict):
    """
    format freq dict into the form acceptable by torch jit script
    :param freq_dict:
    :return:
    """
    freq_count_map = {}
    rewards = []
    dones = []
    next_states = []
    counts = []
    for state, subdict in freq_dict.items():
        if state not in freq_count_map:
            freq_count_map[state] = {}
        for action, subsubdict in subdict.items():
            if action not in freq_count_map[state]:
                freq_count_map[state][action] = []
            for (r, d, s1), count in subsubdict.items():
                freq_count_map[state][action].append(len(rewards))
                rewards.append(r)
                dones.append(d)
                next_states.append(s1)
                counts.append(count)
    return freq_count_map, rewards, dones, next_states, counts


@torch.jit.script
def value_backup(target_q, q_idx, rewards, dones,
                 double_dqn:bool, discount:float, state_value:bool=False):
    if state_value:
        return rewards + (1-dones.float())*discount*target_q

    if double_dqn:
        return rewards + (1-dones.float())*discount*target_q[q_idx]
    else:
        return rewards + (1-dones.float())*discount*torch.max(target_q, dim=-1)[0]


@torch.jit.script
def dist_backup(target_ps, q_idx, rewards, dones,
                double_dqn:bool, discount:float, state_value:bool=False,
                v_min:float=-10.0, v_max:float=10.0, n_atoms:int=51):
    if len(rewards.shape) == 0:
        target_ps = target_ps.unsqueeze(0)
        rewards = rewards.unsqueeze(0)
        dones = dones.unsqueeze(0)
        reshaped = True
        if double_dqn and not state_value:
            q_idx = q_idx.unsqueeze(0)
    else:
        reshaped = False

    delta_z = (v_max - v_min) / (n_atoms - 1)
    z = torch.linspace(v_min, v_max, n_atoms)
    next_z = z * discount  # [P']
    next_z = torch.ger(1 - dones.float(), next_z)  # [B,P']
    ret = rewards.unsqueeze(1)  # [B,1]
    next_z = torch.clamp(ret + next_z, v_min, v_max)  # [B,P']

    z_bc = z.view(1, -1, 1)  # [1,P,1]
    next_z_bc = next_z.unsqueeze(1)  # [B,1,P']
    abs_diff_on_delta = abs(next_z_bc - z_bc) / delta_z
    projection_coeffs = torch.clamp(1 - abs_diff_on_delta, 0, 1)

    with torch.no_grad():
        if double_dqn:
            next_a = q_idx
        else:
            target_qs = torch.tensordot(target_ps, z, dims=1)  # [B,A]
            next_a = torch.argmax(target_qs, dim=-1)  # [B]
        if state_value:
            target_p_unproj = target_ps.unsqueeze(1)  # [B,1,P']
        else:
            target_p_unproj = select_at_indexes(next_a, target_ps)  # [B,P']
            target_p_unproj = target_p_unproj.unsqueeze(1)  # [B,1,P']
        target_p = (target_p_unproj * projection_coeffs).sum(-1)

    if reshaped:
        target_p = target_p[0]
    return target_p

def backup(distributional_rl:bool,target_ps, q_idx, rewards, dones,
                double_dqn:bool, discount:float, state_value:bool=False,
                v_min:float=-10.0, v_max:float=10.0, n_atoms:int=51):
    if distributional_rl:
        return dist_backup(target_ps, q_idx, rewards, dones, double_dqn, discount, state_value,
                           v_min, v_max, n_atoms)
    else:
        return value_backup(target_ps, q_idx, rewards, dones, double_dqn, discount, state_value)

def mixed_backup_with_graph(sa_all:List[Tuple[int, List[Tuple[int, int]]]],
                            freq_count_map:Dict[int, Dict[int, List[int]]],
                            rewards:List[float], dones:List[bool], next_states:List[int], counts:List[int],
                            i2q:Dict[int, torch.Tensor], i2max:Dict[int, torch.Tensor], actions:List[int],
                            double_dqn:bool, distributional_rl:bool, discount:float,
                            v_min: float=-10.0,
                            v_max: float=10.0,
                            n_atoms: int=51):
    targets = []
    for n, i_and_sa in enumerate(sa_all):
        source_idx: int = i_and_sa[0]
        sa: List[Tuple[int, int]] = i_and_sa[1]
        i2v: Dict[int, torch.Tensor] = {}
        for state_and_action in reversed(sa):
            state: int = state_and_action[0]
            action: int = state_and_action[1]
            v = 0
            overall_count = 0
            for freq_idx in freq_count_map[state][action]:  # loop through different possibilities
                r, d, next_state, count = rewards[freq_idx], dones[freq_idx], next_states[freq_idx], counts[freq_idx]
                overall_count += count

                if next_state in i2v:
                    next_value = i2v[next_state]
                    v += count * backup(distributional_rl, next_value, torch.zeros(0), torch.tensor(r), torch.tensor(d),
                                              double_dqn=double_dqn, discount=discount,
                                              state_value=True, v_min=v_min, v_max=v_max, n_atoms=n_atoms)
                else:
                    next_value = i2q[next_state]
                    idx = i2max[next_state] if double_dqn else torch.zeros(0)
                    v += count * backup(distributional_rl, next_value, idx, torch.tensor(r), torch.tensor(d),
                                              double_dqn=double_dqn, discount=discount,
                                              state_value=False)

            i2v[state] = torch.tensor(v / overall_count)
        source_action = actions[n]
        target = 0
        overall_count = 0
        for freq_idx in freq_count_map[source_idx][source_action]:  # loop through different possibilities
            r, d, next_state, count = rewards[freq_idx], dones[freq_idx], next_states[freq_idx], counts[freq_idx]
            overall_count += count
            if not d and next_state not in i2v:
                overall_count -= count
            else:
                if d:
                    next_value = torch.zeros_like(i2v[source_idx])
                else:
                    next_value = i2v[next_state]
                target += count * backup(distributional_rl, next_value, torch.zeros(0), torch.tensor(r), torch.tensor(d),
                                               double_dqn=double_dqn, discount=discount,
                                               state_value=True)

        targets.append(target/torch.tensor(overall_count))
    return torch.stack(targets)
