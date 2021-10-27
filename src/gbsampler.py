import numpy as np
import torch
from random import choices
from rlpyt.utils.tensor import select_at_indexes, valid_mean
from rlpyt.samplers.collectors import (DecorrelatingStartCollector,
    BaseEvalCollector)
from rlpyt.agents.base import AgentInputs
from rlpyt.utils.buffer import (torchify_buffer, numpify_buffer, buffer_from_example,
    buffer_method)
from typing import Optional, Dict, Tuple, List

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

    def get_index(self, state):
        if isinstance(state, torch.Tensor):
            np_state = state.cpu().numpy()
        else:
            np_state = state
        key = hashing(np_state, self.hashing_method)
        if key in self.data:
            return self.data[key]
        else:
            return None

    def get_indexes(self, states):
        return [self.get_index(state) for state in states]

    def get_states(self, indexs):
        return [self.states[i] for i in indexs]

    def append_state(self, state):
        if isinstance(state, torch.Tensor):
            np_state = state.cpu().numpy()
        else:
            np_state = state
        key = hashing(np_state, self.hashing_method)
        if key not in self.data:
            self.data[key] = len(self.data)
            self.states.append(state)
            self.max += 1
            return False
        else:
            return True

class TransitionFreq():
    def __init__(self):
        self.freq = {}
        self.transition_count = 0

    def append(self, s, a, r, d, s1):
        self.transition_count += 1
        s = int(s)
        a = int(a)
        s1 = int(s1)
        if s not in self.freq:
            self.freq[s] = {}
        if a not in self.freq[s]:
            self.freq[s][a] = {}
        if (r, d, s1) not in self.freq[s][a]:
            self.freq[s][a][(r, d, s1)] = 0
        self.freq[s][a][(r, d, s1)] += 1


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

        self.s2i.append_state(agent_inputs.observation[0])
        s_idx = self.s2i.get_index(agent_inputs.observation[0])

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
            self.s2i.append_state(o)
            s1_idx = self.s2i.get_index(o)
            self.transition_freq.append(s_idx, action[b], r, d, s1_idx)
            s_idx = s1_idx

            traj_infos[b].step(observation[b], action[b], r, d, agent_info[b],
                env_info)
            if getattr(env_info, "traj_done", d):
                completed_infos.append(traj_infos[b].terminate(o))
                traj_infos[b] = self.TrajInfoCls()
                o = env.reset()

                self.s2i.append_state(o)
                s_idx = self.s2i.get_index(o)
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

def graph_limited_backup(agent, freq, states, s2i, discount, breath, depth, double, dist, one_step_backup):
    targets = []
    source_idxes = s2i.get_indexes(states)
    sa_all = []
    target_states = []

    for source_idx in source_idxes:
        new_s = {source_idx}
        sa = []
        target_states.append(source_idx)
        for step in range(depth):
            new_trans = set()
            for s in new_s:
                if s in freq.freq:
                    for action in freq.freq[s].keys():
                        for r, d, next_state in freq.freq[s][action]:  # loop though different possibilities
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
                       depth, double, dist, one_step_backup):
    source_idxes = s2i.get_indexes(states)
    targets = []
    sa_all = []
    target_states = []

    for n, source_idx in enumerate(source_idxes):
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
                            new_trans.add((s, action, r, d, next_state))
            target_states.extend([t[-1] for t in new_trans])
            if step == 0: # only expand source action in source state
                new_trans = set()
                for r, d, next_state in freq.freq[source_idx][actions[n]]:  # loop though different possibilities
                    new_trans.add((source_idx, actions[n], r, d, next_state))

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

    actions = [a for a in actions]
    result = mixed_backup_with_graph(sa_all, freq.freq, i2q, i2max, actions, double, discount)
    return result


def mixed_backup_with_graph(sa_all:List[Tuple[int, List[Tuple[int]]]],
                            freq_count:Dict[int, Dict[int, Dict[Tuple[float, bool, int], int]]],
                            i2q:Dict[int, torch.Tensor], i2max:Dict[int, torch.Tensor], actions:List[int],
                            double_dqn:bool, discount:float):
    targets = []
    for n,(source_idx, sa) in enumerate(sa_all):
        i2v = {}
        for state, action in reversed(sa):
            v = 0
            overall_count = 0
            for r, d, next_state in freq_count[state][action]:  # loop through different possibilities
                count = freq_count[state][action][(r, d, next_state)]
                overall_count += count

                if next_state in i2v:
                    next_value = i2v[next_state]
                    v += count * value_backup(next_value, torch.zeros(0), torch.tensor(r), torch.tensor(d),
                                              double_dqn=double_dqn, discount=discount,
                                              state_value=True)
                else:
                    next_value = i2q[next_state]
                    idx = i2max[next_state] if double_dqn else torch.zeros(0)
                    v += count * value_backup(next_value, idx, torch.tensor(r), torch.tensor(d),
                                              double_dqn=double_dqn, discount=discount,
                                              state_value=False)

            i2v[state] = v / overall_count
        source_action = actions[n]
        target = 0
        overall_count = 0
        for r, d, next_state in freq_count[source_idx][source_action]:  # loop through different possibilities
            count = freq_count[source_idx][source_action][(r, d, next_state)]
            overall_count += count
            if not d and next_state not in i2v:
                overall_count -= count
            else:
                if d:
                    next_value = torch.zeros_like(i2v[source_idx])
                else:
                    next_value = i2v[next_state]
                target += count * value_backup(next_value, torch.zeros(0), torch.tensor(r), torch.tensor(d),
                                               double_dqn=double_dqn, discount=discount,
                                               state_value=True)

        targets.append(target/overall_count)
    return torch.stack(targets)


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
                double_dqn:bool, v_min:float, v_max:float, n_atoms:int, discount:float, state_value:bool=False):
    if len(rewards.shape) == 0:
        target_ps = target_ps.unsqueeze(0)
        rewards = rewards.unsqueeze(0)
        dones = dones.unsqueeze(0)
        if double_dqn and not state_value:
            q_idx = q_idx.unsqueeze(0)

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

    if len(rewards.shape) == 1:
        target_p = target_p[0]
    return target_p