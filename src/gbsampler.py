import numpy as np
import torch
from random import choices
from rlpyt.samplers.collectors import (DecorrelatingStartCollector,
    BaseEvalCollector)
from rlpyt.agents.base import AgentInputs
from rlpyt.utils.buffer import (torchify_buffer, numpify_buffer, buffer_from_example,
    buffer_method)

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

    def append(self, s, a, r, s1):
        self.transition_count += 1
        s = int(s)
        a = int(a)
        s1 = int(s1)
        if s not in self.freq:
            self.freq[s] = {}
        if a not in self.freq[s]:
            self.freq[s][a] = {}
        if (r, s1) not in self.freq[s][a]:
            self.freq[s][a][(r,s1)] = 0
        self.freq[s][a][(r,s1)] += 1


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

            self.s2i.append_state(o)
            s1_idx = self.s2i.get_index(o)
            self.transition_freq.append(s_idx, action[b], r, s1_idx)
            s_idx = s1_idx

            traj_infos[b].step(observation[b], action[b], r, d, agent_info[b],
                env_info)
            if getattr(env_info, "traj_done", d):
                completed_infos.append(traj_infos[b].terminate(o))
                traj_infos[b] = self.TrajInfoCls()
                o = env.reset()
            if d:
                self.agent.reset_one(idx=b)
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

def graph_limited_backup(agent, freq, states, s2i, discount, breath, depth, aggregate_q=q2v):
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
                        for r, next_state in freq.freq[s][action]:  # loop though different possibilities
                            new_trans.add((s, action, r, next_state))
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
    qs = agent.target(states_array, None, None)
    i2q = {s: qs[i].cpu().numpy() for i, s in enumerate(target_states)}

    for source_idx, sa, in sa_all:
        i2q_temp = {}
        for state, action in reversed(sa):
            if state not in i2q_temp:
                i2q_temp[state] = i2q[state].copy()
            v = 0
            overall_count = 0
            for r, next_state in freq.freq[state][action]:  # loop though different possibilities
                count = freq.freq[state][action][(r, next_state)]
                overall_count += count
                if next_state in freq.freq:
                    if next_state not in i2q_temp:
                        i2q_temp[next_state] = i2q[next_state].copy()
                    v += count * (r + discount * aggregate_q(i2q_temp[next_state]))
                else:
                    v += count * r
            i2q_temp[state][action] = v / overall_count
        targets.append(i2q_temp[source_idx])
    return torch.tensor(np.array(targets))


def graph_mixed_backup(agent, freq, states, actions, s2i, discount, breath, depth, aggregate_q=q2v):
    targets = []
    source_idxes = s2i.get_indexes(states)
    sa_all = []
    target_states = []

    for n, source_idx in enumerate(source_idxes):
        new_s = {source_idx}
        #sa = [(source_idx, actions[n])]
        sa = []
        target_states.append(source_idx)
        for step in range(depth):
            new_trans = set()
            if depth == 0: # only expand source action in source state
                for r, next_state in freq.freq[source_idx][actions[n]]:  # loop though different possibilities
                    new_trans.add((source_idx, actions[n], r, next_state))
            else:
                for s in new_s:
                    if s in freq.freq:
                        for action in freq.freq[s].keys():
                            for r, next_state in freq.freq[s][action]:  # loop though different possibilities
                                new_trans.add((s, action, r, next_state))
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
    qs = agent.target(states_array, None, None)
    i2q = {s: qs[i].cpu().numpy() for i, s in enumerate(target_states)}

    for n,(source_idx, sa) in enumerate(sa_all):
        i2v = {}
        for state, _ in reversed(sa):
            v = 0
            overall_count = 0
            for action in freq.freq[state]:
                for r, next_state in freq.freq[state][action]:  # loop through different possibilities
                    count = freq.freq[state][action][(r, next_state)]
                    overall_count += count
                    if next_state in freq.freq:
                        if next_state in i2v:
                            v += count * (r + discount * i2v[next_state])
                        else:
                            v += aggregate_q(i2q[next_state])
                    else:
                        v += count * r
            i2v[state] = v / overall_count
        source_action = actions[n]
        target = 0
        overall_count = 0
        for r, next_state in freq.freq[source_idx][source_action]:  # loop through different possibilities
            count = freq.freq[source_idx][source_action][(r, next_state)]
            overall_count += count
            if next_state in freq.freq:
                if next_state in i2v:
                    target += count * (r + discount * i2v[next_state])
                else: # dealing with nodes that are not expanded
                    overall_count -= count
            else:
                target += count * r
        targets.append(target/overall_count)
    return torch.tensor(np.array(targets))