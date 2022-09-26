import gym
import numpy as np
from gym_minigrid import wrappers as wrappers
import pickle
import wandb
import os
import glob
from rainbow.agent import Agent as RainbowAgent
import torch
import torch.nn.functional as F
from core.util import get_gym_env, pad, test
from random import choices

def get_obs_shape(args):
    args.game = args.env
    if "MiniGrid" in args.env:
        env = wrappers.FullyObsWrapper(gym.make(args.env))
        shape = env.reset()["image"].shape
    elif "Minatar" in args.env:
        env, _ = get_gym_env(args)
        shape = env.reset().shape
    return shape


def q2v(q, policy, epsilon):
    if policy == "greedy":
        return np.max(q)
    elif policy == "epsilon_greedy":
        return np.max(q)*(1-epsilon) + np.sum(epsilon/np.sum(q.shape)*q)


def train_DQN(args, model, minibatch, device, optimizer, replay):
    model.train()
    if isinstance(minibatch[0][1], np.int32):
        a = torch.tensor(np.array([action for _, action, _, _,_ in minibatch]), dtype=torch.long, device=device)
    else:
        a = torch.stack([action for _, action,_,_,_ in minibatch])
    X = torch.stack([obs for obs, _, _, _, _ in minibatch])
    target = torch.tensor(np.array([t for _, _, t, _,_ in minibatch]), dtype=torch.float32, device=device)
    weights = torch.stack([weight for _, _, _, _, weight in minibatch])
    t_idx = np.array([idx for _, _,_,idx,_ in minibatch])

    values = model.online_net(X, None)
    sliced_values = values[range(0,len(minibatch)), a[None, :]]
    difference = sliced_values.flatten() - target.flatten()
    replay.update_priorities(t_idx, difference.abs().detach().cpu().numpy()+1e-4)
    loss = torch.square(difference)
    #loss = (values - target) ** 2
    #loss = torch.mean(torch.gather(loss, 1, a[None, :]))
    if args.importance_weight:
        loss = torch.mean(loss*weights)
    else:
        loss = torch.mean(loss)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss


def train_distill(args, model, minibatch, device, optimizer, seen_only):
    model.train()
    X = torch.tensor(np.array([obs for obs, _, _ in minibatch]), dtype=torch.float32, device=device)
    Y = torch.tensor(np.array([target for _, target,_ in minibatch]), dtype=torch.float32, device=device)
    seen = torch.tensor(np.array([seen for _, _,seen in minibatch]), dtype=torch.float32, device=device)

    if args.distill_target == "policy":
        prediction = model(dict(frame=X[None, :]))["policy_logits"][0]
        loss = F.kl_div(F.log_softmax(prediction), Y)
    else:
        if not args.disable_dist:
            Y = Y.clamp(min=model.Vmin, max=model.Vmax)  # Clamp between supported values
            # Compute L2 projection of Tz onto fixed support z
            Y = (Y - model.Vmin) / model.delta_z  # b = (Tz - Vmin) / Δz
            nb_actions = Y.shape[1]
            l, u = Y.floor().to(torch.int64), Y.ceil().to(torch.int64)
            # Fix disappearing probability mass when l = Y = u (Y is int)
            l[(u > 0) * (l == u)] -= 1
            u[(l < (model.atoms - 1)) * (l == u)] += 1
            l = l.view(-1)
            u = u.view(-1)
            # Distribute probability of Tz
            m = Y.new_zeros(args.batch_size, nb_actions, args.atoms)
            offset = torch.linspace(0, ((args.batch_size*nb_actions - 1) * args.atoms),
                                    args.batch_size*nb_actions).to(l)
            Y = Y.view(-1)
            m.view(-1).index_add_(0, (l + offset).view(-1),
                                  (1 * (u.float() - Y)).view(-1))  # m_l = m_l + p(s_t+n, a*)(u - b)
            m.view(-1).index_add_(0, (u + offset).view(-1),
                                  (1 * (Y - l.float())).view(-1))  # m_u = m_u + p(s_t+n, a*)(b - l)
            log_ps = model.online_net(X, None, log=True)  # Log probabilities log p(s_t, ·; θonline)
            loss = -torch.sum(m * log_ps, 2)# Cross-entropy loss (minimises DKL(m||p(s_t, a_t)))

        else:
            values = model.online_net(X, None)
            loss = (values-Y)**2
    if seen_only == "action":
        loss = loss*(seen>0)
    elif seen_only == "state":
        loss = torch.sum(loss, dim=-1)*(torch.sum(seen, dim=-1)>0)
    loss = torch.mean(loss)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss

def prepare_data(args, table, s2i, target_type, batch_size, SA=None):
    dataset = []

    indices = [i for i in list(np.random.randint(0, s2i.max, [batch_size]))]

    for i in indices:
        q = table[i]
        s = s2i.states[i]
        if target_type == "policy":
            action_dist = (q == np.max(q))
            action_dist = action_dist / np.sum(action_dist)
            dataset.append(s, action_dist, np.nan)
        elif target_type == "q":
            if SA is None:
                dataset.append((s.transpose([2, 0, 1]), q, np.nan))
            else:
                seen = SA[i] > 0
                dataset.append((s.transpose([2, 0, 1]), q, seen))
    return dataset

def graph_backup(buffer, table, env_steps, batch_size, s2i):
    dataset = []
    minibatch = buffer.sample(batch_size)
    for i in range(batch_size):
        index, s, a, r, s1, nonterminal, weight, step = tuple([item[i] for item in minibatch])
        idx_s = s2i.get_index(s)
        q = table[idx_s]
        dataset.append((s.transpose([2, 0, 1]), a, q[a], index, weight))
    return dataset

def graph_limited_backup(buffer, eval_Q, batch_size, f, discount, steps, nb_actions,
                         branching_limit, backup_target_update, aggregate_q, s2i):
    dataset = []
    minibatch = buffer.sample(batch_size)

    for i in range(batch_size):
        index, tree_idx, s_source, a_source, r, s1, nonterminal, weight, step_t = tuple([item[i] for item in minibatch])
        s_source_idx = s2i.get_index(s_source)
        new_s = {s_source_idx}
        sa = []
        for step in range(steps):
            new_trans = set()
            for s in new_s:
                for action in range(nb_actions):
                    if (s, action) in f:
                        for r, next_state in f[(s, action)]:  # loop though different possibilities
                            new_trans.add((s, action, r, next_state))
            if len(new_trans) > branching_limit:
                new_trans = list(new_trans)
                counts = [f[t[:2]][t[2:]] for t in new_trans]
                prob = counts
                new_trans = choices(new_trans, weights=prob, k=branching_limit)
                sa.extend([t[:2] for t in new_trans])
                new_s = set([t[-1] for t in new_trans])
            else:
                sa.extend([t[:2] for t in new_trans])
                new_s = set([t[-1] for t in new_trans])

        if backup_target_update:
            q_updated = eval_Q
        else:
            q_updated = eval_Q.copy()

        for state, action in reversed(sa):
            v = 0
            overall_count = 0
            for r, next_state in f[(state, action)]:  # loop though different possibilities
                count = f[(state, action)][(r, next_state)]
                overall_count += count
                done = True
                for a in range(nb_actions):
                    if (next_state, a) in f:
                        done = False
                v += count * (r + discount * aggregate_q(q_updated[next_state])*(1-done))
            q_updated[state, action] = v / overall_count
        dataset.append((s_source, a_source, q_updated[s_source_idx, a_source], tree_idx, weight))
        """
        if s_source[-2,-3, 0] == 10 and s_source[-2,-3, 2] == 1 and a_source==2:
            print(f" target before goal {q_updated[idx_s, a_source]}")
        if s_source[-2, -4, 0] == 10 and s_source[-2, -4, 2] == 1 and a_source == 2:
            print(f" target corner down {q_updated[idx_s, a_source]}")
        """
    return dataset

def graph_mixed_backup(buffer, eval_Q, batch_size, f, discount, steps, nb_actions,
                         branching_limit, backup_target_update, aggregate_q, s2i):
    dataset = []
    minibatch = buffer.sample(batch_size)

    for i in range(batch_size):
        index, tree_idx, s_source, a_source, r, s1, nonterminal, weight, step_t = tuple([item[i] for item in minibatch])
        s_source_idx = s2i.get_index(s_source)
        new_s = {s_source_idx}
        sa = []
        for step in range(steps):
            new_trans = set()
            for s in new_s:
                for action in range(nb_actions):
                    if (s, action) in f:
                        for r, next_state in f[(s, action)]:  # loop though different possibilities
                            new_trans.add((s, action, r, next_state))
            if len(new_trans) > branching_limit:
                new_trans = list(new_trans)
                counts = [f[t[:2]][t[2:]] for t in new_trans]
                prob = counts
                new_trans = choices(new_trans, weights=prob, k=branching_limit)
                sa.extend([t[:2] for t in new_trans])
                new_s = set([t[-1] for t in new_trans])
            else:
                sa.extend([t[:2] for t in new_trans])
                new_s = set([t[-1] for t in new_trans])

        if backup_target_update:
            q_updated = eval_Q
        else:
            q_updated = eval_Q.copy()

        i2v= {}
        for state_and_action in reversed(sa):
            state: int = state_and_action[0]
            action: int = state_and_action[1]
            v = 0
            overall_count = 0
            for r, next_state in f[(state,action)]:  # loop through different possibilities
                count = f[(state, action)][(r, next_state)]
                overall_count += count
                done = True
                for a in range(nb_actions):
                    if (next_state, a) in f:
                        done = False

                if next_state in i2v:
                    next_value = i2v[next_state]
                    v += count * (r + discount * next_value * (1-done))
                else:
                    next_value = eval_Q[next_state]
                    v += count * (r + discount * (aggregate_q(next_value) * (1-done)))

            i2v[state] = torch.tensor(v / overall_count)
        target = 0
        overall_count = 0
        for r, next_state in f[(s_source_idx,a_source.item())]:  # loop through different possibilities
            count = f[(s_source_idx, a_source.item())][(r, next_state)]
            overall_count += count
            done = True
            for a in range(nb_actions):
                if (next_state, a) in f:
                    done = False

            if not done and next_state not in i2v:
                overall_count -= count
            else:
                if done:
                    next_value = 0
                else:
                    next_value = i2v[next_state]

                target += count * (r + discount * (next_value * (1 - done)))
        dataset.append((s_source, a_source, target, tree_idx, weight))
    return dataset


def tree_backup(buffer, table, env_steps, batch_size, steps, discount, aggregate_q, s2i):
    dataset = []
    minibatch = buffer.sample(batch_size)
    for i in range(batch_size):
        source_t_index, tree_idx, s_source, a_source, r, s1_source, non_t, weight, step = tuple([item[i] for item in minibatch])
        max_steps = 0

        for step in range(steps):
            if source_t_index+step>=buffer.transitions.index:
                break
            s, a, r, s1, non_t, step_t = buffer._get_sample_with_idx([source_t_index+step], buffer.n, False)
            max_steps += 1
            if not non_t:
                break

        future_index = s2i.get_index(buffer._get_sample_with_idx([source_t_index + max_steps-1], buffer.n, False)[3])
        updated_q = table[future_index].copy()

        for step in reversed(range(max_steps)):
            s, a, r, s1, non_t, step_t = buffer._get_sample_with_idx([source_t_index + step], buffer.n, False)
            idx_s = s2i.get_index(s)
            target = r + discount * (aggregate_q(updated_q) * non_t)  # last step backup
            updated_q = table[idx_s].copy()
            updated_q[a] = target

        dataset.append((s, a, target, tree_idx, weight))
    return dataset


def n_step_Q_backup(buffer, table, env_steps, batch_size, steps, discount, aggregate_q, s2i):
    dataset = []
    minibatch = buffer.sample(batch_size)
    for i in range(batch_size):
        source_t_index, tree_idx, s_source, a_source, r, s1_source, non_t, weight, step = tuple([item[i] for item in minibatch])
        max_steps = 0
        n_step_return = 0

        for step in range(steps):
            if source_t_index+step>=buffer.transitions.index:
                break
            s, a, r, s1, non_t, step_t = buffer._get_sample_with_idx([source_t_index+step], buffer.n, False)
            max_steps += 1
            n_step_return += r*(discount**step)
            if not non_t:
                break

        future_index = s2i.get_index(buffer._get_sample_with_idx([source_t_index + max_steps-1],
                                                                 buffer.n, False)[3])
        target = n_step_return + discount**steps *(aggregate_q(table[future_index])*non_t) # last step backup

        dataset.append((s, a, target, tree_idx, weight))
    return dataset


def init_distill(args, table):
    dataset = []
    obs_shape = get_obs_shape(args)
    padded_width = -1 if "MiniGrid" not in args.env else 32
    device = args.device

    dataset = prepare_data(args, table, None)

    nb_actions = dataset[0][1].shape[0]
    obs_shape = dataset[0][0].shape
    model = RainbowAgent(args, nb_actions, nb_features=obs_shape[-1])
    optimizer = model.optimiser
    return model, optimizer, dataset, device

def distill(args, total_steps, table, eval=False):
    nb_steps = 0
    batch_size = args.batch_size
    model, optimizer, dataset, device = init_distill(args, table)

    while nb_steps < total_steps:
        loss = train_distill(args, model, dataset, batch_size, device, optimizer)

        if nb_steps % 100 == 0:
            if eval:
                test_returns = test(args, model, args.seed, all_seeds=True)
                wandb.log(dict(loss=loss, mean_test_returns=np.mean(test_returns)), step=nb_steps)
                print(f"step: {nb_steps}, loss: {loss}, mean_test_returns: {np.mean(test_returns)}")
        nb_steps += 1
    print(f"distill loss: {loss}")
    return model

def distill_from_file(args):
    terms = args.id.split("-")
    group = terms[0] + "-" + terms[1]
    source_dir = os.path.expandvars(os.path.expanduser(f"~/logs/ava/{group}-*/qtable.pkl"))
    results_dir = os.path.expandvars(os.path.expanduser(os.path.join('~/logs/distill', group+"-distill")))
    wandb.init(config=args, project="Distill", name=args.id, group=group, dir=results_dir)

    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    merged_table = {}
    nb_train_seeds = 0
    for filename in glob.glob(source_dir):
        with open(filename, 'rb') as f:
            merged_table = {**merged_table, **pickle.load(f)}
        nb_train_seeds += 1
    distill(args, args.num_steps, table=merged_table, eval=True)


def format_input(bytes, raw_shape, padded_width=-1):
    """
    format and padding the input bytes array
    """
    array = np.frombuffer(bytes, dtype=np.uint8)
    try:
        array = array.reshape(raw_shape).astype(dtype=np.float32)
    except Exception as e:
        array = np.frombuffer(bytes, dtype=np.int16)
        array = array.reshape(raw_shape).astype(dtype=np.float32)
    if padded_width == -1:
        return array
    else:
        img = pad(array, padded_width)
        return img

