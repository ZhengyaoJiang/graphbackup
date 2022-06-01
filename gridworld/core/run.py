import numpy as np
import argparse
import copy
import pandas as pd
import os
from core.backup import distill_from_file, train_distill, prepare_data, graph_backup, \
    tree_backup, train_DQN, graph_limited_backup, q2v, n_step_Q_backup, graph_mixed_backup
from core.util import get_nb_actions, get_gym_env, pad, test
from core.memory import State2Index
import torch
from rainbow.agent import Agent as RainbowAgent
from rainbow.memory import ReplayMemory
from core.util import Timings

torch.set_num_threads(2)

parser = argparse.ArgumentParser()
parser.add_argument('--id', type=str, default='T-0-0', help='Experiment ID')
parser.add_argument('--exp_group', type=str, default='test', help='exp_group')
parser.add_argument("--mode", type=str, default="train", choices=["train", "distill"])
parser.add_argument("--num_steps", type=int, default=1000000,
                        help="number of steps")
parser.add_argument("--lr", type=float, default=1.,
                        help="learning rate for training")
parser.add_argument("--full_obs", type=int, default=0,
                        help="whether to use full observation")
parser.add_argument("--full_obs_ir", type=int, default=0,
                        help="whether to use full observation for intrinsic reward")
parser.add_argument("--env", type=str, default=None,
                        help="environmet for training")
parser.add_argument("--initialization", type=str, default=None,
                        help="which initialization to use for Q",
                        choices=["optimistic", "zero", "rainbow", "distilled", "distilled_opt", "distilled_action",
                                 "distilled_opt_gated", "optimistic_gated"])
parser.add_argument("--exploration", type=str, default="epsilon_greedy",
                        choices=["greedy", "epsilon_greedy"])
parser.add_argument("--with_steps", action="store_true")
parser.add_argument("--seed", type=int, default=-1,
                        help="what seed to use for the environment")
parser.add_argument("--seed_range", type=int, default=1e10,
                    help="what seed to use for the environment")
parser.add_argument("--epsilon", type=float, default=0.02)
parser.add_argument("--buffer_length", type=int, default=1000000)
parser.add_argument("--update_period", type=int, default=1000)
parser.add_argument("--reinit_period", type=int, default=1000)
parser.add_argument("--eval_period", type=int, default=1000)

parser.add_argument("--buffer_key", type=str, default="transition", choices=["state", "transition"])

# Atari Setting
parser.add_argument('--history-length', type=int, default=4, metavar='T', help='Number of consecutive states processed')
parser.add_argument('--reward-clip', type=int, default=1, metavar='VALUE', help='Reward clipping (0 to disable)')
parser.add_argument('--max-episode-length', type=int, default=int(108e3), metavar='LENGTH', help='Max episode length in game frames (0 to disable)')
parser.add_argument("--no_op", action="store_true")

# Distill Setting
parser.add_argument("--distill_target", type=str, default="q", choices=["q", "policy"])

# NN initialization setting
parser.add_argument("--learn-start", type=int, default=5000)
parser.add_argument('--architecture', type=str, default='canonical', choices=['canonical', 'data-efficient', 'gridworld'], metavar='ARCH', help='Network architecture')
parser.add_argument('--hidden-size', type=int, default=512, metavar='SIZE', help='Network hidden size')
parser.add_argument('--noisy-std', type=float, default=0.1, metavar='σ', help='Initial standard deviation of noisy linear layers')
parser.add_argument('--atoms', type=int, default=51, metavar='C', help='Discretised size of value distribution')
parser.add_argument('--V-min', type=float, default=-10, metavar='V', help='Minimum of value distribution support')
parser.add_argument('--V-max', type=float, default=10, metavar='V', help='Maximum of value distribution support')
parser.add_argument('--model', type=str, metavar='PARAMS', help='Pretrained model (state dict)')
parser.add_argument('--memory-capacity', type=int, default=int(1e6), metavar='CAPACITY', help='Experience replay memory capacity')
parser.add_argument('--replay-frequency', type=int, default=4, metavar='k', help='Frequency of sampling from memory')
parser.add_argument('--distill_steps', type=int, default=1, metavar='k', help='Frequency of sampling from memory')
parser.add_argument('--priority-exponent', type=float, default=0.5, metavar='ω', help='Prioritised experience replay exponent (originally denoted α)')
parser.add_argument('--priority-weight', type=float, default=0.4, metavar='β', help='Initial prioritised experience replay importance sampling weight')
parser.add_argument('--multi-step', type=int, default=3, metavar='n', help='Number of steps for multi-step return')
parser.add_argument('--discount', type=float, default=0.99, metavar='γ', help='Discount factor')
parser.add_argument('--target-update', type=int, default=int(8e3), metavar='τ', help='Number of steps after which to update target network')
parser.add_argument('--learning-rate', type=float, default=0.0000625, metavar='η', help='Learning rate')
parser.add_argument('--adam-eps', type=float, default=1.5e-4, metavar='ε', help='Adam epsilon')
parser.add_argument('--batch-size', type=int, default=32, metavar='SIZE', help='Batch size')
parser.add_argument('--norm-clip', type=float, default=10, metavar='NORM', help='Max L2 norm for gradient clipping')
parser.add_argument("--disable_dist", action="store_true")
parser.add_argument("--disable_noisy", action="store_true")
parser.add_argument("--disable_duelling", action="store_true")
parser.add_argument("--disable_double", action="store_true")
parser.add_argument('--T-max', type=int, default=int(50e6), metavar='STEPS', help='Number of training steps (4x number of frames)')

parser.add_argument("--disable_cuda", action="store_true")

parser.add_argument("--seq_solve", action="store_true")
parser.add_argument("--debug", action="store_true")
parser.add_argument("--action", type=str, default="raw", choices=["raw", "move_dir"])

parser.add_argument("--ram", action="store_true")
parser.add_argument("--log_table", action="store_true")
parser.add_argument("--net_code", type=str, default="2c2f")
parser.add_argument("--hashing", type=str, default="exact")

parser.add_argument("--seen_only", type=str, default="state", choices=["none", "state", "action"])
parser.add_argument("--max", type=str, default="table", choices=["table", "online"])
parser.add_argument("--buffer_sample", type=str, default="uniform", choices=["full", "uniform", "linear"])
parser.add_argument("--sampled_buffer_size", type=int, default=1000)

parser.add_argument("--backup_target", type=str, default="graph", choices=["graph", "tree", "graph-vi", "graph-limited",
                                                                           "n-step-Q", "graph-mixed"])

parser.add_argument('--branching_limit', type=int, default=50, metavar='SIZE', help='Batch size')
parser.add_argument('--backup_target_update', action="store_true")
parser.add_argument('--importance_weight', action="store_true")
parser.add_argument("--target_policy", default="greedy", choices=["greedy", "epsilon_greedy"])

args = parser.parse_args()
if (not args.disable_cuda) and torch.cuda.is_available():
    args.device = torch.device("cuda:" + str(torch.cuda.current_device()))
else:
    args.device = torch.device("cpu")
args.disable_steps = not args.with_steps


def init_table(s, idx_s, Q, eval_Q, env, dqn, learn_start, episode_steps, SA):
    if args.initialization == 'zero':
        eval_Q[idx_s] = np.zeros(get_nb_actions(env))
    elif args.initialization == 'random':
        eval_Q[idx_s] = np.random.randn(get_nb_actions(env))
    elif args.initialization in ["distilled", "rainbow"]:
        eval_Q[idx_s] = dqn.evaluate_q(torch.tensor(s, device=args.device, dtype=torch.float32).permute(2, 0, 1),
                                       episode_steps, use_target=True).cpu().numpy()
    else:
        assert False


def train():
    terms = args.id.split("-")
    group = terms[0] + "-" + terms[1]
    results_dir = os.path.expandvars(os.path.expanduser(os.path.join('~/locallogs/ava', args.id)))
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    if "policy" in args.exploration:
        args.distill_target = "policy"

    env, max_steps = get_gym_env(args)
    nb_actions = get_nb_actions(env)
    aggregate_q = lambda q: q2v(q, args.target_policy, args.epsilon)
    # Initialize table with all zeros to be uniform

    s2idx = State2Index(args.hashing)
    Q = np.zeros([args.num_steps, nb_actions], dtype=np.float32)
    eval_Q = np.zeros([args.num_steps, nb_actions], dtype=np.float32)
    SA = np.zeros([args.num_steps, nb_actions], dtype=np.int32)
    f = {}

    # Learning parameters
    lr = args.lr
    gamma = args.discount
    num_steps = args.num_steps
    # creating lists to contain total rewards and steps per episode
    first_reward = False
    rList = []
    record = dict(step=[], mean_episode_return=[], number_of_states=[])
    i = 0
    env_steps = 0

    if torch.cuda.is_available() and not args.disable_cuda:
        args.device = torch.device('cuda')
    else:
        args.device = torch.device('cpu')

    if "MiniGrid" in args.env:
        args.history_length = 1
        args.V_min=0
        args.V_max=1
        env_type = "gridworld"
    elif "Minatar" in args.env:
        args.history_length = 1
        env_type = "minatar"
    else:
        env_type = "atari"

    if "MiniGrid" in args.env:
        image_size = 32
        nb_features = 3
        args.architecture = "gridworld"
    elif "Minatar" in args.env:
        image_size = 10
        nb_features = env.observation_space.shape[-1]
        args.architecture = "minatar"
    elif "procgen" in args.env:
        image_size = 64
        nb_features = 3
    else:
        image_size = 84
        nb_features = 1

    nb_actions = env.action_space.n

    if args.initialization in ["distilled"]:
        dqn = RainbowAgent(args, nb_actions, nb_features)
        optimizer = dqn.optimiser
    else:
        dqn = None
        novelty_model = None

    args.disable_steps = not args.with_steps
    current_seed = args.seed
    max_seed = args.seed + args.seed_range
    last_solved_steps = 0
    learn_start = False

    timings = Timings()  # Keep track of how fast things are.
    replay_buffer = ReplayMemory(args, args.memory_capacity, image_size, nb_features, env_type, 1)

    while (not args.seq_solve and env_steps < num_steps) or (args.seq_solve and current_seed < max_seed):
        i += 1
        rAll = 0
        d = False
        episode_steps=0
        #Reset environment and get first new observation
        if args.seed > -1:
            env.seed(current_seed)
        else:
            env.seed(np.random.randint(99999))
        if args.seq_solve and args.debug:
            end_image=env.render(mode="rgb_array")
        s_ir = env.reset()
        s = copy.deepcopy(s_ir)
        if "MiniGrid" in args.env:
            s = pad(s)
        elif "Minatar" not in args.env:
            s = s.cpu().numpy()
        if args.seq_solve and args.debug:
            restart_image=env.render(mode="rgb_array")

        seen_state = s2idx.append_state(s)


        idx_s = s2idx.get_index(s)

        if not seen_state:
            init_table(s, idx_s, Q, eval_Q, env, dqn, learn_start, episode_steps, SA)
        timings.time("init_episode")

        #Q-Learning Table algorithm
        while True:
            env_steps += 1
            episode_steps += 1
            # Update the S dict
            # Update heatmap
            # agent_pos = np.where(s == 10)
            # heatmap[agent_pos[0][0], agent_pos[1][0]] += 1

            if args.debug:
                env.render()


            # Choose an action by greedily(with noise) picking from Q table
            if not learn_start:
                a = np.random.randint(get_nb_actions(env))
            elif args.exploration == "epsilon_greedy":
                #Break equal tie
                q = dqn.evaluate_q(torch.tensor(s, device=args.device, dtype=torch.float32).permute(2, 0, 1),
                               episode_steps).cpu().numpy()

                Qmax = np.max(q)
                if args.debug:
                    print(q)
                    print(np.argmax(q))
                p = (q == Qmax).astype(float)/np.sum(q == Qmax).astype(float)
                if np.random.rand() < args.epsilon:
                    a = np.random.randint(get_nb_actions(env))
                else:
                    a = np.random.choice(np.arange(get_nb_actions(env)), p=p)
            elif args.exploration == "greedy":
                q = dqn.evaluate_q(torch.tensor(s, device=args.device, dtype=torch.float32).permute(2, 0, 1),
                                   episode_steps).cpu().numpy()
                Qmax = np.max(q)
                p = (q == Qmax).astype(float)/np.sum(q == Qmax).astype(float)
                a = np.random.choice(np.arange(get_nb_actions(env)), p=p)
            timings.time("inference")

            #Get new state and reward from environment
            s1_ir,r,d,*_ = env.step(a)

            timings.time("act")
            if args.reward_clip > 0:
                if args.disable_steps and env_type == "gridworld":
                    r = float(r > 0)
                if env_type == "atari":
                    r = max(min(r, args.reward_clip), -args.reward_clip)  # Clip rewards

            s1 = copy.deepcopy(s1_ir)
            if "MiniGrid" in args.env:
                s1 = pad(s1)
            elif "Minatar" not in args.env:
                s1 = s1.cpu().numpy()
            #Update Q-Table with new knowledge

            seen_s1 = s2idx.append_state(s1)
            idx_s1 = s2idx.get_index(s1)

            SA[idx_s][a] += 1
            if args.backup_target in ["graph-vi", "graph-limited", "graph-mixed"]:
                if not((idx_s, a) in f):
                    f[(idx_s, a)] = {}
                if (r, idx_s1) in f[(idx_s, a)]:
                    f[(idx_s, a)][r, idx_s1] += 1
                else:
                    f[(idx_s, a)][r, idx_s1] = 1

            if not seen_s1:
                init_table(s1, idx_s1, Q, eval_Q, env, dqn, learn_start, episode_steps, SA)
            timings.time("init_table")

            replay_buffer.append(torch.tensor(s, device=args.device).permute([2, 0, 1]),
                                 torch.tensor(a, device=args.device),
                                 torch.tensor(r, device=args.device),
                                 torch.tensor(d, device=args.device),
                                 torch.tensor(episode_steps-1, device=args.device))    # Append transition to memory
            #index_buffer.append((s, a, s1, r, d))
            #if len(index_buffer) > args.buffer_length:
            #    del index_buffer[0]

            #time.sleep(0.1)
            if not first_reward and r>0:
                print(f"get first reward {r} after {env_steps} steps")
                first_reward = True
            if env_steps>args.learn_start:
                learn_start = True
            rAll += r
            s = s1
            s_ir = s1_ir
            idx_s = idx_s1

            if env_steps % args.update_period == 0 and env_steps>args.sampled_buffer_size:

                if args.backup_target=="graph":
                    sampled_buffer = replay_buffer.sample(args.sampled_buffer_size)

                    for j, (_, state, action, reward, next_state, nonterminal, weight, step) in enumerate(sampled_buffer):
                        # reward clipping to reduce the effects of reward shaping
                        lr = args.lr #* 0.9995**j
                        state = s2idx.get_index(state)
                        next_state = s2idx.get_index(next_state)
                        reward = reward if "Minatar" in args.env else float(reward > 0)-float(reward < 0)
                        # TODO: max with online net
                        target_value = aggregate_q(eval_Q[next_state])
                        eval_Q[state][action] = eval_Q[state][action] + lr * (
                                reward + gamma*target_value*nonterminal - eval_Q[state][action])
                    # np.save('heatmap_'+args.env+'_'+str(i)+'_'+str(args.full_obs_ir)+'_'+args.ir+'_'+args.initialization+'.npy', heatmap)
                elif args.backup_target=="graph-vi":
                    if args.buffer_sample == "full":
                        sa = f.keys()
                    elif args.buffer_sample == "uniform":
                        indexes = np.random.randint(0, len(f.keys()), size=[args.sampled_buffer_size])
                        keys = list(f.keys())
                        sa = [keys[i] for i in indexes]
                    for state, action in sa:
                        v = 0
                        overall_count = 0
                        for r, next_state in f[(state, action)]: # loop though different possibilities
                            count = f[(state, action)][(r, next_state)]
                            overall_count += count
                            v += count*(r + args.discount*np.max(eval_Q[next_state]))
                        eval_Q[state, action] = v / overall_count
                timings.time("update table")


            if learn_start:
                if args.initialization in ["distilled"]:
                    if env_steps % args.replay_frequency == 0:
                        for _ in range(args.distill_steps):
                            if args.buffer_key == "transition":
                                if args.backup_target in ["graph", "graph-vi"]:
                                    dataset = graph_backup(replay_buffer, eval_Q, env_steps,
                                                                      args.batch_size, s2idx)
                                elif args.backup_target in ["graph-limited"]:
                                    dataset = graph_limited_backup(replay_buffer, eval_Q,
                                                           args.batch_size, f, args.discount, args.multi_step,
                                                                   nb_actions, args.branching_limit, args.backup_target_update,
                                                                   aggregate_q, s2idx)
                                elif args.backup_target in ["graph-mixed"]:
                                    dataset = graph_mixed_backup(replay_buffer, eval_Q,
                                                                   args.batch_size, f, args.discount, args.multi_step,
                                                                   nb_actions, args.branching_limit,
                                                                   args.backup_target_update,
                                                                   aggregate_q, s2idx)
                                elif args.backup_target == "tree":
                                    dataset = graph_limited_backup(replay_buffer, eval_Q,
                                                                   args.batch_size, f, args.discount, args.multi_step,
                                                                   nb_actions, 1, args.backup_target_update,
                                                                   aggregate_q, s2idx)
                                elif args.backup_target == "n-step-Q":
                                    dataset = n_step_Q_backup(replay_buffer, eval_Q, env_steps,
                                                              args.batch_size, args.multi_step, args.discount,
                                                              aggregate_q, s2idx)
                                else:
                                    raise ValueError()
                                #loss = train_distill(args, dqn, dataset, args.device, optimizer, args.seen_only)
                                loss = train_DQN(args, dqn, dataset, args.device, optimizer, replay_buffer)
                            else:
                                dataset = prepare_data(args, eval_Q, s2idx, args.distill_target, args.batch_size, SA)
                                loss = train_distill(args, dqn, dataset, args.device, optimizer, args.seen_only)

                    if env_steps % args.eval_period == 0:
                        print(f"distill loss {loss}")
                    if env_steps % args.target_update == 0:
                        dqn.update_target_net()
                        for state, s_vector in enumerate(s2idx.states):
                            init_table(s_vector, state, Q, eval_Q, env, dqn, learn_start, episode_steps, SA)
                timings.time("distill")

            if env_steps % args.eval_period == 0:
                if "MiniGrid" in args.env:
                    if args.initialization in ["distilled"]:
                        test_scores = test(args, dqn, s2idx, seed=current_seed, rounds=3)
                    else:
                        test_scores = test(args, eval_Q, s2idx, seed=current_seed, rounds=3)
                else:
                    if args.initialization in ["distilled"]:
                        test_scores = test(args, dqn, s2idx, seed=current_seed, rounds=3)
                    else:
                        test_scores = test(args, eval_Q, s2idx, seed=current_seed, rounds=3)

                test_return = np.mean(test_scores)
                if len(rList) > 3:
                    actor_return = np.mean(np.array(rList[-3:]))
                else:
                    actor_return = np.mean(np.array(rList))

                print('Steps: ', env_steps, 'Iteration: ', i, ' Actor_return: ', actor_return,
                      f"Test score: {test_return} +- {np.std(test_scores)}, number of states {s2idx.max}")

                record["step"].append(env_steps)
                record["mean_episode_return"].append(test_return)
                record["number_of_states"].append(s2idx.max)
                timings.time("eval")
                print(timings.summary2())
                timings.reset()

                pd.DataFrame(record).to_csv(f"{results_dir}/logs.csv")
                if args.log_table and env_steps % (args.eval_period*10)==0:
                    print(f"solved level {current_seed-args.seed} with {env_steps-last_solved_steps} steps")
                    current_seed += 1
                    last_solved_steps = env_steps
                    replay_buffer = []
                    break

            if d == True or episode_steps == max_steps-1:
                rList.append(rAll)
                break


if args.mode == "train":
    train()
elif args.mode == "distill":
    distill_from_file(args)
