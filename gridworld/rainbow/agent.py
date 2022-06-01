# -*- coding: utf-8 -*-
from __future__ import division
import os
import numpy as np
import torch
from torch import optim
from torch.nn.utils import clip_grad_norm_
from rainbow.model import DQN

def initialize_visits(visit_dict, state, nb_actions):
    if state not in visit_dict:
        visit_dict[state] = np.zeros(nb_actions, dtype=np.int32)

class Agent():
    def __init__(self, args, nb_actions, nb_features):
        self.action_space = nb_actions
        self.atoms = args.atoms
        self.Vmin = args.V_min
        self.Vmax = args.V_max
        self.support = torch.linspace(args.V_min, args.V_max, self.atoms).to(device=args.device)    # Support (range) of z
        self.delta_z = (args.V_max - args.V_min) / (self.atoms - 1)
        self.batch_size = args.batch_size
        self.n = args.multi_step
        self.discount = args.discount
        self.norm_clip = args.norm_clip
        self.exploration = args.exploration
        self.device= args.device
        with_steps = True if "MiniGrid" in args.env and not args.disable_steps else False
        self.mask_loss = True if "MiniGrid" in args.env and not with_steps else False
        self.with_steps = with_steps
        self.disable_dist = args.disable_dist
        self.disable_double = args.disable_double

        self.online_net = DQN(args, self.action_space, with_steps, nb_features).to(device=args.device)
        if args.model:    # Load pretrained model if provided
            if os.path.isfile(args.model):
                state_dict = torch.load(args.model, map_location='cpu')    # Always load tensors onto CPU by default, will shift to GPU if necessary
                if 'conv1.weight' in state_dict.keys():
                    for old_key, new_key in (('conv1.weight', 'convs.0.weight'), ('conv1.bias', 'convs.0.bias'), ('conv2.weight', 'convs.2.weight'), ('conv2.bias', 'convs.2.bias'), ('conv3.weight', 'convs.4.weight'), ('conv3.bias', 'convs.4.bias')):
                        state_dict[new_key] = state_dict[old_key]    # Re-map state dict for old pretrained models
                        del state_dict[old_key]    # Delete old keys for strict load_state_dict
                self.online_net.load_state_dict(state_dict)
                print("Loading pretrained model: " + args.model)
            else:    # Raise error if incorrect model path provided
                raise FileNotFoundError(args.model)

        self.online_net.train()

        self.target_net = DQN(args, self.action_space, with_steps, nb_features).to(device=args.device)
        self.update_target_net()
        self.target_net.train()
        for param in self.target_net.parameters():
            param.requires_grad = False

        self.optimiser = optim.Adam(self.online_net.parameters(), lr=args.learning_rate, eps=args.adam_eps)
        self.visit_dict = {}

    # Resets noisy weights in all linear layers (of online net only)
    def reset_noise(self):
        self.online_net.reset_noise()

    # Acts based on single state (no batch)
    def act(self, state, steps):
        with torch.no_grad():
            return self.evaluate_q(state, steps).argmax().item()


    # Acts with an ε-greedy policy (used for evaluation only)
    def act_with_exploration(self, state, steps, epsilon, test=False):    # High ε can reduce evaluation scores drastically
        if self.exploration in "epsilon_greedy" or test:
            return np.random.randint(0, self.action_space) if np.random.random() < epsilon else self.act(state, steps)
        elif self.exploration == "AVAHash_epsilon":
            key = state.cpu().numpy().tobytes()
            initialize_visits(self.visit_dict, key, self.action_space)
            q_values = self.evaluate_q(state, steps)
            action = q_values.argmax(1).item()
            if np.random.random() < epsilon:
                visitations = self.visit_dict[key]
                candidates = np.argwhere(visitations == np.amin(visitations)).flatten()
                action = np.random.choice(candidates)
            if not test:
                self.visit_dict[key][action] += 1
            return action
        elif self.exploration == "noisy_net":
            if test:
                return np.random.randint(0, self.action_space) if np.random.random() < epsilon else self.act(state, steps)
            else:
                return self.act(state, steps)

    def learn(self, mem, t=0):
        # Sample transitions
        idxs, states, actions, returns, next_states, nonterminals, weights, steps = mem.sample(self.batch_size)

        #if nonterminals[0]==0 and returns[0] == 0.0:
        #    return


        if self.disable_dist:
            q = self.online_net(states, steps)
            with torch.no_grad():
                q_next_o = self.online_net(next_states, steps+self.n)    # Probabilities p(s_t+n, ·; θtarget)
                self.target_net.reset_noise()    # Sample new target net noise
                q_next_t = self.target_net(next_states, steps+self.n)
                if self.disable_double:
                    argmax_indices_ns = q_next_t.argmax(1)
                else:
                    argmax_indices_ns = q_next_o.argmax(1)    # Perform argmax action selection using online network: argmax_a[(z, q(s_t+n, a; θonline))]
                q_next_a = q_next_t[range(self.batch_size), argmax_indices_ns]    # Double-Q probabilities p(s_t+n, argmax_a[(z, p(s_t+n, a; θonline))]; θtarget)
                target = returns + nonterminals[:,0] * (self.discount ** self.n) * q_next_a
            q_a = q[range(self.batch_size), actions]
            #if target.max() > 1.0:
            #    print("")
            loss = (target- q_a)**2
            #if returns.sum() > 0:
            #    print(end="")
            #if nonterminals.sum() == 0:
            #    print(end="")
        else:
            # Calculate current state probabilities (online network noise already sampled)
            log_ps = self.online_net(states, steps, log=True)  # Log probabilities log p(s_t, ·; θonline)
            log_ps_a = log_ps[range(self.batch_size), actions]  # log p(s_t, a_t; θonline)
            with torch.no_grad():
                # Calculate nth next state probabilities
                pns = self.online_net(next_states, steps+self.n)    # Probabilities p(s_t+n, ·; θonline)
                dns = self.support.expand_as(pns) * pns    # Distribution d_t+n = (z, p(s_t+n, ·; θonline))
                argmax_indices_ns = dns.sum(2).argmax(1)    # Perform argmax action selection using online network: argmax_a[(z, p(s_t+n, a; θonline))]
                self.target_net.reset_noise()    # Sample new target net noise
                pns = self.target_net(next_states, steps+self.n)    # Probabilities p(s_t+n, ·; θtarget)
                pns_a = pns[range(self.batch_size), argmax_indices_ns]    # Double-Q probabilities p(s_t+n, argmax_a[(z, p(s_t+n, a; θonline))]; θtarget)

                # Compute Tz (Bellman operator T applied to z)
                Tz = returns.unsqueeze(1) + nonterminals * (self.discount ** self.n) * self.support.unsqueeze(0)    # Tz = R^n + (γ^n)z (accounting for terminal states)
                Tz = Tz.clamp(min=self.Vmin, max=self.Vmax)    # Clamp between supported values
                # Compute L2 projection of Tz onto fixed support z
                b = (Tz - self.Vmin) / self.delta_z    # b = (Tz - Vmin) / Δz
                l, u = b.floor().to(torch.int64), b.ceil().to(torch.int64)
                # Fix disappearing probability mass when l = b = u (b is int)
                l[(u > 0) * (l == u)] -= 1
                u[(l < (self.atoms - 1)) * (l == u)] += 1

                # Distribute probability of Tz
                m = states.new_zeros(self.batch_size, self.atoms)
                offset = torch.linspace(0, ((self.batch_size - 1) * self.atoms), self.batch_size).unsqueeze(1).expand(self.batch_size, self.atoms).to(actions)
                m.view(-1).index_add_(0, (l + offset).view(-1), (pns_a * (u.float() - b)).view(-1))    # m_l = m_l + p(s_t+n, a*)(u - b)
                m.view(-1).index_add_(0, (u + offset).view(-1), (pns_a * (b - l.float())).view(-1))    # m_u = m_u + p(s_t+n, a*)(b - l)
            #if returns.sum() > 0:
            #    print()
            #if not nonterminals.sum():
            #    print()

            loss = -torch.sum(m * log_ps_a, 1)    # Cross-entropy loss (minimises DKL(m||p(s_t, a_t)))

        #if states[0, 0, -4, -4] == 10 and states[0, 2, -4, -4] == 0:
        #    print(f"action {actions[0]}, step {steps[0]},"
        #          f"target {target[0]}, output {self.evaluate_q(states[0], steps[0])[actions[0]]}")

        #if states[0,0,-2,-3] == 10 and states[0,2,-2,-3] == 1 and actions[0]==2:
        #    q1 = self.evaluate_q(states[0], steps[0])[actions[0]]
        #    wandb.log(dict(q_before_goal=q1, qa=q_a), step=t)
        #    print(f"action {actions[0]}, step {steps[0]},"
        #          f" target {target[0]}, output {q1}")

        if self.mask_loss:
            loss *= torch.logical_or(nonterminals[:,0], returns > 0)
        self.online_net.zero_grad()
        (weights * loss).mean().backward()    # Backpropagate importance-weighted minibatch loss
        #loss.mean().backward()
        clip_grad_norm_(self.online_net.parameters(), self.norm_clip)    # Clip gradients by L2 norm
        self.optimiser.step()

        #if states[0,0,-2,-3] == 10 and states[0,2,-2,-3] == 1 and actions[0]==2:
        #    q2 = self.evaluate_q(states[0], steps[0])[actions[0]]
        #    wandb.log(dict(q_diff=q2-q1), step=t)
        #    print(f"output after update {q2}")

        if self.disable_dist:
            mem.update_priorities(idxs, torch.sqrt(loss).detach().cpu().numpy() + 1e-4)  # Update priorities of sampled transitions
        else:
            mem.update_priorities(idxs, loss.detach().cpu().numpy())    # Update priorities of sampled transitions

    def update_target_net(self):
        self.target_net.load_state_dict(self.online_net.state_dict())

    # Save model parameters on current device (don't move model between devices)
    def save(self, path, name='model.pth'):
        torch.save(self.online_net.state_dict(), os.path.join(path, name))

    # Evaluates Q-value based on single state (no batch)
    def evaluate_q(self, state, steps, use_target=False):
        with torch.no_grad():
            if steps is None:
                t = steps
            else:
                t = torch.tensor([steps]).unsqueeze(0).to(state)
            if len(state.shape) == 3:
                state = state.unsqueeze(0)
            if self.disable_dist:
                if use_target:
                    return self.target_net(state, t)[0]
                else:
                    return self.online_net(state, t)[0]
            else:
                if use_target:
                    return (self.target_net(state, t) * self.support).sum(2)[0]
                else:
                    return (self.online_net(state, t) * self.support).sum(2)[0]

    def train(self):
        self.online_net.train()

    def eval(self):
        self.online_net.eval()
