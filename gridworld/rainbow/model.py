# -*- coding: utf-8 -*-
from __future__ import division
import math
import torch
from torch import nn
from torch.nn import functional as F


# Factorised NoisyLinear layer with bias
class NoisyLinear(nn.Module):
    def __init__(self, in_features, out_features, std_init=0.5):
        super(NoisyLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.std_init = std_init
        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.register_buffer('weight_epsilon', torch.empty(out_features, in_features))
        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))
        self.register_buffer('bias_epsilon', torch.empty(out_features))
        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self):
        mu_range = 1 / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.std_init / math.sqrt(self.in_features))
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.std_init / math.sqrt(self.out_features))

    def _scale_noise(self, size):
        x = torch.randn(size, device=self.weight_mu.device)
        return x.sign().mul_(x.abs().sqrt_())

    def reset_noise(self):
        epsilon_in = self._scale_noise(self.in_features)
        epsilon_out = self._scale_noise(self.out_features)
        self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)

    def forward(self, input):
        if self.training:
            return F.linear(input, self.weight_mu + self.weight_sigma * self.weight_epsilon, self.bias_mu + self.bias_sigma * self.bias_epsilon)
        else:
            return F.linear(input, self.weight_mu, self.bias_mu)


class DQN(nn.Module):
    def __init__(self, args, action_space, with_time, nb_features):
        super(DQN, self).__init__()
        self.atoms = args.atoms
        self.action_space = action_space
        self.with_time = with_time
        self.disable_dist = args.disable_dist
        self.disable_duelling = args.disable_duelling

        if self.disable_dist:
            self.atoms = 1

        if args.architecture == 'canonical':
            self.convs = nn.Sequential(nn.Conv2d(args.history_length, 32, 8, stride=4, padding=0), nn.ReLU(),
                                                                 nn.Conv2d(32, 64, 4, stride=2, padding=0), nn.ReLU(),
                                                                 nn.Conv2d(64, 64, 3, stride=1, padding=0), nn.ReLU())
            self.conv_output_size = 3136
        elif args.architecture == 'data-efficient':
            self.convs = nn.Sequential(nn.Conv2d(args.history_length, 32, 5, stride=5, padding=0), nn.ReLU(),
                                                                 nn.Conv2d(32, 64, 5, stride=5, padding=0), nn.ReLU())
            self.conv_output_size = 576
        elif args.architecture == "gridworld":
            self.convs = nn.Sequential(nn.Conv2d(3, 8, 4, stride=2, padding=0), nn.ReLU(),
                                       nn.Conv2d(8, 16, 3, stride=1, padding=0), nn.ReLU())
            self.conv_output_size = 2704
        elif args.architecture == "minatar":
            self.convs = nn.Sequential(nn.Conv2d(nb_features, 12, 3, stride=1, padding=0), nn.ReLU(),
                                       nn.Conv2d(12, 12, 3, stride=1, padding=0), nn.ReLU())
            self.conv_output_size = 12*6*6
        #self.conv_output_size = 32*32*3
        self.disable_noisy = args.disable_noisy
        if args.disable_noisy:
            self.fc_h_v = nn.Linear(self.conv_output_size + int(with_time), args.hidden_size)
            if args.architecture in ["gridworld", "minatar"]:
                self.fc_h_a = nn.Sequential(nn.Linear(self.conv_output_size+int(with_time), args.hidden_size),
                                            nn.ReLU(),
                                            nn.Linear(args.hidden_size, args.hidden_size))
            else:
                self.fc_h_a = nn.Linear(self.conv_output_size+int(with_time), args.hidden_size)
            self.fc_z_v = nn.Linear(args.hidden_size, self.atoms)
            self.fc_z_a = nn.Linear(args.hidden_size, action_space * self.atoms)
        else:
            self.fc_h_v = NoisyLinear(self.conv_output_size+int(with_time), args.hidden_size, std_init=args.noisy_std)
            if args.architecture in ["gridworld", "minatar"]:
                self.fc_h_a = nn.Sequential(NoisyLinear(self.conv_output_size + int(with_time), args.hidden_size),
                                            nn.ReLU(),
                                            NoisyLinear(args.hidden_size, args.hidden_size))
            else:
                self.fc_h_a = NoisyLinear(self.conv_output_size + int(with_time), args.hidden_size)
            self.fc_z_v = NoisyLinear(args.hidden_size, self.atoms, std_init=args.noisy_std)
            self.fc_z_a = NoisyLinear(args.hidden_size, action_space * self.atoms, std_init=args.noisy_std)

    def forward(self, x, t=None, log=False):
        x = self.convs(x)
        #x = x.view(-1, self.conv_output_size)
        x = x.reshape(-1, self.conv_output_size)
        if self.with_time:
            x = torch.cat([x, t], dim=1)
        v = self.fc_z_v(F.relu(self.fc_h_v(x)))    # Value stream
        a = self.fc_z_a(F.relu(self.fc_h_a(x)))    # Advantage stream
        v, a = v.view(-1, 1, self.atoms), a.view(-1, self.action_space, self.atoms)
        if self.disable_duelling:
            q = a
        else:
            q = v + a - a.mean(1, keepdim=True)  # Combine streams

        if not self.disable_dist:
            if log:    # Use log softmax for numerical stability
                q = F.log_softmax(q, dim=2)    # Log probabilities with action over second dimension
            else:
                q = F.softmax(q, dim=2)    # Probabilities with action over second dimension
        else:
            q = q[:,:,0]
        return q

    def reset_noise(self):
        for name, module in self.named_children():
            if 'fc' in name:
                if not self.disable_noisy:
                    if isinstance(module, nn.Sequential):
                        for n, child in module.named_children():
                            if 'fc' in n:
                                child.reset_noise()
                    else:
                        module.reset_noise()
