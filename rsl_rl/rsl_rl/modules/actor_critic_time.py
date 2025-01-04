# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

import numpy as np

import torch
import torch.nn as nn
from torch.distributions import Normal
from torch.nn.modules import rnn

class ActorCriticTime(nn.Module):
    is_recurrent = False
    def __init__(self,  num_actor_obs,
                        num_critic_obs,
                        num_actions,
                        actor_encoder_hidden_dims=[512, 256, 128],
                        critic_encoder_hidden_dims=[512, 256, 128],
                        time_encoder_hidden_dims=[256, 128],
                        policy_hidden_dims=[512, 256, 128],
                        value_hidden_dims=[512, 256, 128],
                        activation='elu',
                        init_noise_std=1.0,
                        **kwargs):
        if kwargs:
            print("ActorCritic.__init__ got unexpected arguments, which will be ignored: " + str([key for key in kwargs.keys()]))
        super(ActorCriticTime, self).__init__()

        assert actor_encoder_hidden_dims[-1] == critic_encoder_hidden_dims[-1], "Last hidden dim of actor and critic encoder must match"
        assert actor_encoder_hidden_dims[-1] == time_encoder_hidden_dims[-1], "Last hidden dim of actor and time encoder must match"

        activation = get_activation(activation)

        # actor_encoder
        actor_encoder_layers = []
        actor_encoder_layers.append(nn.Linear(num_actor_obs - 6, actor_encoder_hidden_dims[0]))
        actor_encoder_layers.append(activation)
        for l in range(len(actor_encoder_hidden_dims) - 1):
            actor_encoder_layers.append(nn.Linear(actor_encoder_hidden_dims[l], actor_encoder_hidden_dims[l + 1]))
            actor_encoder_layers.append(activation)
        self.actor_encoder = nn.Sequential(*actor_encoder_layers)

        # critic_encoder
        critic_encoder_layers = []
        critic_encoder_layers.append(nn.Linear(num_critic_obs - 6, critic_encoder_hidden_dims[0]))
        critic_encoder_layers.append(activation)
        for l in range(len(critic_encoder_hidden_dims) - 1):
            critic_encoder_layers.append(nn.Linear(critic_encoder_hidden_dims[l], critic_encoder_hidden_dims[l + 1]))
            critic_encoder_layers.append(activation)
        self.critic_encoder = nn.Sequential(*critic_encoder_layers)

        # time_encoder
        time_encoder_layers = []
        time_encoder_layers.append(nn.Linear(6, time_encoder_hidden_dims[0]))
        time_encoder_layers.append(activation)
        for l in range(len(time_encoder_hidden_dims) - 1):
            time_encoder_layers.append(nn.Linear(time_encoder_hidden_dims[l], time_encoder_hidden_dims[l + 1]))
            time_encoder_layers.append(activation)
        self.time_encoder = nn.Sequential(*time_encoder_layers)

        # policy
        policy_hidden_dims.append(num_actions)
        policy_layers = []
        policy_layers.append(nn.Linear(actor_encoder_hidden_dims[-1], policy_hidden_dims[0]))
        policy_layers.append(activation)
        for l in range(len(policy_hidden_dims) - 1):
            policy_layers.append(nn.Linear(policy_hidden_dims[l], policy_hidden_dims[l + 1]))
            policy_layers.append(activation)
        self.policy = nn.Sequential(*policy_layers)

        # value function
        value_hidden_dims.append(1)
        value_layers = []
        value_layers.append(nn.Linear(critic_encoder_hidden_dims[-1], value_hidden_dims[0]))
        value_layers.append(activation)
        for l in range(len(value_hidden_dims) - 1):
            value_layers.append(nn.Linear(value_hidden_dims[l], value_hidden_dims[l + 1]))
            value_layers.append(activation)
        self.value = nn.Sequential(*value_layers)

        print(f"Actor Encoder MLP: {self.actor_encoder}")
        print(f"Critic Encoder MLP: {self.critic_encoder}")
        print(f"Time Encoder MLP: {self.time_encoder}")
        print(f"Policy MLP: {self.policy}")
        print(f"Value MLP: {self.value}")

        # Action noise
        self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        self.distribution = None
        # disable args validation for speedup
        Normal.set_default_validate_args = False
        
        # seems that we get better performance without init
        # self.init_memory_weights(self.memory_a, 0.001, 0.)
        # self.init_memory_weights(self.memory_c, 0.001, 0.)

    @staticmethod
    # not used at the moment
    def init_weights(sequential, scales):
        [torch.nn.init.orthogonal_(module.weight, gain=scales[idx]) for idx, module in
         enumerate(mod for mod in sequential if isinstance(mod, nn.Linear))]


    def reset(self, dones=None):
        pass

    def forward(self):
        raise NotImplementedError
    
    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev
    
    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)

    def update_distribution(self, observations):
        time_hidden = self.time_encoder(observations[..., -6:])
        actor_hidden = self.actor_encoder(observations[..., :-6])
        mean = self.value(time_hidden * actor_hidden + time_hidden)
        self.distribution = Normal(mean, mean*0. + self.std)

    def act(self, observations, **kwargs):
        self.update_distribution(observations)
        return self.distribution.sample()
    
    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    def act_inference(self, observations):
        time_hidden = self.time_encoder(observations[..., -6:])
        actor_hidden = self.actor_encoder(observations[..., :-6])
        mean = self.value(time_hidden * actor_hidden + time_hidden)
        return mean

    def evaluate(self, critic_observations, **kwargs):
        time_hidden = self.time_encoder(critic_observations[..., -6:])
        critic_hidden = self.critic_encoder(critic_observations[..., :-6])
        value = self.value(time_hidden * critic_hidden + time_hidden)
        return value

def get_activation(act_name):
    if act_name == "elu":
        return nn.ELU()
    elif act_name == "selu":
        return nn.SELU()
    elif act_name == "relu":
        return nn.ReLU()
    elif act_name == "crelu":
        return nn.ReLU()
    elif act_name == "lrelu":
        return nn.LeakyReLU()
    elif act_name == "tanh":
        return nn.Tanh()
    elif act_name == "sigmoid":
        return nn.Sigmoid()
    else:
        print("invalid activation function!")
        return None
