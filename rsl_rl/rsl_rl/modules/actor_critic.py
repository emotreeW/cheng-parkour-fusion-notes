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

import code
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from torch.nn.modules import rnn
from torch.nn.modules.activation import ReLU

FUSION_MODES = ("concat", "film_prop", "hyper_last", "film_vis", "film_cat")


class StateHistoryEncoder(nn.Module):
    def __init__(self, activation_fn, input_size, tsteps, output_size, tanh_encoder_output=False):
        # self.device = device
        super(StateHistoryEncoder, self).__init__()
        self.activation_fn = activation_fn
        self.tsteps = tsteps

        channel_size = 10
        # last_activation = nn.ELU()

        self.encoder = nn.Sequential(
                nn.Linear(input_size, 3 * channel_size), self.activation_fn,
                )

        if tsteps == 50:
            self.conv_layers = nn.Sequential(
                    nn.Conv1d(in_channels = 3 * channel_size, out_channels = 2 * channel_size, kernel_size = 8, stride = 4), self.activation_fn,
                    nn.Conv1d(in_channels = 2 * channel_size, out_channels = channel_size, kernel_size = 5, stride = 1), self.activation_fn,
                    nn.Conv1d(in_channels = channel_size, out_channels = channel_size, kernel_size = 5, stride = 1), self.activation_fn, nn.Flatten())
        elif tsteps == 10:
            self.conv_layers = nn.Sequential(
                nn.Conv1d(in_channels = 3 * channel_size, out_channels = 2 * channel_size, kernel_size = 4, stride = 2), self.activation_fn,
                nn.Conv1d(in_channels = 2 * channel_size, out_channels = channel_size, kernel_size = 2, stride = 1), self.activation_fn,
                nn.Flatten())
        elif tsteps == 20:
            self.conv_layers = nn.Sequential(
                nn.Conv1d(in_channels = 3 * channel_size, out_channels = 2 * channel_size, kernel_size = 6, stride = 2), self.activation_fn,
                nn.Conv1d(in_channels = 2 * channel_size, out_channels = channel_size, kernel_size = 4, stride = 2), self.activation_fn,
                nn.Flatten())
        else:
            raise(ValueError("tsteps must be 10, 20 or 50"))

        self.linear_output = nn.Sequential(
                nn.Linear(channel_size * 3, output_size), self.activation_fn
                )

    def forward(self, obs):
        # nd * T * n_proprio
        nd = obs.shape[0]
        T = self.tsteps
        # print("obs device", obs.device)
        # print("encoder device", next(self.encoder.parameters()).device)
        projection = self.encoder(obs.reshape([nd * T, -1])) # do projection for n_proprio -> 32
        output = self.conv_layers(projection.reshape([nd, T, -1]).permute((0, 2, 1)))
        output = self.linear_output(output)
        return output

class Actor(nn.Module):
    def __init__(self, num_prop, 
                 num_scan, 
                 num_actions, 
                 scan_encoder_dims,
                 actor_hidden_dims, 
                 priv_encoder_dims, 
                 num_priv_latent, 
                 num_priv_explicit, 
                 num_hist, activation, 
                 tanh_encoder_output=False,
                 fusion_mode="concat") -> None:
        super().__init__()
        # prop -> scan -> priv_explicit -> priv_latent -> hist
        # actor input: prop -> scan -> priv_explicit -> latent
        if fusion_mode not in FUSION_MODES:
            raise ValueError("fusion_mode must be one of {}, got {}".format(FUSION_MODES, fusion_mode))
        self.fusion_mode = fusion_mode
        self.num_prop = num_prop
        self.num_scan = num_scan
        self.num_hist = num_hist
        self.num_actions = num_actions
        self.num_priv_latent = num_priv_latent
        self.num_priv_explicit = num_priv_explicit
        self.actor_hidden_dims = list(actor_hidden_dims)
        self.if_scan_encode = scan_encoder_dims is not None and num_scan > 0
        # film_prop / hyper_last: z modulates the MLP, not concatenated
        self.concat_z = fusion_mode in ("concat", "film_vis", "film_cat")

        if len(priv_encoder_dims) > 0:
                    priv_encoder_layers = []
                    priv_encoder_layers.append(nn.Linear(num_priv_latent, priv_encoder_dims[0]))
                    priv_encoder_layers.append(activation)
                    for l in range(len(priv_encoder_dims) - 1):
                        priv_encoder_layers.append(nn.Linear(priv_encoder_dims[l], priv_encoder_dims[l + 1]))
                        priv_encoder_layers.append(activation)
                    self.priv_encoder = nn.Sequential(*priv_encoder_layers)
                    priv_encoder_output_dim = priv_encoder_dims[-1]
        else:
            self.priv_encoder = nn.Identity()
            priv_encoder_output_dim = num_priv_latent
        self.priv_encoder_output_dim = priv_encoder_output_dim

        self.history_encoder = StateHistoryEncoder(activation, num_prop, num_hist, priv_encoder_output_dim)

        if self.if_scan_encode:
            scan_encoder = []
            scan_encoder.append(nn.Linear(num_scan, scan_encoder_dims[0]))
            scan_encoder.append(activation)
            for l in range(len(scan_encoder_dims) - 1):
                if l == len(scan_encoder_dims) - 2:
                    scan_encoder.append(nn.Linear(scan_encoder_dims[l], scan_encoder_dims[l+1]))
                    scan_encoder.append(nn.Tanh())
                else:
                    scan_encoder.append(nn.Linear(scan_encoder_dims[l], scan_encoder_dims[l + 1]))
                    scan_encoder.append(activation)
            self.scan_encoder = nn.Sequential(*scan_encoder)
            self.scan_encoder_output_dim = scan_encoder_dims[-1]
        else:
            self.scan_encoder = nn.Identity()
            self.scan_encoder_output_dim = num_scan

        scan_in_dim = self.scan_encoder_output_dim if self.concat_z else 0
        in_dim = num_prop + scan_in_dim + num_priv_explicit + priv_encoder_output_dim
        hidden0 = actor_hidden_dims[0]
        z_dim = self.scan_encoder_output_dim

        actor_layers = []
        actor_layers.append(nn.Linear(in_dim, hidden0))
        actor_layers.append(activation)
        for l in range(len(actor_hidden_dims)):
            if l == len(actor_hidden_dims) - 1:
                if fusion_mode != "hyper_last":
                    actor_layers.append(nn.Linear(actor_hidden_dims[l], num_actions))
            else:
                actor_layers.append(nn.Linear(actor_hidden_dims[l], actor_hidden_dims[l + 1]))
                actor_layers.append(activation)
        if tanh_encoder_output:
            actor_layers.append(nn.Tanh())
        self.actor_backbone = nn.Sequential(*actor_layers)

        if fusion_mode in ("film_prop", "film_cat"):
            self.film_gen = nn.Sequential(
                nn.Linear(z_dim, 256),
                nn.ELU(),
                nn.Linear(256, 2 * hidden0),
            )
            nn.init.zeros_(self.film_gen[-1].weight)
            nn.init.zeros_(self.film_gen[-1].bias)
            self.film_gen[-1].bias.data[:hidden0] = 1.0
        else:
            self.film_gen = None

        if fusion_mode == "hyper_last":
            last_h = actor_hidden_dims[-1]
            self.hyper_W_base = nn.Parameter(torch.zeros(num_actions, last_h))
            self.hyper_b_base = nn.Parameter(torch.zeros(num_actions))
            self.hyper_gen = nn.Sequential(
                nn.Linear(z_dim, 128),
                nn.ELU(),
                nn.Linear(128, num_actions * last_h + num_actions),
            )
            nn.init.zeros_(self.hyper_gen[-1].weight)
            nn.init.zeros_(self.hyper_gen[-1].bias)
        else:
            self.hyper_W_base = None
            self.hyper_b_base = None
            self.hyper_gen = None

    def _scan_latent(self, obs, scandots_latent):
        if not self.if_scan_encode:
            return None
        if scandots_latent is not None:
            return scandots_latent
        obs_scan = obs[:, self.num_prop:self.num_prop + self.num_scan]
        return self.scan_encoder(obs_scan)

    def _priv_and_hist(self, obs, hist_encoding):
        obs_priv_explicit = obs[:, self.num_prop + self.num_scan:self.num_prop + self.num_scan + self.num_priv_explicit]
        if hist_encoding:
            latent = self.infer_hist_latent(obs)
        else:
            latent = self.infer_priv_latent(obs)
        return obs_priv_explicit, latent

    def forward(self, obs, hist_encoding: bool, eval=False, scandots_latent=None):
        scan_latent = self._scan_latent(obs, scandots_latent)
        obs_priv_explicit, latent = self._priv_and_hist(obs, hist_encoding)
        if self.concat_z:
            if scan_latent is None:
                obs_prop_scan = obs[:, :self.num_prop + self.num_scan]
            else:
                obs_prop_scan = torch.cat([obs[:, :self.num_prop], scan_latent], dim=1)
            backbone_input = torch.cat([obs_prop_scan, obs_priv_explicit, latent], dim=1)
        else:
            backbone_input = torch.cat([obs[:, :self.num_prop], obs_priv_explicit, latent], dim=1)

        if self.fusion_mode in ("film_prop", "film_cat"):
            h = self.actor_backbone[0](backbone_input)
            h = self.actor_backbone[1](h)
            gb = self.film_gen(scan_latent)
            gamma, beta = gb.chunk(2, dim=-1)
            h = gamma * h + beta
            return self.actor_backbone[2:](h)

        if self.fusion_mode == "hyper_last":
            h = self.actor_backbone(backbone_input)
            delta = self.hyper_gen(scan_latent)
            n_act = self.num_actions
            last_h = self.actor_hidden_dims[-1]
            dW = delta[:, :n_act * last_h].view(-1, n_act, last_h)
            db = delta[:, n_act * last_h:]
            W = self.hyper_W_base.unsqueeze(0) + dW
            b = self.hyper_b_base.unsqueeze(0) + db
            return torch.bmm(W, h.unsqueeze(-1)).squeeze(-1) + b

        return self.actor_backbone(backbone_input)
    
    def infer_priv_latent(self, obs):
        priv = obs[:, self.num_prop + self.num_scan + self.num_priv_explicit: self.num_prop + self.num_scan + self.num_priv_explicit + self.num_priv_latent]
        return self.priv_encoder(priv)
    
    def infer_hist_latent(self, obs):
        hist = obs[:, -self.num_hist*self.num_prop:]
        return self.history_encoder(hist.view(-1, self.num_hist, self.num_prop))
    
    def infer_scandots_latent(self, obs):
        scan = obs[:, self.num_prop:self.num_prop + self.num_scan]
        return self.scan_encoder(scan)


def _copy_linear(src, dst):
    dst.weight.data.copy_(src.weight.data)
    if src.bias is not None and dst.bias is not None:
        dst.bias.data.copy_(src.bias.data)


def copy_compatible_actor(src, dst):
    """Copy history/priv/scan encoders and matching MLP slices from teacher concat actor."""
    dst.priv_encoder.load_state_dict(src.priv_encoder.state_dict())
    dst.history_encoder.load_state_dict(src.history_encoder.state_dict())
    dst.scan_encoder.load_state_dict(src.scan_encoder.state_dict())
    src_in = src.actor_backbone[0]
    dst_in = dst.actor_backbone[0]
    if src_in.weight.shape == dst_in.weight.shape:
        _copy_linear(src_in, dst_in)
    elif dst.fusion_mode in ("film_prop", "hyper_last"):
        # teacher first layer is [prop | scan_z | priv_explicit | hist]
        n_prop = dst.num_prop
        n_z = src.scan_encoder_output_dim
        n_priv = dst.num_priv_explicit
        w = src_in.weight.data
        dst_in.weight.data.copy_(torch.cat([w[:, :n_prop], w[:, n_prop + n_z:n_prop + n_z + n_priv + dst.priv_encoder_output_dim]], dim=1))
        if src_in.bias is not None and dst_in.bias is not None:
            dst_in.bias.data.copy_(src_in.bias.data)
    src_lin = [m for m in src.actor_backbone.modules() if isinstance(m, nn.Linear)]
    dst_lin = [m for m in dst.actor_backbone.modules() if isinstance(m, nn.Linear)]
    used = {0}
    for d in dst_lin[1:]:
        for i, s in enumerate(src_lin):
            if i in used:
                continue
            if s.weight.shape == d.weight.shape:
                _copy_linear(s, d)
                used.add(i)
                break
    if dst.fusion_mode == "hyper_last" and src_lin:
        last = src_lin[-1]
        if last.weight.shape == dst.hyper_W_base.shape:
            dst.hyper_W_base.data.copy_(last.weight.data)
            if last.bias is not None:
                dst.hyper_b_base.data.copy_(last.bias.data)


class ActorCriticRMA(nn.Module):
    is_recurrent = False
    def __init__(self,  num_prop,
                        num_scan,
                        num_critic_obs,
                        num_priv_latent, 
                        num_priv_explicit,
                        num_hist,
                        num_actions,
                        scan_encoder_dims=[256, 256, 256],
                        actor_hidden_dims=[256, 256, 256],
                        critic_hidden_dims=[256, 256, 256],
                        activation='elu',
                        init_noise_std=1.0,
                        **kwargs):
        if kwargs:
            print("ActorCritic.__init__ got unexpected arguments, which will be ignored: " + str([key for key in kwargs.keys()]))
        super(ActorCriticRMA, self).__init__()

        self.kwargs = kwargs
        priv_encoder_dims= kwargs['priv_encoder_dims']
        activation = get_activation(activation)
        
        self.actor = Actor(num_prop, num_scan, num_actions, scan_encoder_dims, actor_hidden_dims, priv_encoder_dims, num_priv_latent, num_priv_explicit, num_hist, activation, tanh_encoder_output=kwargs['tanh_encoder_output'])
        

        # Value function
        critic_layers = []
        critic_layers.append(nn.Linear(num_critic_obs, critic_hidden_dims[0]))
        critic_layers.append(activation)
        for l in range(len(critic_hidden_dims)):
            if l == len(critic_hidden_dims) - 1:
                critic_layers.append(nn.Linear(critic_hidden_dims[l], 1))
            else:
                critic_layers.append(nn.Linear(critic_hidden_dims[l], critic_hidden_dims[l + 1]))
                critic_layers.append(activation)
        self.critic = nn.Sequential(*critic_layers)

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

    def update_distribution(self, observations, hist_encoding):
        mean = self.actor(observations, hist_encoding)
        self.distribution = Normal(mean, mean*0. + self.std)

    def act(self, observations, hist_encoding=False, **kwargs):
        self.update_distribution(observations, hist_encoding)
        return self.distribution.sample()
    
    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    def act_inference(self, observations, hist_encoding=False, eval=False, scandots_latent=None, **kwargs):
        if not eval:
            actions_mean = self.actor(observations, hist_encoding, eval, scandots_latent)
            return actions_mean
        else:
            actions_mean, latent_hist, latent_priv = self.actor(observations, hist_encoding, eval=True)
            return actions_mean, latent_hist, latent_priv

    def evaluate(self, critic_observations, **kwargs):
        value = self.critic(critic_observations)
        return value
    
    def reset_std(self, std, num_actions, device):
        new_std = std * torch.ones(num_actions, device=device)
        self.std.data = new_std.data

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
