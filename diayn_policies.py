import numpy as np
import torch
from torch import nn as nn
import torch.nn.functional as F

import abc
from torch.distributions import Distribution, Normal
from dqn_model import CoreNet, HeadNet, DuelingHeadNet

LOG_SIG_MAX = 2
LOG_SIG_MIN = -20

def my_ce_loss(input, target):
    return -torch.sum(F.log_softmax(input, dim=1) * target, dim=1)

def weight_init(module):
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)
        module.bias.data.zero_()

def identity(x):
    return x

def concat_obs_zs(obs, z):
    """Concatenates the observation to a one-hot encoding of Z."""
    states = []
    for x, y in zip(obs, z):
        states.append(concat_obs_z(x, y))
    return states

def concat_obs_z(obs, z):
    """Concatenates the observation to a one-hot encoding of Z."""
    return np.concatenate((obs, z), axis=0)

def z_one_hot(z, num_skills):
    assert np.isscalar(z)
    z_one_hot = np.zeros(num_skills)
    z_one_hot[0][int(z)] = 1
    return np.expand_dims(z_one_hot, axis=0)

def z_one_hots(z, num_skills):
    zs = []
    for x in z:
        zs.append(z_one_hot(x, num_skills))
    return zs
    
class QNet(nn.Module):
    def __init__(self, n_actions, network_output_size, num_channels, dueling=False):
        super(QNet, self).__init__()
        self.core_net = CoreNet(network_output_size=network_output_size, num_channels=num_channels)
        self.dueling = dueling
        if self.dueling:
            print("using dueling dqn")
            self.head = DuelingHeadNet(n_actions=n_actions)
            self.head2 = DuelingHeadNet(n_actions=n_actions)

        else:
            self.head = HeadNet(n_actions=n_actions)
            self.head2 = HeadNet(n_actions=n_actions)

    def _core(self, x):
        return self.core_net(x)

    def forward(self, x):
        x = self.core_net(x)
        return self.head(x), self.head2(x)

class VNet(nn.Module):
    def __init__(self, n_actions, network_output_size, num_channels, dueling=False):
        super(VNet, self).__init__()
        self.core_net = CoreNet(network_output_size=network_output_size, num_channels=num_channels)
        self.dueling = dueling
        if self.dueling:
            print("using dueling dqn")
            self.head = DuelingHeadNet(n_actions=n_actions)

        else:
            self.head = HeadNet(n_actions=n_actions)

    def _core(self, x):
        return self.core_net(x)

    def forward(self, x):
        x = self.core_net(x)
        return self.head(x)

class TanhNormal(Distribution):
    """
    Represent distribution of X where
        X ~ tanh(Z)
        Z ~ N(mean, std)
    Note: this is not very numerically stable.
    """
    def __init__(self, normal_mean, normal_std, epsilon=1e-6):
        """
        :param normal_mean: Mean of the normal distribution
        :param normal_std: Std of the normal distribution
        :param epsilon: Numerical stability epsilon when computing log-prob.
        """
        self.normal_mean = normal_mean
        self.normal_std = normal_std
        self.normal = Normal(normal_mean, normal_std)
        self.epsilon = epsilon

    def sample_n(self, n, return_pre_tanh_value=False):
        z = self.normal.sample_n(n)
        if return_pre_tanh_value:
            return torch.tanh(z), z
        else:
            return torch.tanh(z)

    def log_prob(self, value, pre_tanh_value=None):
        """
        :param value: some value, x
        :param pre_tanh_value: arctanh(x)
        :return:
        """
        if pre_tanh_value is None:
            pre_tanh_value = torch.log(
                (1+value) / (1-value)
            ) / 2
        return self.normal.log_prob(pre_tanh_value) - torch.log(
            1 - value * value + self.epsilon
        )

    def sample(self, return_pretanh_value=False):
        """
        Gradients will and should *not* pass through this operation.
        See https://github.com/pytorch/pytorch/issues/4620 for discussion.
        """
        z = self.normal.sample().detach()

        if return_pretanh_value:
            return torch.tanh(z), z
        else:
            return torch.tanh(z)

    def rsample(self, return_pretanh_value=False):
        """
        Sampling in the reparameterization case.
        """
        z = (
            self.normal_mean +
            self.normal_std *
            Normal(
                torch.zeros(self.normal_mean.size()),
                torch.ones(self.normal_std.size())
            ).sample()
        )
        z.requires_grad_()

        if return_pretanh_value:
            return torch.tanh(z), z
        else:
            return torch.tanh(z)

# class Mlp(nn.Module):
#     def __init__(
#             self,
#             hidden_sizes,
#             output_size,
#             input_size,
#             init_w=3e-3,
#             hidden_activation=F.relu,
#             output_activation=identity,
#             hidden_init=ptu.fanin_init,
#             b_init_value=0.1,
#             layer_norm=False,
#             layer_norm_kwargs=None,
#     ):
#         super().__init__()

#         if layer_norm_kwargs is None:
#             layer_norm_kwargs = dict()

#         self.input_size = input_size
#         self.output_size = output_size
#         self.hidden_activation = hidden_activation
#         self.output_activation = output_activation
#         self.layer_norm = layer_norm
#         self.fcs = []
#         self.layer_norms = []
#         in_size = input_size

#         for i, next_size in enumerate(hidden_sizes):
#             fc = nn.Linear(in_size, next_size)
#             in_size = next_size
#             hidden_init(fc.weight)
#             fc.bias.data.fill_(b_init_value)
#             self.__setattr__("fc{}".format(i), fc)
#             self.fcs.append(fc)

#             if self.layer_norm:
#                 ln = LayerNorm(next_size)
#                 self.__setattr__("layer_norm{}".format(i), ln)
#                 self.layer_norms.append(ln)

#         self.last_fc = nn.Linear(in_size, output_size)
#         self.last_fc.weight.data.uniform_(-init_w, init_w)
#         self.last_fc.bias.data.uniform_(-init_w, init_w)

#     def forward(self, input, return_preactivations=False):
#         h = input
#         for i, fc in enumerate(self.fcs):
#             h = fc(h)
#             if self.layer_norm and i < len(self.fcs) - 1:
#                 h = self.layer_norms[i](h)
#             h = self.hidden_activation(h)
#         preactivation = self.last_fc(h)
#         output = self.output_activation(preactivation)
#         if return_preactivations:
#             return output, preactivation
#         else:
#             return output

class Policy(object, metaclass=abc.ABCMeta):
    """
    General policy interface.
    """
    @abc.abstractmethod
    def get_action(self, observation):
        """
        :param observation:
        :return: action, debug_dictionary
        """
        pass

    def reset(self):
        pass


class ExplorationPolicy(Policy, metaclass=abc.ABCMeta):
    def set_num_steps_total(self, t):
        pass

class TanhGaussianPolicy(ExplorationPolicy):
    """
    Usage:
    ```
    policy = TanhGaussianPolicy(...)
    action, mean, log_std, _ = policy(obs)
    action, mean, log_std, _ = policy(obs, deterministic=True)
    action, mean, log_std, log_prob = policy(obs, return_log_prob=True)
    ```
    Here, mean and log_std are the mean and log_std of the Gaussian that is
    sampled from.
    If deterministic is True, action = tanh(mean).
    If return_log_prob is False (default), log_prob = None
        This is done because computing the log_prob can be a bit expensive.
    """
    def __init__(
            self,
            obs_dim,
            action_dim,
            n_ensemble,
            history_size,
            if_dueling,
            device,
            std=None,
            policy_net = None,
            init_w=1e-3,
            **kwargs
    ):
        self.policy_net = QNet(n_actions=action_dim,
                                network_output_size=obs_dim,
                                num_channels=history_size, dueling=if_dueling).to(device)
        if policy_net:
            self.policy_net.load_state_dict(policy_net.state_dict())

        self.log_std = None
        self.std = std
        self.n_ensemble = n_ensemble
        self.obs_dim = obs_dim
        if std is None:
            pass
            # last_hidden_size = obs_dim
            # if len(hidden_sizes) > 0:
            #     last_hidden_size = hidden_sizes[-1]
            # self.last_fc_log_std = nn.Linear(last_hidden_size, action_dim)
            # self.last_fc_log_std.weight.data.uniform_(-init_w, init_w)
            # self.last_fc_log_std.bias.data.uniform_(-init_w, init_w)
        else:
            self.log_std = np.log(std)
            assert LOG_SIG_MIN <= self.log_std <= LOG_SIG_MAX

    def get_action(self, obs_np, deterministic=False):
        actions = self.get_actions(obs_np, deterministic=deterministic)
        return actions[0]

    def get_actions(self, obs_np, deterministic=False):
        return self.forward(obs_np, deterministic=deterministic)
    
    def get_network(self):
        return self.policy_net

    def forward(
            self,
            obs,
            reparameterize=True,
            deterministic=False,
            return_log_prob=True,
    ):
        """
        :param obs: Observation
        :param deterministic: If True, do not sample
        :param return_log_prob: If True, return a sample and its log probability
        """
        # if skill is not None:
        #     h = torch.cat([obs, skill], -1)
        # else:
        #     h = obs
        # for i, fc in enumerate(self.fcs):
        #     h = self.hidden_activation(fc(h))
        # mean = self.last_fc(h)

        mean, log_std = self.policy_net(obs)

        if self.std is None:
            log_std = torch.clamp(log_std, LOG_SIG_MIN, LOG_SIG_MAX)
            std = torch.exp(log_std)
        else:
            std = self.std
            log_std = self.log_std

        log_prob = None
        entropy = None
        mean_action_log_prob = None
        pre_tanh_value = None
        if deterministic:
            a = np.argmax(mean, dim=-1)
        else:
            tanh_normal = TanhNormal(mean, std)
            if return_log_prob:
                if reparameterize is True:
                    action, pre_tanh_value = tanh_normal.rsample(
                        return_pretanh_value=True
                    )
                else:
                    action, pre_tanh_value = tanh_normal.sample(
                        return_pretanh_value=True
                    )
                log_prob = tanh_normal.log_prob(
                    action,
                    pre_tanh_value=pre_tanh_value
                )
                log_prob = log_prob.sum(dim=1, keepdim=True)
            else:
                if reparameterize is True:
                    action = tanh_normal.rsample()
                else:
                    action = tanh_normal.sample()
            action = torch.distributions.Categorical(logits=action)
            a = action.sample()

        return (
            a, mean, log_std, log_prob.squeeze(), entropy, std,
            mean_action_log_prob, pre_tanh_value,
        )

    # def sample(self, state):
    #     mean, log_std = self.forward(state)
    #     std = log_std.exp()
    #     normal = Normal(mean, std)
    #     x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
    #     y_t = torch.tanh(x_t)
    #     action = y_t * self.action_scale + self.action_bias
    #     log_prob = normal.log_prob(x_t)
    #     # Enforcing Action Bound
    #     log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + epsilon)
    #     log_prob = log_prob.sum(1, keepdim=True)
    #     mean = torch.tanh(mean) * self.action_scale + self.action_bias
    #     return action, log_prob, mean


class MakeDeterministic(Policy):
    def __init__(self, stochastic_policy):
        self.stochastic_policy = stochastic_policy

    def get_action(self, observation, skill=None):
        return self.stochastic_policy.get_action(observation,
                                                 deterministic=True)


# class CategoricalMLPPolicy(Policy):
#     """Policy network based on a multi-layer perceptron (MLP), with a 
#     `Categorical` distribution output. This policy network can be used on tasks 
#     with discrete action spaces (eg. `TabularMDPEnv`). The code is adapted from 
#     https://github.com/cbfinn/maml_rl/blob/9c8e2ebd741cb0c7b8bf2d040c4caeeb8e06cc95/sandbox/rocky/tf/policies/maml_minimal_categorical_mlp_policy.py
#     """
#     def __init__(self, input_size, output_size,
#                  hidden_sizes=(), nonlinearity=F.relu):
#         super(CategoricalMLPPolicy, self).__init__(
#             input_size=input_size, output_size=output_size)
#         self.hidden_sizes = hidden_sizes
#         self.nonlinearity = nonlinearity
#         self.num_layers = len(hidden_sizes) + 1

#         layer_sizes = (input_size,) + hidden_sizes + (output_size,)
#         for i in range(1, self.num_layers + 1):
#             self.add_module('layer{0}'.format(i),
#                 nn.Linear(layer_sizes[i - 1], layer_sizes[i]))
#         self.apply(weight_init)

#     def torch_ify(self, np_array_or_other):
#         if isinstance(np_array_or_other, np.ndarray):
#             return ptu.from_numpy(np_array_or_other)
#         else:
#             return np_array_or_other

#     def forward(self, input, params=None):
#         if params is None:
#             params = OrderedDict(self.named_parameters())
#         input = self.torch_ify(input)
#         output = input
#         for i in range(1, self.num_layers):
#             output = F.linear(output,
#                 weight=params['layer{0}.weight'.format(i)],
#                 bias=params['layer{0}.bias'.format(i)])
#             output = self.nonlinearity(output)
#         logits = F.linear(output,
#             weight=params['layer{0}.weight'.format(self.num_layers)],
#             bias=params['layer{0}.bias'.format(self.num_layers)])

#         return OneHotCategorical(logits=logits)
