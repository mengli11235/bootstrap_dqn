import numpy as np
import torch
from torch import nn as nn
import abc
from torch.distributions import Distribution, Normal
from dqn_model import EnsembleNet, NetWithPrior

LOG_SIG_MAX = 2
LOG_SIG_MIN = -20

def weight_init(module):
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)
        module.bias.data.zero_()

def eval_np(module, *args, **kwargs):
    """
    Eval this module with a numpy interface
    Same as a call to __call__ except all Variable input/outputs are
    replaced with numpy equivalents.
    Assumes the output is either a single object or a tuple of objects.
    """
    torch_args = tuple(torch_ify(x) for x in args)
    torch_kwargs = {k: torch_ify(v) for k, v in kwargs.items()}
    outputs = module(*torch_args, **torch_kwargs)
    if isinstance(outputs, tuple):
        return tuple(np_ify(x) for x in outputs)
    else:
        return np_ify(outputs)

def identity(x):
    return x

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
                ptu.zeros(self.normal_mean.size()),
                ptu.ones(self.normal_std.size())
            ).sample()
        )
        z.requires_grad_()

        if return_pretanh_value:
            return torch.tanh(z), z
        else:
            return torch.tanh(z)

class Mlp(nn.Module):
    def __init__(
            self,
            hidden_sizes,
            output_size,
            input_size,
            init_w=3e-3,
            hidden_activation=F.relu,
            output_activation=identity,
            hidden_init=ptu.fanin_init,
            b_init_value=0.1,
            layer_norm=False,
            layer_norm_kwargs=None,
    ):
        super().__init__()

        if layer_norm_kwargs is None:
            layer_norm_kwargs = dict()

        self.input_size = input_size
        self.output_size = output_size
        self.hidden_activation = hidden_activation
        self.output_activation = output_activation
        self.layer_norm = layer_norm
        self.fcs = []
        self.layer_norms = []
        in_size = input_size

        for i, next_size in enumerate(hidden_sizes):
            fc = nn.Linear(in_size, next_size)
            in_size = next_size
            hidden_init(fc.weight)
            fc.bias.data.fill_(b_init_value)
            self.__setattr__("fc{}".format(i), fc)
            self.fcs.append(fc)

            if self.layer_norm:
                ln = LayerNorm(next_size)
                self.__setattr__("layer_norm{}".format(i), ln)
                self.layer_norms.append(ln)

        self.last_fc = nn.Linear(in_size, output_size)
        self.last_fc.weight.data.uniform_(-init_w, init_w)
        self.last_fc.bias.data.uniform_(-init_w, init_w)

    def forward(self, input, return_preactivations=False):
        h = input
        for i, fc in enumerate(self.fcs):
            h = fc(h)
            if self.layer_norm and i < len(self.fcs) - 1:
                h = self.layer_norms[i](h)
            h = self.hidden_activation(h)
        preactivation = self.last_fc(h)
        output = self.output_activation(preactivation)
        if return_preactivations:
            return output, preactivation
        else:
            return output

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
            if_dueling,
            std=None,
            init_w=1e-3,
            **kwargs
    ):
        self.policy_net = EnsembleNet(n_ensemble=num_skills,
                                      n_actions=action_dim,
                                      network_output_size=obs_dim,
                                      num_channels=history_size, dueling=if_dueling).to(device)
        self.log_std = None
        self.std = std
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

    def get_action(self, obs_np, skill=None, deterministic=False):
        actions = self.get_actions(obs_np[None], skill, deterministic=deterministic)
        return actions[0, :], {}

    def get_actions(self, obs_np, skill, deterministic=False):
        return eval_np(self, obs_np, skill, deterministic=deterministic)[0]
    
    def get_network(self):
        return self.policy_net

    def forward(
            self,
            obs,
            skill=None,
            reparameterize=True,
            deterministic=False,
            return_log_prob=False,
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
        mean = self.policy_net(obs, skill)

        if self.std is None:
            log_std = self.last_fc_log_std(h)
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
            action = torch.tanh(mean)
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

        return (
            action, mean, log_std, log_prob, entropy, std,
            mean_action_log_prob, pre_tanh_value,
        )


class MakeDeterministic(Policy):
    def __init__(self, stochastic_policy):
        self.stochastic_policy = stochastic_policy

    def get_action(self, observation, skill=None):
        return self.stochastic_policy.get_action(observation,
                                                 skill,
                                                 deterministic=True)


class CategoricalMLPPolicy(Policy):
    """Policy network based on a multi-layer perceptron (MLP), with a 
    `Categorical` distribution output. This policy network can be used on tasks 
    with discrete action spaces (eg. `TabularMDPEnv`). The code is adapted from 
    https://github.com/cbfinn/maml_rl/blob/9c8e2ebd741cb0c7b8bf2d040c4caeeb8e06cc95/sandbox/rocky/tf/policies/maml_minimal_categorical_mlp_policy.py
    """
    def __init__(self, input_size, output_size,
                 hidden_sizes=(), nonlinearity=F.relu):
        super(CategoricalMLPPolicy, self).__init__(
            input_size=input_size, output_size=output_size)
        self.hidden_sizes = hidden_sizes
        self.nonlinearity = nonlinearity
        self.num_layers = len(hidden_sizes) + 1

        layer_sizes = (input_size,) + hidden_sizes + (output_size,)
        for i in range(1, self.num_layers + 1):
            self.add_module('layer{0}'.format(i),
                nn.Linear(layer_sizes[i - 1], layer_sizes[i]))
        self.apply(weight_init)

    def torch_ify(self, np_array_or_other):
        if isinstance(np_array_or_other, np.ndarray):
            return ptu.from_numpy(np_array_or_other)
        else:
            return np_array_or_other

    def forward(self, input, params=None):
        if params is None:
            params = OrderedDict(self.named_parameters())
        input = self.torch_ify(input)
        output = input
        for i in range(1, self.num_layers):
            output = F.linear(output,
                weight=params['layer{0}.weight'.format(i)],
                bias=params['layer{0}.bias'.format(i)])
            output = self.nonlinearity(output)
        logits = F.linear(output,
            weight=params['layer{0}.weight'.format(self.num_layers)],
            bias=params['layer{0}.bias'.format(self.num_layers)])

        return OneHotCategorical(logits=logits)