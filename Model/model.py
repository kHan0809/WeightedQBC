import torch
import torch.nn as nn
from torch.distributions.normal import Normal

def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

def weight_init_Xavier(module):
    if isinstance(module, nn.Linear):
        torch.nn.init.xavier_normal_(module.weight, gain=0.01)
        module.bias.data.zero_()

def weight_init(m):
    """Custom weight init for Conv2D and Linear layers.
        Reference: https://github.com/MishaLaskin/rad/blob/master/curl_sac.py"""

    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        m.bias.data.fill_(0.0)
    elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        # delta-orthogonal init from https://arxiv.org/pdf/1806.05393.pdf
        assert m.weight.size(2) == m.weight.size(3)
        m.weight.data.fill_(0.0)
        m.bias.data.fill_(0.0)
        mid = m.weight.size(2) // 2
        gain = nn.init.calculate_gain('relu')
        nn.init.orthogonal_(m.weight.data[:, :, mid, mid], gain)

def multiple_action_q_function(forward):
    # Forward the q function with multiple actions on each state, to be used as a decorator
    def wrapped(self, observations, actions, **kwargs):
        multiple_actions = False
        batch_size = observations.shape[0]
        if actions.ndim == 3 and observations.ndim == 2:
            multiple_actions = True
            observations = extend_and_repeat(observations, 1, actions.shape[1]).reshape(-1, observations.shape[-1])
            actions = actions.reshape(-1, actions.shape[-1])
        q_values = forward(self, observations, actions, **kwargs)
        if multiple_actions:
            q_values = q_values.reshape(batch_size, -1)
        return q_values
    return wrapped



class Qnet(nn.Module):
    def __init__(self, o_dim, a_dim, h_size=256):
        super(Qnet,self).__init__()
        self.fc1 = nn.Linear(o_dim + a_dim, h_size)
        self.fc2 = nn.Linear(h_size       , h_size)
        self.fc3 = nn.Linear(h_size       , 1)
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()

    @multiple_action_q_function
    def forward(self,o_input:torch.Tensor,a_input:torch.Tensor):
        inputs = torch.concat([o_input,a_input],dim=-1)
        layer = self.relu1(self.fc1(inputs))
        layer = self.relu2(self.fc2(layer))
        qval  = self.fc3(layer)
        return torch.squeeze(qval,dim=-1)

class Vnet(nn.Module):
    def __init__(self, o_dim, h_size=256):
        super(Vnet,self).__init__()
        self.fc1 = nn.Linear(o_dim        , h_size)
        self.fc2 = nn.Linear(h_size       , h_size)
        self.fc3 = nn.Linear(h_size       , 1)
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()

    def forward(self,o_input:torch.Tensor):
        layer = self.relu1(self.fc1(o_input))
        layer = self.relu2(self.fc2(layer))
        vval  = self.fc3(layer)
        return torch.squeeze(vval,dim=-1)

class Policy(nn.Module):
    def __init__(self,o_dim,a_dim, h_size=256):
        super(Policy,self).__init__()
        self.fc1 = nn.Linear(o_dim        , h_size)
        self.fc2 = nn.Linear(h_size       , h_size)
        self.fc3 = nn.Linear(h_size       , a_dim)
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.tanh  = nn.Tanh()

    def forward(self,o_input:torch.Tensor,repeat=None):
        if repeat is not None:
            o_input = extend_and_repeat(o_input, 1, repeat)
        layer = self.relu1(self.fc1(o_input))
        layer = self.relu2(self.fc2(layer))
        action  = self.tanh(self.fc3(layer))
        return action


def extend_and_repeat(tensor, dim, repeat):
    # Extend and repeast the tensor along dim axie and repeat it
    ones_shape = [1 for _ in range(tensor.ndim + 1)]
    ones_shape[dim] = repeat
    return torch.unsqueeze(tensor, dim) * tensor.new_ones(ones_shape)

