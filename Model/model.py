import torch
import torch.nn as nn

def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

def hard_update(target, source):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)

def weight_init_Xavier(module):
    if isinstance(module, nn.Linear):
        torch.nn.init.xavier_normal_(module.weight, gain=0.01)
        module.bias.data.zero_()

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

def extend_and_repeat(tensor, dim, repeat):
    # Extend and repeast the tensor along dim axie and repeat it
    ones_shape = [1 for _ in range(tensor.ndim + 1)]
    ones_shape[dim] = repeat
    return torch.unsqueeze(tensor, dim) * tensor.new_ones(ones_shape)

class Qnet(nn.Module):
    def __init__(self, o_dim, a_dim, h_size=256):
        super(Qnet,self).__init__()

        self.net = nn.Sequential(
            nn.Linear(o_dim + a_dim, h_size),
            nn.ReLU(),
            nn.Linear(h_size, h_size),
            nn.ReLU(),
            nn.Linear(h_size, 1),
        )

    @multiple_action_q_function
    def forward(self,o_input:torch.Tensor,a_input:torch.Tensor):
        inputs = torch.concat([o_input,a_input],dim=-1)
        qval = self.net(inputs)
        return torch.squeeze(qval,dim=-1)

class Policy(nn.Module):
    def __init__(self,o_dim,a_dim, h_size=256):
        super(Policy,self).__init__()
        self.net = nn.Sequential(
            nn.Linear(o_dim, h_size),
            nn.ReLU(),
            nn.Linear(h_size, h_size),
            nn.ReLU(),
            nn.Linear(h_size, a_dim),
            nn.Tanh()
        )
    def forward(self,o_input:torch.Tensor,repeat=None):
        if repeat is not None:
            o_input = extend_and_repeat(o_input, 1, repeat)
        action = self.net(o_input)
        return action