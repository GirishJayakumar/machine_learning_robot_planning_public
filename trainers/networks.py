import torch.nn as nn
import torch
import torch.nn.functional as F
from typing import List


class BaseNetwork(nn.Module):
    def __init__(self, network_name: str, input_dim, output_dim, hidden_dims: List, nonlin=nn.LeakyReLU()):
        super(BaseNetwork, self).__init__()
        self.name = network_name
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dims = hidden_dims
        self.nonlin = nonlin

        # construct network
        self.network = nn.Sequential()

        self.network.add_module(self.name + "_FC0", nn.Linear(input_dim, hidden_dims[0]))
        for i in range(len(hidden_dims) - 1):
            self.network.add_module(self.name + "_nonlin{}".format(i), nonlin)
            self.network.add_module(self.name + "_FC{}".format(i + 1), nn.Linear(hidden_dims[i], hidden_dims[i + 1]))

        self.network.add_module(self.name + "_nonlin{}".format(len(hidden_dims) - 1), nonlin)
        self.network.add_module(self.name + "_FC{}".format(len(hidden_dims)), nn.Linear(hidden_dims[-1], output_dim))

    def forward(self, x, y):
        raise NotImplemented


class ActorNet(BaseNetwork):
    def __init__(self, agent_index, input_dim, output_dim, hidden_dims: List, max_action,
                 nonlin=nn.LeakyReLU()):
        super(ActorNet, self).__init__(network_name="Agent{}".format(agent_index) + "_ActorNet", input_dim=input_dim,
                                       output_dim=output_dim, hidden_dims=hidden_dims, nonlin=nonlin)

        self.max_action = torch.tensor(max_action)

    def forward(self, obs, actions=None):
        output = self.network(obs)

        if len(output.shape) == 1:
            output = torch.unsqueeze(output, dim=0)

        action = self.max_action * torch.tanh(output)

        return action


class CriticNet(BaseNetwork):
    def __init__(self, agent_index, input_dim, output_dim, hidden_dims: List, nonlin=nn.LeakyReLU()):
        super(CriticNet, self).__init__(network_name="Agent{}".format(agent_index) + "_CriticNet", input_dim=input_dim,
                                        output_dim=output_dim, hidden_dims=hidden_dims, nonlin=nonlin)

    def forward(self, obs, actions):
        # obs = [torch.tensor(obs[i], dtype=torch.float32, requires_grad=False) for i in range(len(obs))]
        # actions = [torch.tensor(actions[i], dtype=torch.float32, requires_grad=False) for i in range(len(actions))]

        obs = torch.cat(obs, dim=1)

        actions = torch.cat(actions, dim=1)

        x = torch.cat([obs, actions], dim=1)
        q = self.network(x.float())

        return q
