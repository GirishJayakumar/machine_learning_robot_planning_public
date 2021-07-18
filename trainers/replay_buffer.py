import numpy as np
from torch import Tensor
from torch.autograd import Variable
import torch
from robot_planning.environment.environment import Environment


class ReplayBuffer(object):
    def __init__(self):
        '''
        Experience replay buffer
        :param max_size (int): maximum buffer size
        '''

        self.max_size = None
        self.n_agents = None
        self.obs_dims = None
        self.action_dims = None

        self.buffer = []

    def initialize_from_config(self, config_data, section_name):
        self.max_size = config_data.getint(section_name, 'buffer_size')

    def initialize_from_env(self, env: Environment):
        self.n_agents = len(env.agent_list)
        self.obs_dims = [env.agent_list[i].dynamics.get_obs_dim()[0] for i in range(self.n_agents)]
        self.obs_dims = [env.agent_list[i].dynamics.get_action_dim()[0] for i in range(self.n_agents)]

    def __len__(self):
        return len(self.buffer)

    def _n_array(self, n_agent, batch_size: int, dim):
        return [np.zeros((batch_size, dim[i])) for i in range(n_agent)]

    def push(self, observations, actions, rewards, next_observations, done):
        done_mask = [1 if done[i] else 0 for i in range(self.n_agents)]
        transition = [observations, actions, rewards, next_observations, done_mask]
        self.buffer.append(transition)

        if len(self.buffer) > self.max_size:
            del self.buffer[0:int(len(self.buffer) / 10)]

    def sample(self, batch_size: int):
        # indices = np.random.choice(len(self.buffer), size=batch_size, replace=False)
        indices = np.random.randint(0, len(self.buffer), batch_size)

        n = self.n_agents
        observations = self._n_array(n_agent=n, batch_size=batch_size, dim=self.obs_dims)
        actions = self._n_array(n_agent=n, batch_size=batch_size, dim=self.action_dims)
        rewards = self._n_array(n_agent=n, batch_size=batch_size, dim=[int(1) for _ in range(n)])
        next_observations = self._n_array(n_agent=n, batch_size=batch_size, dim=self.obs_dims)
        done = self._n_array(n_agent=n, batch_size=batch_size, dim=[int(1) for _ in range(n)])

        for agent_index in range(self.n_agents):
            for i, index in enumerate(indices):
                observations[agent_index][i, :] = self.buffer[index][0][agent_index]
                actions[agent_index][i, :] = self.buffer[index][1][agent_index]
                rewards[agent_index][i, :] = self.buffer[index][2][agent_index]
                next_observations[agent_index][i, :] = self.buffer[index][3][agent_index]
                done[agent_index][i, :] = self.buffer[index][4][agent_index]

        # observations = [torch.tensor(observations[i], dtype=torch.float32, requires_grad=False) for i in
        #                 range(len(observations))]
        # actions = [torch.tensor(actions[i], dtype=torch.float32, requires_grad=False) for i in range(len(actions))]
        # rewards = [torch.tensor(rewards[i], dtype=torch.float32, requires_grad=False) for i in range(len(rewards))]
        # next_observations = [torch.tensor(next_observations[i], dtype=torch.float32, requires_grad=False) for i in
        #                      range(len(next_observations))]
        # done = [torch.tensor(done[i], dtype=torch.float32, requires_grad=False) for i in
        #         range(len(done))]

        return observations, actions, rewards, next_observations, done
