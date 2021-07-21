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
        self.batch_size = None
        self.n_agents = None
        self.obs_dims = None
        self.action_dims = None
        self.buffer = []

    def __len__(self):
        return len(self.buffer)

    def ready(self):
        if len(self.buffer) >= self.batch_size:
            return True
        else:
            return False

    def initialize_from_config(self, config_data, section_name):
        self.max_size = config_data.getint(section_name, 'buffer_size')
        self.batch_size = config_data.getint(section_name, 'batch_size')

    def initialize_from_env(self, env: Environment):
        self.n_agents = len(env.agent_list)
        self.obs_dims = [env.agent_list[i].observer.get_obs_dim()[0] for i in range(self.n_agents)]
        self.action_dims = [env.agent_list[i].dynamics.get_action_dim()[0] for i in range(self.n_agents)]

    def push(self, observations, actions, rewards, next_observations, done=None):
        done_mask = [0 for i in range(self.n_agents)]
        if done is not None:
            done_mask = [1 if done[i] else 0 for i in range(self.n_agents)]
        transition = [observations, actions, rewards, next_observations, done_mask]
        self.buffer.append(transition)

        if len(self.buffer) > self.max_size:
            del self.buffer[0:int(len(self.buffer) / 10)]

    def sample(self, batch_size=None):

        if batch_size is None:
            batch_size = self.batch_size

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

        return observations, actions, rewards, next_observations, done

    def _n_array(self, n_agent, batch_size: int, dim):
        return [np.zeros((batch_size, dim[i])) for i in range(n_agent)]
