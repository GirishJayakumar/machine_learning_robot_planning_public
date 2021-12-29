import numpy as np
from robot_planning.environment.environment import Environment


class Observer(object):
    def __init__(self, agent_index=None, agent_list=None):
        self.agent_index = agent_index
        self.agent_list = agent_list

    def initialize_from_config(self, config_data, section_name):
        pass

    def set_agent_list(self, agent_list, agent_index):
        self.agent_index = agent_index
        self.agent_list = agent_list

    def get_obs_dim(self):
        raise NotImplementedError

    def observe(self):
        raise NotImplementedError


class FullStateObserver(Observer):
    def observe(self):
        states = [self.agent_list[_].get_state() for _ in range(len(self.agent_list))]
        obs = np.concatenate(states)
        return obs

    def get_obs_dim(self):
        obs_dim = sum([self.agent_list[_].dynamics.get_state_dim()[0] for _ in range(len(self.agent_list))])
        return (obs_dim,)


class LocalStateObserver(Observer):
    def observe(self):
        state = self.agent_list[self.agent_index].get_state()
        return state

    def get_obs_dim(self):
        obs_dim = self.agent_list[self.agent_index].dynamics.get_state_dim()[0]
        return (obs_dim,)


class AbstractFullStateObserver(Observer):
    def observe(self):
        states = [self.agent_list[_].get_state() for _ in range(len(self.agent_list))]
        obs = np.concatenate(states)
        return obs

    def get_obs_dim(self):
        obs_dim = sum([self.agent_list[_].dynamics.get_state_dim()[0] for _ in range(len(self.agent_list))])
        return (obs_dim,)
