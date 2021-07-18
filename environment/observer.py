import numpy as np
from robot_planning.environment.environment import Environment

class Observer(object):
    def __init__(self):
        self.obs_dim = None

    def initialize_from_config(self, config_data, section_name):
        raise NotImplementedError

    def initialize_from_env(self, env: Environment):
        raise NotImplementedError

    def get_obs_dim(self):
        if self.obs_dim is None:
            raise Exception('observer not fully initialized!')
        return self.obs_dim

    def generate_observation(self, states):
        raise NotImplementedError


class FullStateObserver(Observer):
    def __init__(self):
        super(FullStateObserver).__init__()

    def initialize_from_config(self, config_data, section_name):
        pass

    def initialize_from_env(self, env: Environment):
        self.obs_dim = sum([env.agent_list[i].dynamics.get_state_dim()[0] for i in range(len(env.agent_list))])

    def generate_observation(self, states):
        obs = np.concatenate(states)
        return obs
