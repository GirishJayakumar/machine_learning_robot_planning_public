try:
    import ConfigParser
except ImportError:
    import configparser as ConfigParser

from robot_planning.environment.environment import Environment
from robot_planning.trainers.utils import np2tensor
from copy import deepcopy
import torch
import ast


class Trainer(object):
    def __init__(self, env=None, agents=None):
        self.env = env
        self.env_name = None
        self.agents = agents
        self.n_agents = None

    def initialize_from_config(self, config_data, section_name):
        self.env = self._init_env(config_data, section_name)
        self.n_agents = len(self.env.agent_list)
        agent_section_names = list(ast.literal_eval(config_data.get(section_name, 'agent_names')))

    def _init_agents(self):
        pass

    def _init_env(self, config_data, section_name):
        self.env_name = config_data.get(section_name, 'environment_name')

        env_config_path = "configs/envs/{}_environment.cfg".format(self.env_name)
        env_config_data = ConfigParser.ConfigParser()
        env_config_data.read(env_config_path)

        environment = Environment()
        environment.initialize_from_config(env_config_data, 'environment')
        return environment
