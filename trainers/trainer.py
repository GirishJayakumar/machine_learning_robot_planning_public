try:
    import ConfigParser
except ImportError:
    import configparser as ConfigParser

from robot_planning.environment.environment import Environment
from robot_planning.trainers.utils import np2tensor
from robot_planning.factory.factory_from_config import factory_from_config
from robot_planning.factory.factories import rl_agent_factory_base

from copy import deepcopy
import torch
import ast


class Trainer(object):
    def __init__(self, env=None, agents=None):
        self.env = env
        self.env_name = None
        self.agents = agents
        self.n_agents = None
        self.agent_names = None

    def initialize_from_config(self, config_data, section_name):
        # Init environment
        self.env = self._init_env(config_data, section_name)

        # Init agents
        self.n_agents = len(self.env.agent_list)
        self.agent_names = list(ast.literal_eval(config_data.get(section_name, 'agent_names')))
        self.agents = self._init_agents(config_data)


    def _init_agents(self, config_data):
        self.agents = []
        for agent_name in self.agent_names:
            rl_agent = factory_from_config(rl_agent_factory_base, config_data, agent_name)
            rl_agent.initialize_from_env(self.env)
            rl_agent.initialize_networks()
            self.agents.append(rl_agent)


    def _init_env(self, config_data, section_name):
        self.env_name = config_data.get(section_name, 'environment_name')

        env_config_path = "configs/envs/{}_environment.cfg".format(self.env_name)
        env_config_data = ConfigParser.ConfigParser()
        env_config_data.read(env_config_path)

        environment = Environment()
        environment.initialize_from_config(env_config_data, 'environment')
        return environment
