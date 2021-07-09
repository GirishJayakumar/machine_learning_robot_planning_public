from robot_planning.environment.environment import Environment
from robot_planning.trainers.utils import np2tensor
from copy import deepcopy
import torch

class BaseTrainer(object):
    def __init__(self, env: Environment, agents = None, trainer_type=None, trainer_config_data=None):
        self.n_agents = len(env.agent_list)
        self.trainer_type = trainer_type
        self.config_data = trainer_config_data
        self.env = env
        self.agents = agents

    def initialize_from_config(self, config_data):
        self.trainer_type = config_data.

    def _init_agents(self):
