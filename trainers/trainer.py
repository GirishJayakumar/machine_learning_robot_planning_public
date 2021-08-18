try:
    import ConfigParser
except ImportError:
    import configparser as ConfigParser

from robot_planning.environment.environment import Environment
from robot_planning.trainers.utils import np2tensor
from robot_planning.factory.factory_from_config import factory_from_config
from robot_planning.factory.factories import rl_agent_factory_base
import os
from pathlib import Path
from robot_planning.utils import DATA_ROOT_DIR
from robot_planning.trainers.utils import list_np2list_tensor

from copy import deepcopy
import torch
import numpy as np
import ast
from tqdm import tqdm


class Trainer(object):
    class HyperParameters:
        def __init__(self, episode_len=None, n_episodes=None, seed=None, noise_rate=None, epsilon=None):
            self.n_episodes = n_episodes
            self.episode_len = episode_len
            self.seed = seed
            self.noise_rate = noise_rate
            self.epsilon = epsilon

        def initialize_from_config(self, config_data, section_name):
            # set episode length and number of episodes
            self.n_episodes = config_data.getint(section_name, 'n_episodes')
            self.episode_len = config_data.getint(section_name, 'episode_length')

            # Init exploration parameters
            self.epsilon = float(config_data.get(section_name, 'epsilon_greedy'))
            self.noise_rate = float(config_data.get(section_name, 'noise_rate'))

            # set random seeds
            if config_data.has_option(section_name, 'random_seed'):
                self.seed = config_data.getint(section_name, 'random_seed')
                torch.manual_seed(self.seed)
                np.random.seed(self.seed)

    def __init__(self, env=None, agents=None, data_path=None, device=None):
        self.env = env
        self.env_name = None
        self.agents = agents
        self.n_agents = None
        self.agent_names = None
        self.data_path = data_path
        self.device = device

        self.hyper_parameters = self.HyperParameters()

    def train(self):

        # Init reward record
        rewards = [[] for _ in range(self.n_agents)]

        # Init noise_rate and epsilon greedy
        noise_rate = self.hyper_parameters.noise_rate
        epsilon = self.hyper_parameters.epsilon

        # Start training
        for ep in tqdm(range(0, self.hyper_parameters.n_episodes), position=0, leave=True, desc="Training Episodes"):
            _, observations, _ = self.env.reset()

            for t in range(self.hyper_parameters.episode_len):
                obs_torch = list_np2list_tensor(observations)

                # generate action based on observations
                actions = self.step(obs_torch, noise_rate=noise_rate, epsilon=epsilon)
                # step environment
                _, next_observations, rewards = self.env.step(actions)

                # push into replay buffer
                self._replay_buffer_push(observations, actions, rewards, next_observations)

                # update observation
                observations = deepcopy(next_observations)

                # train only if buffer size is large enough
                if self.agents[0].replay_buffer.ready():
                    for agent in self.agents:
                        agent.train_agent()

            # update exploration
            noise_rate = max(0.05, noise_rate - 5e-7)
            epsilon = max(0.05, epsilon - 5e-7)

    def evaluate(self, initial_state=None, visualize=False):
        # TODO: implement evaluate
        pass

    def step(self, obs_torch, noise_rate, epsilon):
        actions = []
        for obs, agent in zip(obs_torch, self.agents):
            action = agent.step(obs=obs, epsilon_greedy=epsilon, noise_rate=noise_rate, exploration=True)
            action = np.squeeze(action)
            actions.append(action)
        return actions

    def initialize_from_config(self, config_data, section_name):
        # Init environment
        self.env = self._init_env(config_data, section_name)

        # Init path
        self.data_path = self._init_data_path()

        # Init hyper parameters
        hyper_parameter_section_name = config_data.get(section_name, 'hyper_parameters')
        self.hyper_parameters = self.HyperParameters()
        self.hyper_parameters.initialize_from_config(config_data, hyper_parameter_section_name)

        # set device
        self.device = torch.device("cpu")
        if config_data.has_option(section_name, 'device'):
            if config_data.get(section_name, 'device') == 'cuda':
                self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Init agents
        self.n_agents = len(self.env.agent_list)
        self.agent_names = list(ast.literal_eval(config_data.get(section_name, 'agent_names')))
        self.agents = self._init_agents(config_data)

    def _replay_buffer_push(self, observations, actions, rewards, next_observations):
        for agent in self.agents:
            agent.replay_buffer.push(observations, actions, rewards, next_observations)

    def _init_data_path(self):
        model_dir = Path(DATA_ROOT_DIR) / self.env_name
        if not model_dir.exists():
            os.makedirs(model_dir)
        return model_dir

    def _init_agents(self, config_data):
        agents = []
        for agent_name in self.agent_names:
            rl_agent = factory_from_config(rl_agent_factory_base, config_data, agent_name)
            rl_agent.initialize_from_env(self.env)
            rl_agent.initialize_networks()
            agents.append(rl_agent)
        for agent in agents:
            agent.set_agent_list(agents)
        return agents

    def _init_env(self, config_data, section_name):
        self.env_name = config_data.get(section_name, 'environment_name')

        env_config_path = "configs/envs/{}.cfg".format(self.env_name)
        env_config_data = ConfigParser.ConfigParser()
        env_config_data.read(env_config_path)

        environment = Environment()
        environment.initialize_from_config(env_config_data, 'environment')
        return environment
