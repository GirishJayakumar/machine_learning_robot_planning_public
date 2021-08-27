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
from robot_planning.trainers.utils import list_np2list_tensor, print_table
from robot_planning.utils import ROOT_DIR
from robot_planning.trainers.utils import list_np2list_tensor

from copy import deepcopy
import torch
import numpy as np
import ast
from tqdm import tqdm


class Trainer(object):
    class HyperParameters:
        def __init__(self, episode_len=None, n_episodes=None, seed=None, noise_rate=None, epsilon=None,
                     n_eval_episodes=None, eval_per_episodes=None, eval_episode_length=None, save_per_episodes=None):
            # training
            self.n_episodes = n_episodes
            self.episode_len = episode_len
            self.seed = seed
            self.noise_rate = noise_rate
            self.epsilon = epsilon
            self.save_per_episodes = save_per_episodes

            # evaluation
            self.n_eval_episodes = n_eval_episodes
            self.eval_per_episodes = eval_per_episodes
            self.eval_episode_length = eval_episode_length

        def initialize_from_config(self, config_data, section_name):
            # set episode length and number of episodes
            self.n_episodes = config_data.getint(section_name, 'n_episodes')
            self.episode_len = config_data.getint(section_name, 'episode_length')
            self.n_eval_episodes = config_data.getint(section_name, 'n_eval_episodes')
            self.eval_per_episodes = config_data.getint(section_name, 'eval_per_episodes')
            self.eval_episode_length = config_data.getint(section_name, 'eval_episode_length')
            self.save_per_episodes = config_data.getint(section_name, 'save_per_episodes')

            # Init exploration parameters
            self.epsilon = float(config_data.get(section_name, 'epsilon_greedy'))
            self.noise_rate = float(config_data.get(section_name, 'noise_rate'))

            # set random seeds
            if config_data.has_option(section_name, 'random_seed'):
                self.seed = config_data.getint(section_name, 'random_seed')
                torch.manual_seed(self.seed)
                np.random.seed(self.seed)

    def __init__(self, env=None, agents=None, model_dir=None, device=None):
        self.env = env
        self.env_name = None
        self.agents = agents
        self.n_agents = None
        self.agent_names = None
        self.model_dir = model_dir
        self.network_dir = None
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
                _, next_observations, _, rewards = self.env.step(actions)

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

            # save model
            if ep % self.hyper_parameters.save_per_episodes == 0:
                self.save_model(episode=ep)

            # evaluate model
            if ep % self.hyper_parameters.eval_per_episodes == 0:
                self.evaluate()

        # save final network
        print(' ')
        print('##################')
        print('Tranning complete!')
        print('##################')
        self.save_model(episode=ep)

    def evaluate(self, initial_states=None, random=False, visualize=False, n_eval_episodes=None,
                 eval_episode_length=None):
        if n_eval_episodes is None:
            n_eval_episodes = self.hyper_parameters.n_eval_episodes
        if eval_episode_length is None:
            eval_episode_length = self.hyper_parameters.eval_episode_length

        returns = []
        mean_returns = [0 for _ in range(self.n_agents)]

        for ep in range(n_eval_episodes):
            episode_return = [0 for _ in range(self.n_agents)]
            _, observations, _ = self.env.reset(initial_states=initial_states, random=random)
            for t in range(eval_episode_length):
                if visualize:
                    self.env.render()
                    self.env.renderer.show()
                    self.env.renderer.clear()

                # step
                obs_torch = list_np2list_tensor(observations)
                actions = self.step(obs_torch, exploration=False)
                _, next_observations, _, rewards, = self.env.step(actions)
                observations = deepcopy(next_observations)

                # record rewards
                for agent_index in range(self.n_agents):
                    episode_return[agent_index] += rewards[agent_index]
                    mean_returns[agent_index] += rewards[agent_index]

            episode_return.insert(0, ep)
            returns.append(episode_return)

        mean_return = [mean_returns[i] / n_eval_episodes for i in range(self.n_agents)]
        mean_return.insert(0, "Mean")

        returns.append(mean_return)

        header = ["Agent {}".format(i) for i in range(self.n_agents)]
        header.insert(0, "Episode")
        print("\n")
        print_table(header=header, data=returns)
        mean_return.pop(0)

        return mean_return

    def step(self, obs_torch, noise_rate=0.1, epsilon=0.1, exploration=True):
        actions = []
        for obs, agent in zip(obs_torch, self.agents):
            action = agent.step(obs=obs, epsilon_greedy=epsilon, noise_rate=noise_rate, exploration=exploration)
            action = np.squeeze(action)
            actions.append(action)
        return actions

    def initialize_from_config(self, config_data, section_name):
        # Init environment
        self.env = self._init_env(config_data, section_name)

        # Init path
        self.model_dir, self.network_dir = self._init_data_path()

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
        model_dir = Path(ROOT_DIR) / 'data' / self.env_name
        if not model_dir.exists():
            os.makedirs(model_dir)
        network_dir = model_dir / 'network_data'
        if not network_dir.exists():
            os.makedirs(network_dir)
        return model_dir, network_dir

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

        env_config_path = "configs/environment_configs/{}.cfg".format(self.env_name)
        env_config_data = ConfigParser.ConfigParser()
        env_config_data.read(env_config_path)

        environment = Environment()
        environment.initialize_from_config(env_config_data, 'environment')
        return environment

    def save_model(self, episode: int):
        save_path = self.network_dir
        if not os.path.exists(save_path):
            raise ValueError('Network save path not initialized!')

        for agent_index, agent in enumerate(self.agents):
            actor_net_state_dict = agent.actor.state_dict()
            target_actor_net_state_dict = agent.target_actor.state_dict()
            critic_net_state_dict = agent.critic.state_dict()
            target_critic_net_state_dict = agent.target_critic.state_dict()

            actor_optimizer_state_dict = agent.actor_optimizer.state_dict()
            critic_optimizer_state_dict = agent.critic_optimizer.state_dict()

            torch.save({
                'actor_net_state_dict': actor_net_state_dict,
                "target_actor_net_state_dict": target_actor_net_state_dict,
                "critic_net_state_dict": critic_net_state_dict,
                'target_critic_net_state_dict': target_critic_net_state_dict,
                'actor_optimizer_state_dict': actor_optimizer_state_dict,
                "critic_optimizer_state_dict": critic_optimizer_state_dict},
                str(save_path) + '/' + "Episode" + str(episode) + '-Agent{}'.format(agent_index) + "-model.pt")

    def load_model(self, episode: int, training=False):
        save_path = self.network_dir

        for agent_index, agent in enumerate(self.agents):
            checkpoint = torch.load(
                str(save_path) + '/' + "Episode" + str(episode) + '-Agent{}'.format(agent_index) + "-model.pt")
            agent.actor.load_state_dict(checkpoint['actor_net_state_dict'])
            agent.target_actor.load_state_dict(checkpoint['target_actor_net_state_dict'])
            agent.critic.load_state_dict(checkpoint['critic_net_state_dict'])
            agent.target_critic.load_state_dict(checkpoint['target_critic_net_state_dict'])
            agent.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
            agent.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])

            if training:
                agent.actor.train()
                agent.target_actor.train()
                agent.critic.train()
                agent.target_critic.train()
            else:
                agent.actor.eval()
                agent.target_actor.eval()
                agent.critic.eval()
                agent.target_critic.eval()
