from robot_planning.trainers.replay_buffer import ReplayBuffer
from robot_planning.trainers.utils import list_np2list_tensor, soft_update
import ast
import numpy as np
import torch

class BaseAgent:
    def __init__(self):
        self.agent_index = None
        self.framework = None
        self.max_action = None

        self.action_dim = None
        self.obs_dim = None
        self.actor_in_dim = None
        self.critic_in_dim = None

        self.gamma = None
        self.tau = None

        self.actor = None
        self.critic = None
        self.target_actor = None
        self.target_critic = None

        self.actor_hidden_dims = None
        self.critic_hidden_dims = None
        self.actor_lr = None
        self.critic_lr = None

        self.actor_optimizer = None
        self.critic_optimizer = None

        self.replay_buffer = None

        self.agent_list = None

    def initialize_from_config(self, config_data, section_name):
        self.agent_index = config_data.getint(section_name, 'index')
        self.framework = config_data.get(section_name, 'type')
        self.actor_hidden_dims = list(ast.literal_eval(config_data.get(section_name, 'actor_hidden_dims')))
        self.critic_hidden_dims = list(ast.literal_eval(config_data.get(section_name, 'critic_hidden_dims')))
        self.actor_lr = config_data.getfloat(section_name, 'actor_lr')
        self.critic_lr = config_data.getfloat(section_name, 'critic_lr')
        self.tau = config_data.getfloat(section_name, 'tau')
        self.gamma = config_data.getfloat(section_name, 'gamma')
        self.replay_buffer = ReplayBuffer()
        self.replay_buffer.initialize_from_config(config_data, section_name)

    def initialize_from_env(self, env):
        raise NotImplementedError

    def set_agent_list(self, agent_list):
        pass

    def initialize_networks(self):
        raise NotImplementedError

    def step(self, obs, epsilon_greedy=0.1, noise_rate=0.1, exploration=True):
        if np.random.uniform() < epsilon_greedy:
            action = np.random.uniform(-self.max_action, self.max_action, self.action_dim)
            return action

        else:
            policy_out = self.actor(obs)
            action = policy_out.detach().cpu().numpy()

            # if no exploration, directly output the action
            if not exploration:
                return action.copy()

            # add gaussian noise
            noise = noise_rate * self.max_action * np.random.randn(*action.shape)
            action += noise

        return action.copy()

    def train_agent(self):
        raise NotImplementedError


    def train_AC(self, transitions, next_actions):
        observations, actions, rewards, next_observations, done = transitions

        observations = list_np2list_tensor(observations)
        actions = list_np2list_tensor(actions)
        rewards = list_np2list_tensor(rewards)
        next_observations = list_np2list_tensor(next_observations)
        done = list_np2list_tensor(done)
        next_actions = list_np2list_tensor(next_actions)

        with torch.no_grad():
            # Critic Loss
            q_next = self.target_critic(next_observations, next_actions).detach()
            target_q = (rewards[self.agent_index] + self.gamma * q_next).detach()

        q = self.critic(observations, actions)

        critic_loss = (target_q - q).pow(2).mean()

        # dot = make_dot(critic_loss)
        # dot.format = 'png'
        # dot.render(filename='torchviz-sample')

        # Actor Loss
        actions[self.agent_index] = self.actor(observations[self.agent_index])

        actor_loss = - self.critic(observations, actions).mean()

        # Update Actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Update Critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Soft update target nets
        soft_update(source_net=self.critic, target_net=self.target_critic, tau=self.tau)
        soft_update(source_net=self.actor, target_net=self.target_actor, tau=self.tau)
