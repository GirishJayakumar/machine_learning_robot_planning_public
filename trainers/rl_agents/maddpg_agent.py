from robot_planning.trainers.rl_agents.base_agent import BaseAgent
from robot_planning.environment.environment import Environment
from robot_planning.trainers.networks import ActorNet, CriticNet
from robot_planning.trainers.utils import hard_update, soft_update, np2tensor

import torch
from torch.optim import Adam
from copy import deepcopy


class MADDPG_Agent(BaseAgent):
    def __init__(self):
        super(MADDPG_Agent).__init__()

    def initialize_from_env(self, env: Environment):
        self.replay_buffer.initialize_from_env(env)
        self.max_action = env.agent_list[self.agent_index].dynamics.get_max_action()
        self.action_dim = env.agent_list[self.agent_index].dynamics.get_action_dim()[0]
        self.obs_dim = env.agent_list[self.agent_index].observer.get_obs_dim()[0]
        self.critic_in_dim = sum([env.get_all_obs_dims()[_][0] for _ in range(env.n_agents)]) + sum(
            [env.get_all_action_dims()[_][0] for _ in range(env.n_agents)])
        self.actor_in_dim = self.obs_dim

    def set_agent_list(self, agent_list):
        self.agent_list = agent_list

    def initialize_networks(self):
        self.actor = ActorNet(agent_index=self.agent_index,
                              input_dim=self.actor_in_dim, output_dim=self.action_dim,
                              hidden_dims=self.actor_hidden_dims, max_action=self.max_action)

        self.critic = CriticNet(agent_index=self.agent_index,
                                input_dim=self.critic_in_dim, output_dim=1,
                                hidden_dims=self.critic_hidden_dims)

        # Target networks
        self.target_actor = ActorNet(agent_index=self.agent_index,
                                     input_dim=self.actor_in_dim, output_dim=self.action_dim,
                                     hidden_dims=self.actor_hidden_dims, max_action=self.max_action)

        self.target_critic = CriticNet(agent_index=self.agent_index,
                                       input_dim=self.critic_in_dim, output_dim=1,
                                       hidden_dims=self.critic_hidden_dims)

        # sync networks and target networks
        hard_update(target_net=self.target_actor, source_net=self.actor)
        hard_update(target_net=self.target_critic, source_net=self.critic)

        # set up optimizer
        self.actor_optimizer = Adam(self.actor.parameters(), lr=self.actor_lr)
        self.critic_optimizer = Adam(self.critic.parameters(), lr=self.critic_lr)

    def train_agent(self):
        samples = self.replay_buffer.sample(batch_size=None)
        observations, actions, rewards, next_observations, done = samples

        next_actions = []
        with torch.no_grad():
            for agent_index in range(len(self.agent_list)):
                # next_observation_torch = next_observations_torch[agent_index]
                next_observation = next_observations[agent_index]
                # next_action = (self.agents[agent_index].target_policy(next_observation)).detach().clone()
                # next_action.requires_grad = False
                next_observation = np2tensor(next_observation)
                next_action = (self.agent_list[agent_index].target_actor(next_observation)).detach().cpu().numpy()
                next_actions.append(next_action)

        next_actions_ = deepcopy(next_actions)
        samples_ = deepcopy(samples)
        self.train_AC(transitions=samples_, next_actions=next_actions_)
