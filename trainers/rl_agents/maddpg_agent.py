from robot_planning.trainers.rl_agents.base_agent import BaseAgent
from robot_planning.environment.environment import Environment
from robot_planning.trainers.networks import ActorNet, CriticNet
from robot_planning.trainers.utils import hard_update, soft_update

from torch.optim import Adam


class MADDPG_Agent(BaseAgent):
    def __init__(self):
        super(MADDPG_Agent).__init__()

    def initialize_from_env(self, env: Environment):
        self.max_action = env.agent_list[self.agent_index].dynamics.get_max_action()
        self.action_dim = env.agent_list[self.agent_index].dynamics.get_action_dim()[0]
        self.obs_dim = env.agent_list[self.agent_index].dynamics.get_state_dim()[0]
        self.critic_in_dim = sum([env.get_all_state_dims()[_][0] for _ in range(len(env.get_all_state_dims()))])
        self.actor_in_dim = self.obs_dim

    def initialize_networks(self):
        self.actor = ActorNet(agent_index=self.agent_index,
                              input_dim=self.actor_in_dim, output_dim=self.action_dim,
                              hidden_dims=self.actor_hidden_dims, max_action=self.max_action)

        self.critic = CriticNet(agent_index=self.agent_index,
                                input_dim=self.critic_in_dim, output_dim=1,
                                hidden_dims=self.critic_hidden_dims)

        # Target networks
        self.actor_target = ActorNet(agent_index=self.agent_index,
                                     input_dim=self.actor_in_dim, output_dim=self.action_dim,
                                     hidden_dims=self.actor_hidden_dims, max_action=self.max_action)

        self.critc_target = CriticNet(agent_index=self.agent_index,
                                      input_dim=self.critic_in_dim, output_dim=1,
                                      hidden_dims=self.critic_hidden_dims)

        # sync networks and target networks
        hard_update(target_net=self.actor_target, source_net=self.actor)
        hard_update(target_net=self.critc_target, source_net=self.critic)

        # set up optimizer
        self.actor_optimizer = Adam(self.actor.parameters(), lr=self.actor_lr)
        self.critic_optimizer = Adam(self.critic.parameters(), lr=self.critic_lr)

    def _generate_actor_input(self, full_state):
        actor_input = full_state[self.agent_index]
        return actor_input

    def _generate_critic_input(self, full_state):
        critic_input = full_state
        return critic_input