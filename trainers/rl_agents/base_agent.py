from robot_planning.environment.environment import Environment
import ast
import numpy as np

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

    def initialize_from_config(self, config_data, section_name):
        self.agent_index = config_data.getint(section_name, 'index')
        self.framework = config_data.get(section_name, 'type')
        self.actor_hidden_dims = list(ast.literal_eval(config_data.get(section_name, 'actor_hidden_dims')))
        self.critic_hidden_dims = list(ast.literal_eval(config_data.get(section_name, 'critic_hidden_dims')))
        self.actor_lr = config_data.getfloat(section_name, 'actor_lr')
        self.critic_lr = config_data.getfloat(section_name, 'critic_lr')
        self.tau = config_data.getfloat(section_name, 'tau')
        self.gamma = config_data.getfloat(section_name, 'gamma')

    def initialize_from_env(self, env: Environment):
        raise NotImplementedError

    def initialize_networks(self):
        raise NotImplementedError

    def step(self, full_state, epsilon_greedy=0.1, noise_rate=0.1, exploration=True):
        obs = self._generate_actor_input(full_state)
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

    def _generate_actor_input(self, full_state):
        '''
        Given the full state information of the N-agent system, generate the proper input to the actor network
        :param full_state: full state information
        :return: input to actor network
        '''
        raise NotImplementedError

    def _generate_critic_input(self, full_state):
        '''
        Given the full state information of the N-agent system, generate the proper input to the critic network
        :param full_state: full state information
        :return: input to critic network
        '''
        raise NotImplementedError