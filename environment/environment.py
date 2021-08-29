import numpy as np
import os
import math
import time
import ast
import copy
from robot_planning.factory.factory_from_config import factory_from_config
from robot_planning.factory.factories import dynamics_factory_base
from robot_planning.factory.factories import robot_factory_base, renderer_factory_base


class Environment(object):
    def __init__(self, agent_list=None, steps_per_action=None, renderer=None):
        self.agent_list = agent_list
        self.steps_per_action = steps_per_action
        self.n_agents = None
        self.renderer = renderer

    def initialize_from_config(self, config_data, section_name):
        # self.num_robots = config_data.getint(section_name, 'num_robots')
        agent_section_names = list(ast.literal_eval(config_data.get(section_name, 'agent_names')))
        self.agent_list = []
        for agent_section_name in agent_section_names:
            agent = factory_from_config(robot_factory_base, config_data, agent_section_name)
            self.agent_list.append(agent)
        for i in range(len(self.agent_list)):
            self.agent_list[i].observer.set_agent_list(agent_list=self.agent_list, agent_index=i)
            other_agents_list = copy.copy(self.agent_list)
            other_agents_list.pop(i)
            self.agent_list[i].cost_evaluator.collision_checker.set_other_agents_list(other_agents_list)
        if config_data.has_option(section_name, 'steps_per_action'):
            self.steps_per_action = config_data.getint(section_name, 'steps_per_action')
        else:
            self.steps_per_action = 1

        # renderer
        renderer_section_name = config_data.get(section_name, 'renderer')
        self.renderer = factory_from_config(renderer_factory_base, config_data, renderer_section_name)

        self.n_agents = len(self.agent_list)

    def single_step(self, actions):
        states = []
        observations = []
        costs = np.zeros(len(self.agent_list))
        for i in range(len(self.agent_list)):
            state_next = self.agent_list[i].propagate_robot(actions[i])
            states.append(state_next)
        for i in range(len(self.agent_list)):
            cost = self.agent_list[i].evaluate_state_action_pair_cost(states[i], actions[i])
            costs[i] = cost
            observation_next = self.agent_list[i].observer.observe()
            observations.append(observation_next)
        rl_rewards = - costs
        return states, observations, costs, rl_rewards

    def step(self, actions):
        costs_sum = np.zeros(len(self.agent_list))
        rl_rewards_sum = np.zeros(len(self.agent_list))
        states, observations = None, None
        for i in range(self.steps_per_action):
            states, observations, costs, rl_rewards = self.single_step(actions)
            costs_sum += costs
            rl_rewards_sum += rl_rewards
        if states is None:
            raise ValueError('States are None')
        if observations is None:
            raise ValueError('observations are None')
        return states, observations, costs_sum, rl_rewards_sum

    def reset(self, initial_states=None, random=False):
        states = []
        observations = []
        costs = []

        collision_free = False
        n_attempts = 0
        while not collision_free:
            for i in range(len(self.agent_list)):
                if initial_states is not None:
                    initial_state = initial_states[i]
                else:
                    initial_state = None

                self.agent_list[i].reset_state(initial_state=initial_state, random=random)
                self.agent_list[i].reset_time()
                state = self.agent_list[i].get_state()
                states.append(state)
            costs.append(None)

            collision_free = not any([agent.cost_evaluator.collision_checker.check(agent.get_state()) for agent in self.agent_list])
            n_attempts += 1
            if n_attempts >= 10:
                raise Exception("initial state cannot be randomly initialized! check parameters!")

        for i in range(len(self.agent_list)):
            observation = self.agent_list[i].observer.observe()
            observations.append(observation)
        return states, observations, costs

    def render(self):
        # render robot states
        self.renderer.render_states(state_list=[agent.get_state() for agent in self.agent_list],
                                    kinematics_list=[agent.cost_evaluator.collision_checker.kinematics for
                                                     agent in self.agent_list])
        # render goal
        goal_list = [agent.cost_evaluator.goal_checker.get_goal() for agent in self.agent_list]
        self.renderer.render_goal(goal=goal_list[0], **{'color': "g"})
        self.renderer.render_goal(goal=goal_list[1], **{'color': "r"})

        # render obstacles
        obstacle_list = self.agent_list[0].cost_evaluator.collision_checker.get_obstacle_list()
        self.renderer.render_obstacles(obstacle_list=obstacle_list, **{'color': "k"})



    def get_all_state_dims(self):
        all_state_dims = [agent.dynamics.get_state_dim() for agent in self.agent_list]
        return all_state_dims

    def get_all_action_dims(self):
        all_action_dims = [agent.dynamics.get_action_dim() for agent in self.agent_list]
        return all_action_dims

    def get_all_obs_dims(self):
        all_obs_dims = [agent.observer.get_obs_dim() for agent in self.agent_list]
        return all_obs_dims
