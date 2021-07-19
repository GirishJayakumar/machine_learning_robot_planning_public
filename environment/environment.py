import numpy as np
import os
import math
import time
import ast
import copy
from robot_planning.factory.factory_from_config import factory_from_config
from robot_planning.factory.factories import dynamics_factory_base
from robot_planning.factory.factories import robot_factory_base


class Environment(object):
    def __init__(self, agent_list=None, steps_per_action=None):
        self.agent_list = agent_list
        self.steps_per_action = steps_per_action

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

    def single_step(self, actions):
        states = []
        observations = []
        costs = np.zeros(len(self.agent_list))
        for i in range(len(self.agent_list)):
            state_next = self.agent_list[i].propagate_robot(actions[:, i])
            states.append(state_next)
        for i in range(len(self.agent_list)):
            cost = self.agent_list[i].evaluate_state_action_pair_cost(states[i], actions[:, i])
            costs[i] = cost
            observation_next = self.agent_list[i].observer.observe()
            observations.append(observation_next)

        return states, observations, costs

    def step(self, actions):
        costs_sum = np.zeros(len(self.agent_list))
        states, observations = None, None
        for i in range(self.steps_per_action):
            states, observations, costs = self.single_step(actions)
            costs_sum += costs
        return states, observations, costs_sum

    def reset(self):
        states = []
        observations = []
        costs = []
        for i in range(len(self.agent_list)):
            self.agent_list[i].reset_state(option='initial_state')
            state = self.agent_list[i].get_state()
            states.append(state)
            costs.append(None)
        for i in range(len(self.agent_list)):
            observation = self.agent_list[i].observer.observe()
            observations.append(observation)
        return states, observations, costs

    def get_all_state_dims(self):
        all_state_dims = [agent.dynamics.get_state_dim() for agent in self.agent_list]
        return all_state_dims

    def get_all_action_dims(self):
        all_action_dims = [agent.dynamics.get_action_dim() for agent in self.agent_list]
        return all_action_dims

    def get_all_obs_dims(self):
        all_obs_dims = [agent.observer.get_obs_dim() for agent in self.agent_list]
        return all_obs_dims
