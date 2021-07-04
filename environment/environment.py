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
    def __init__(self, agent_list=None):
        self.agent_list = agent_list

    def initialize_from_config(self, config_data, section_name):
        # self.num_robots = config_data.getint(section_name, 'num_robots')
        agent_section_names = list(ast.literal_eval(config_data.get(section_name, 'agent_names')))
        self.agent_list = []
        for agent_section_name in agent_section_names:
            agent = factory_from_config(robot_factory_base, config_data, agent_section_name)
            self.agent_list.append(agent)
        for i in range(len(self.agent_list)):
            other_agents_list = copy.copy(self.agent_list)
            other_agents_list.pop(i)
            self.agent_list[i].cost_evaluator.collision_checker.set_other_agents_list(other_agents_list)

    def step(self, actions):
        states = []
        costs = []
        for i in range(len(self.agent_list)):
            # print("Debug: ", actions[:, i], " ", actions[:, i].shape)
            state_next, cost = self.agent_list[i].take_action(actions[:, i])
            states.append(state_next)
            costs.append(cost)
        return states, costs
