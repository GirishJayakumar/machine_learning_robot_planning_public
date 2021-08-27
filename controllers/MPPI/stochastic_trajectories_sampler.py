import numpy as np
import ast
import copy
from robot_planning.factory.factory_from_config import factory_from_config
from robot_planning.factory.factories import noise_sampler_factory_base


class StochasticTrajectoriesSampler():
    def __init__(self, number_of_trajectories=None, uncontrolled_trajectories_portion=None, noise_sampler=None):
        self.number_of_trajectories = number_of_trajectories
        self.uncontrolled_trajectories_portion = uncontrolled_trajectories_portion
        self.noise_sampler = noise_sampler

    def initialize_from_config(self, config_data, section_name):
        self.number_of_trajectories = int(config_data.getfloat(section_name, 'number_of_trajectories'))
        self.uncontrolled_trajectories_portion = config_data.getfloat(section_name, 'uncontrolled_trajectories_portion')
        noise_sampler_section_name = config_data.get(section_name, 'noise_sampler')
        self.noise_sampler = factory_from_config(noise_sampler_factory_base, config_data, noise_sampler_section_name)

    def sample(self, state_cur, v, control_horizon, control_dim, dynamics, cost_evaluator):
        raise NotImplementedError

    def get_number_of_trajectories(self):
        return copy.copy(self.number_of_trajectories)

    def set_number_of_trajectories(self, number_of_trajectories):
        self.number_of_trajectories = number_of_trajectories


class MPPIStochasticTrajectoriesSampler(StochasticTrajectoriesSampler):
    def __init__(self, number_of_trajectories=None, uncontrolled_trajectories_portion=None, noise_sampler=None):
        StochasticTrajectoriesSampler.__init__(self, number_of_trajectories, uncontrolled_trajectories_portion, noise_sampler)

    def initialize_from_config(self, config_data, section_name):
        StochasticTrajectoriesSampler.initialize_from_config(self, config_data, section_name)

    def sample(self, state_cur, v, control_horizon, control_dim, dynamics, cost_evaluator):
        #  state_cur is the current state, v is the nominal control sequence
        state_start = state_cur
        us = np.zeros((self.number_of_trajectories, control_dim, control_horizon-1))
        costs = np.zeros((self.number_of_trajectories, 1, 1))
        trajectories = []
        for i in range(self.number_of_trajectories):
            state_cur = state_start
            trajectory = np.zeros((dynamics.get_state_dim()[0], control_horizon))
            trajectory[:, 0] = state_cur
            noises = self.noise_sampler.sample(control_dim, control_horizon - 1)
            if i > (1 - self.uncontrolled_trajectories_portion) * self.number_of_trajectories:
                u = v + noises
            else:
                u = noises
            cost = 0
            for j in range(control_horizon-1):
                cost += cost_evaluator.evaluate(state_cur, u[:, j], dynamics=dynamics)
                state_cur = dynamics.propagate(state_cur, u[:, j])
                trajectory[:, j+1] = state_cur
            cost += cost_evaluator.evaluate_terminal_cost(state_cur, dynamics=dynamics)
            trajectories.append(trajectory)
            us[i, :, :] = u
            costs[i, 0, 0] = cost
        return trajectories, us, costs
