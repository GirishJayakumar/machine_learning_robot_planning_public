import numpy as np
import ast
import copy
from robot_planning.factory.factory_from_config import factory_from_config
from robot_planning.factory.factories import noise_sampler_factory_base
import multiprocessing as mp
from robot_planning.factory.factories import covariance_steering_helper_factory_base
import threading


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


class MPPIStochasticTrajectoriesSamplerSlowLoop(StochasticTrajectoriesSampler):
    def __init__(self, number_of_trajectories=None, uncontrolled_trajectories_portion=None, noise_sampler=None):
        StochasticTrajectoriesSampler.__init__(self, number_of_trajectories, uncontrolled_trajectories_portion, noise_sampler)

    def initialize_from_config(self, config_data, section_name):
        StochasticTrajectoriesSampler.initialize_from_config(self, config_data, section_name)

    def sample(self, state_cur, v, control_horizon, control_dim, dynamics, cost_evaluator):
        #  state_cur is the current state, v is the nominal control sequence
        state_start = state_cur.reshape((-1, 1))
        us = np.zeros((self.number_of_trajectories, control_dim, control_horizon-1))
        costs = np.zeros((self.number_of_trajectories, 1, 1))
        trajectories = []
        for i in range(self.number_of_trajectories):
            state_cur = state_start
            trajectory = np.zeros((dynamics.get_state_dim()[0], control_horizon))
            trajectory[:, :1] = state_cur
            noises = self.noise_sampler.sample(control_dim, control_horizon - 1)
            if i > (1 - self.uncontrolled_trajectories_portion) * self.number_of_trajectories:
                u = v + noises
            else:
                u = noises
            cost = 0
            for j in range(control_horizon-1):
                cost += cost_evaluator.evaluate(state_cur, u[:, j:j+1], dynamics=dynamics)
                state_cur = dynamics.propagate(state_cur, u[:, j]).reshape((-1, 1))
                trajectory[:, j+1:j+2] = state_cur
            cost += cost_evaluator.evaluate_terminal_cost(state_cur, dynamics=dynamics)
            trajectories.append(trajectory)
            us[i, :, :] = u
            costs[i, 0, 0] = cost
        return trajectories, us, costs


class MPPIStochasticTrajectoriesSampler(StochasticTrajectoriesSampler):
    def __init__(self, number_of_trajectories=None, uncontrolled_trajectories_portion=None, noise_sampler=None):
        StochasticTrajectoriesSampler.__init__(self, number_of_trajectories, uncontrolled_trajectories_portion, noise_sampler)

    def initialize_from_config(self, config_data, section_name):
        StochasticTrajectoriesSampler.initialize_from_config(self, config_data, section_name)

    def sample(self, state_cur, v, control_horizon, control_dim, dynamics, cost_evaluator):
        #  state_cur is the current state, v is the nominal control sequence
        state_start = state_cur.copy()
        # us = np.zeros((self.number_of_trajectories, control_dim, control_horizon-1))
        costs = np.zeros((self.number_of_trajectories, 1, 1))
        # trajectories = []
        state_cur = np.tile(state_start.reshape((-1, 1)), (1, self.number_of_trajectories))
        trajectories = np.zeros((state_cur.shape[0], control_horizon, self.number_of_trajectories))
        trajectories[:, 0, :] = state_cur
        noises = self.noise_sampler.sample(control_dim, (control_horizon - 1) * self.number_of_trajectories)
        noises = noises.reshape((control_dim, (control_horizon - 1), self.number_of_trajectories))
        num_controlled_trajectories = int((1 - self.uncontrolled_trajectories_portion) * self.number_of_trajectories)
        us = np.zeros((v.shape[0], v.shape[1], self.number_of_trajectories))
        us[:, :, :num_controlled_trajectories] = np.expand_dims(v, axis=2)
        us += noises
        # cost = 0
        for j in range(control_horizon-1):
            costs += cost_evaluator.evaluate(state_cur, us[:, j, :], noises[:, j, :], dynamics=dynamics)
            state_cur = dynamics.propagate(state_cur, us[:, j, :])
            trajectories[:, j+1, :] = state_cur
        costs += cost_evaluator.evaluate_terminal_cost(state_cur, dynamics=dynamics)
        # trajectories.append(trajectory)
        # us[i, :, :] = u
        # costs[:, 0, 0] = cost
        us = np.moveaxis(us, 2, 0)
        return trajectories, us, costs


class MPPIParallelStochasticTrajectoriesSamplerMultiprocessing(StochasticTrajectoriesSampler):
    def __init__(self, number_of_trajectories=None, uncontrolled_trajectories_portion=None, noise_sampler=None, number_of_processes=8):
        StochasticTrajectoriesSampler.__init__(self, number_of_trajectories, uncontrolled_trajectories_portion,
                                               noise_sampler)
        self.number_of_processes = 8

    def initialize_from_config(self, config_data, section_name):
        StochasticTrajectoriesSampler.initialize_from_config(self, config_data, section_name)
        if config_data.has_option(section_name, 'number_of_processes'):
            self.number_of_processes = config_data.getint(section_name, 'number_of_processes')

    def sample(self, state_cur, v, control_horizon, control_dim, dynamics, cost_evaluator):
        #  state_cur is the current state, v is the nominal control sequence
        state_start = state_cur
        us_array = np.zeros((self.number_of_trajectories, control_dim, control_horizon - 1))
        costs_array = np.zeros((self.number_of_trajectories, 1, 1))
        trajectories_list = []

        noises_queue = mp.JoinableQueue()
        results = mp.Queue()
        for i in range(self.number_of_trajectories):
            noises_queue.put(self.noise_sampler.sample(control_dim, control_horizon - 1))
        for i in range(self.number_of_processes):
            p = mp.Process(target=self.sample_single_traj, args=(state_start, dynamics, cost_evaluator, v, control_horizon, noises_queue, results, i))
            p.start()
        noises_queue.join()
        for i in range(self.number_of_trajectories):
            result = results.get()
            trajectories_list.append(result[0])
            us_array[i, :, :] = result[1]
            costs_array[i, 0, 0] = result[2]

        return trajectories_list, us_array, costs_array

    def sample_single_traj(self, state_start, dynamics, cost_evaluator, v, control_horizon, noises_queue, results, i):
        while noises_queue.empty() is False:
            noises = noises_queue.get()
            state_cur = state_start
            trajectory = np.zeros((dynamics.get_state_dim()[0], control_horizon))
            trajectory[:, 0] = state_cur
            if i > (1 - self.uncontrolled_trajectories_portion) * self.number_of_trajectories:
                u = v + noises
            else:
                u = noises
            cost = 0
            for j in range(control_horizon - 1):
                cost += cost_evaluator.evaluate(state_cur.reshape((-1, 1)), u[:, j:j+1], dynamics=dynamics)
                state_cur = dynamics.propagate(state_cur, u[:, j])
                trajectory[:, j + 1] = state_cur
            cost += cost_evaluator.evaluate_terminal_cost(state_cur.reshape((-1, 1)), dynamics=dynamics)
            results.put([trajectory, u, cost])
            noises_queue.task_done()


class CCMPPIStochasticTrajectoriesSampler(StochasticTrajectoriesSampler):
    def __init__(self, number_of_trajectories=None, uncontrolled_trajectories_portion=None, noise_sampler=None):
        StochasticTrajectoriesSampler.__init__(self, number_of_trajectories, uncontrolled_trajectories_portion,
                                               noise_sampler)

    def initialize_from_config(self, config_data, section_name):
        StochasticTrajectoriesSampler.initialize_from_config(self, config_data, section_name)
        covariance_steering_helper_section_name = config_data.get(section_name, 'covariance_steering_helper')
        self.covariance_steering_helper = factory_from_config(covariance_steering_helper_factory_base, config_data, covariance_steering_helper_section_name)

    def sample(self, state_cur, v, control_horizon, control_dim, dynamics, cost_evaluator):
        #  state_cur is the current state, v is the nominal control sequence
        state_start = state_cur
        us = np.zeros((self.number_of_trajectories, control_dim, control_horizon - 1))
        costs = np.zeros((self.number_of_trajectories, 1, 1))
        trajectories = []
        self.covariance_steering_helper.dynamics_linearizer.set_dynamics(dynamics)
        reference_trajectory = self.rollout_out(state_cur, v, dynamics)
        Ks, As, Bs, ds, Sx_cc, Sx_nocc = self.covariance_steering_helper.covariance_control(state=state_cur.T,
                                                                                            ref_state_vec=reference_trajectory.T,
                                                                                            ref_ctrl_vec=v.T,
                                                                                            return_sx=True, Sigma_epsilon=self.noise_sampler.covariance)
        for i in range(self.number_of_trajectories):
            state_cur = state_start
            trajectory = np.zeros((dynamics.get_state_dim()[0], control_horizon))
            trajectory[:, 0] = state_cur
            noises = self.noise_sampler.sample(control_dim, control_horizon - 1)
            if i > self.uncontrolled_trajectories_portion * self.number_of_trajectories - 0.001:
                u = v + noises
            else:
                u = noises
            cost = 0
            y = np.zeros(state_cur.shape)
            for j in range(control_horizon - 1):
                if i > self.uncontrolled_trajectories_portion * self.number_of_trajectories - 0.001:
                    u[:, j] = u[:, j] + np.dot(Ks[j, :, :], y)
                y = np.dot(As[j, :, :], y) + np.dot(Bs[j, :, :], noises[:, j])
                cost += cost_evaluator.evaluate(state_cur, u[:, j])
                state_cur = dynamics.propagate(state_cur, u[:, j])
                trajectory[:, j + 1] = state_cur
            cost += cost_evaluator.evaluate(
                state_cur)  # final cost TODO: add final_cost_evaluate() function to cost_evaluator
            trajectories.append(trajectory)
            us[i, :, :] = u
            costs[i, 0, 0] = cost
        return trajectories, us, costs

    def rollout_out(self, state_cur, v, dynamics):
        trajectory = np.zeros((dynamics.get_state_dim()[0], v.shape[1]+1))
        trajectory[:, 0] = state_cur
        for i in range(v.shape[1]):
            state_next = dynamics.propagate(state_cur, v[:, i])
            trajectory[:, i+1] = state_next
            state_cur = state_next
        return trajectory


