from robot_planning.controllers.controller import MpcController
from robot_planning.factory.factory_from_config import factory_from_config
from robot_planning.factory.factories import cost_evaluator_factory_base
from robot_planning.factory.factories import stochastic_trajectories_sampler_factory_base
import numpy as np
import ast
import copy


class MPPI(MpcController):
    def __init__(self, control_horizon=None, dynamics=None, cost_evaluator=None, control_dim=None, inverse_temperature=None, initial_control_sequence=None, stochastic_trajectories_sampler=None, renderer=None):
        MpcController.__init__(self, control_horizon, dynamics, cost_evaluator, control_dim, renderer)
        self.inverse_temperature = inverse_temperature
        self.initial_control_sequence = initial_control_sequence
        self.stochastic_trajectories_sampler = stochastic_trajectories_sampler

    def initialize_from_config(self, config_data, section_name):
        MpcController.initialize_from_config(self, config_data, section_name)
        self.inverse_temperature = config_data.getfloat(section_name, 'inverse_temperature')
        if config_data.has_option(section_name, 'initial_control_sequence'):
            self.initial_control_sequence = np.asarray(ast.literal_eval(config_data.get(section_name, 'initial_control_sequence')), dtype=np.float64).reshape((self.get_control_dim(), self.get_control_horizon() - 1))
            if not (self.initial_control_sequence.shape[0] is self.get_control_dim() and self.initial_control_sequence.shape[1] is self.get_control_horizon() - 1):
                raise ValueError('The initial control sequence does not match control dimensions and control horizon')
        else:
            self.initial_control_sequence = np.zeros((self.get_control_dim(), self.get_control_horizon() - 1))

        stochastic_trajectories_sampler_section_name = config_data.get(section_name, 'stochastic_trajectories_sampler')
        self.stochastic_trajectories_sampler = factory_from_config(stochastic_trajectories_sampler_factory_base, config_data, stochastic_trajectories_sampler_section_name)

    def plan(self, state_cur, warm_start_itr=1):
        v = copy.deepcopy(self.initial_control_sequence)
        for _ in range(warm_start_itr):
            trajectories, us, costs = self.stochastic_trajectories_sampler.sample(state_cur, v, self.get_control_horizon(), self.get_control_dim(), self.get_dynamics(), self.get_cost_evaluator())
            beta = np.min(costs)
            eta = np.sum(np.exp(-1/self.inverse_temperature * (costs - beta)))
            omega = 1/eta * np.exp(-1/self.inverse_temperature * (costs - beta))
            v = np.sum(omega.reshape((us.shape[0], 1, 1)) * us, axis=0)  # us shape = (number_of_trajectories, control_dim, control_horizon)
            self.set_initial_control_sequence(v)

        optimal_trajectory = self.rollout_out(state_cur, v)
        if self.renderer is not None:
            self.renderer.render_trajectories(trajectories, **{'color': "b"})
            self.renderer.render_trajectories([optimal_trajectory], **{'color': "r"})
        u = v[:, 0]
        v = np.delete(v, 0, 1)
        v = np.hstack((v, v[:, -1].reshape(v.shape[0], 1)))
        self.set_initial_control_sequence(v)
        return u

    def reset(self):
        self.initial_control_sequence = np.zeros((self.get_control_dim(), self.get_control_horizon() - 1))

    def set_initial_control_sequence(self, initial_control_sequence):
        self.initial_control_sequence = initial_control_sequence

    def rollout_out(self, state_cur, v):
        dynamics = self.get_dynamics()
        trajectory = np.zeros((dynamics.get_state_dim()[0], v.shape[1]+1))
        trajectory[:, 0] = state_cur
        for i in range(v.shape[1]):
            state_next = dynamics.propagate(state_cur, v[:, i])
            trajectory[:, i+1] = state_next
            state_cur = state_next
        return trajectory