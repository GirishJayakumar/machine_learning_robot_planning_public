import numpy as np
from robot_planning.environment.dynamics.simulated_dynamics import NumpySimulatedDynamics
from robot_planning.factory.factory_from_config import factory_from_config
from robot_planning.factory.factories import cost_evaluator_factory_base
import ast
try:
    import ConfigParser
except ImportError:
    import configparser as ConfigParser

class BicycleDynamics(NumpySimulatedDynamics):
    def __init__(self, start_state=None, dynamics_type=None, data_type=None, delta_t=None, mass=None, cog_pos=None, car_length=None, cost_evaluator=None):
        NumpySimulatedDynamics.__init__(self, dynamics_type, data_type, delta_t)
        self.dynamics_type = dynamics_type
        self.data_type = data_type
        self.delta_t = delta_t
        self.mass = mass
        self.cog_pos = cog_pos
        self.car_length = car_length
        self.state = start_state
        self.cost_evaluator = cost_evaluator  # cost needs to be set

    def initialize_from_config(self, config_data, section_name):
        NumpySimulatedDynamics.initialize_from_config(self, config_data, section_name)
        self.mass = config_data.getfloat(section_name, 'mass')
        self.cog_pos = config_data.getfloat(section_name, 'cog_pos')
        self.car_length = config_data.getfloat(section_name, 'car_length')
        self.state = np.asarray(ast.literal_eval(config_data.get(section_name, 'start_state')))
        cost_evaluator_section_name = config_data.get(section_name, 'cost_evaluator')
        self.cost_evaluator = factory_from_config(cost_evaluator_factory_base, config_data, cost_evaluator_section_name)

    def propagate(self, action, delta_t=None):
        if self.state.size != 5:
            raise ValueError("Wrong state size! The bicycle model state has a dimensionality of 5")
        if action.size != 2:
            raise ValueError("Wrong state size! The bicycle model input has a dimensionality of 2")
        x = self.state[0]
        y = self.state[1]
        Phi = self.state[2]
        delta = self.state[3]
        v = self.state[4]

        lr = np.dot(self.car_length, self.cog_pos)
        beta = np.arctan(lr * np.tan(delta) / self.car_length)

        dxdt = v * np.cos(beta + Phi)
        dydt = v * np.sin(beta + Phi)
        dPhidt = v * np.tan(delta) * np.cos(beta) / self.car_length
        ddeltadt = action[0]
        dvdt = action[1]/self.mass

        x = x + dxdt * self.delta_t
        y = y + dydt * self.delta_t
        Phi = Phi + dPhidt * self.delta_t
        delta = delta + ddeltadt * self.delta_t
        v = v + dvdt * self.delta_t

        state_next = np.array([x, y, Phi, delta, v]).reshape((5,))
        self.state = state_next
        cost = self.cost_evaluator.evaluate(state_next, action)
        return state_next, cost

    def set_state(self, state):
        self.state = state

    def set_cost_evaluator(self, cost_evaluator):
        self.cost_evaluator = cost_evaluator

    def get_state(self):
        return self.state

    def get_state_state(self):
        return self.state.state

    def get_state_dim(self):
        return (5,1)

    def get_action_dim(self):
        return (2,1)
