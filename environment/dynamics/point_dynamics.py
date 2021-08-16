import numpy as np
from robot_planning.environment.dynamics.simulated_dynamics import NumpySimulatedDynamics
from robot_planning.factory.factory_from_config import factory_from_config
from robot_planning.factory.factories import cost_evaluator_factory_base
import ast

try:
    import ConfigParser
except ImportError:
    import configparser as ConfigParser


def check_bounds_bounce(x, max_x):
    if abs(x) >= max_x:
        return (x / (abs(x))) * (2*max_x - abs(x))
    else:
        return x


def check_bounds_saturate(x, max_x):
    if abs(x) >= max_x:
        return (x / (abs(x))) * max_x
    else:
        return x


class PointDynamics(NumpySimulatedDynamics):
    def __init__(self, dynamics_type=None, data_type=None, delta_t=None, mass=None, cog_pos=None, car_length=None):
        NumpySimulatedDynamics.__init__(self, dynamics_type, data_type, delta_t)
        self.dynamics_type = dynamics_type
        self.data_type = data_type
        self.delta_t = delta_t
        self.mass = mass
        self.max_vx, self.max_vy, self.max_x, self.max_y = None, None, None, None

    def initialize_from_config(self, config_data, section_name):
        NumpySimulatedDynamics.initialize_from_config(self, config_data, section_name)
        self.mass = config_data.getfloat(section_name, 'mass')
        self.max_vx = config_data.getfloat(section_name, 'max_vx')
        self.max_vy = config_data.getfloat(section_name, 'max_vy')
        self.max_x = config_data.getfloat(section_name, 'max_x')
        self.max_y = config_data.getfloat(section_name, 'max_y')

    def propagate(self, state, action, delta_t=None):
        if state.size != 4:
            raise ValueError("Wrong state size! The bicycle model state has a dimensionality of 5")
        if action.size != 2:
            raise ValueError("Wrong state size! The bicycle model input has a dimensionality of 2")
        x = state[0]
        y = state[1]
        vx = state[2]
        vy = state[3]

        # derivatives
        dxdt = vx
        dydt = vy
        dvxdt = action[0] / self.mass
        dvydt = action[1] / self.mass

        # update states
        x = x + dxdt * self.delta_t
        y = y + dydt * self.delta_t
        vx = vx + dvxdt * self.delta_t
        vy = vy + dvydt * self.delta_t

        # check bounds
        x = check_bounds_bounce(x, self.max_x)
        y = check_bounds_bounce(y, self.max_y)
        vx = check_bounds_saturate(vx, self.max_vx)
        vy = check_bounds_saturate(vy, self.max_vy)

        state_next = np.array([x, y, vx, vy]).reshape((4,))
        return state_next

    def get_state_dim(self):
        return (4,)

    def get_action_dim(self):
        return (2,)

    def get_max_action(self):
        return np.array([1, 1])
