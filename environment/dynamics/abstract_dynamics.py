import numpy as np
from robot_planning.environment.dynamics.simulated_dynamics import NumpySimulatedDynamics
from robot_planning.factory.factory_from_config import factory_from_config
from robot_planning.factory.factories import cost_evaluator_factory_base, robot_factory_base
import ast

try:
    import ConfigParser
except ImportError:
    import configparser as ConfigParser


class AbstractDynamics(NumpySimulatedDynamics):
    def __init__(self, dynamics_type=None, data_type=None, delta_t=None, simulated_robot=None):
        NumpySimulatedDynamics.__init__(self, dynamics_type, data_type, delta_t)
        self.dynamics_type = dynamics_type
        self.data_type = data_type
        self.delta_t = delta_t
        self.simulated_robot = simulated_robot

    def initialize_from_config(self, config_data, section_name):
        NumpySimulatedDynamics.initialize_from_config(self, config_data, section_name)
        simulated_robot_section_name = config_data.get(section_name, 'simulated_robot')
        self.simulated_robot = factory_from_config(robot_factory_base, config_data, simulated_robot_section_name)

    def propagate(self, action, delta_t, state=None):
        assert action.shape == self.get_action_dim()
        self.simulated_robot.set_goal(action)
        state_next = self.simulated_robot.propagate_robot_with_controller(steps=delta_t)
        return state_next

    def get_action_dim(self):
        return self.simulated_robot.dynamics.cost_evaluator.goal_checker.get_goal_dim()

    def get_state_dim(self):
        return self.simulated_robot.dynamics.get_state_dim()

    def get_max_action(self):
        return self.simulated_robot.dynamics.get_state_bounds()
