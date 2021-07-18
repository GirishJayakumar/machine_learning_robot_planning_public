from robot_planning.factory.factory_from_config import factory_from_config
from robot_planning.factory.factories import dynamics_factory_base
from robot_planning.factory.factories import cost_evaluator_factory_base
from robot_planning.factory.factories import observer_base
import numpy as np
import copy
import ast

class Robot(object):
    def __init__(self):
        pass

    def initialize(self):
        pass

    def initialize_from_config(self, config_data, section_name):
        pass

    def initialize_loggers(self):
        pass

    def get_state(self):
        raise NotImplementedError

    def take_action(self, action):
        raise NotImplementedError

    def take_action_sequence(self, actions):
        raise NotImplementedError


class SimulatedRobot(Robot):
    def __init__(self, dynamics=None, start_state=None, steps_per_action=None, data_type=None, cost_evaluator=None):
        Robot.__init__(self)
        self.dynamics = dynamics
        self.state = start_state
        self.cost_evaluator = cost_evaluator
        self.data_type = data_type
        self.steps_per_action = steps_per_action
        self.steps = 0

    def initialize_from_config(self, config_data, section_name):
        Robot.initialize_from_config(self, config_data, section_name)
        dynamics_section_name = config_data.get(section_name, 'dynamics')
        self.dynamics = factory_from_config(dynamics_factory_base, config_data, dynamics_section_name)
        self.state = np.asarray(ast.literal_eval(config_data.get(section_name, 'start_state')))
        cost_evaluator_section_name = config_data.get(section_name, 'cost_evaluator')
        self.cost_evaluator = factory_from_config(cost_evaluator_factory_base, config_data, cost_evaluator_section_name)
        self.data_type = config_data.get(section_name, 'data_type')
        if config_data.has_option(section_name, 'steps_per_action'):
            self.steps_per_action = config_data.getint(section_name, 'steps_per_action')
        else:
            self.steps_per_action = 1
        observer_section_name = config_data.get(section_name, 'observer')
        self.observer = factory_from_config(observer_base, config_data, observer_section_name)

    def get_state(self):
        return copy.copy(self.state)

    def get_time(self):
        return self.steps * self.dynamics.get_delta_t()

    def get_state_dim(self):
        return self.dynamics.get_state_dim()

    def get_action_dim(self):
        return self.dynamics.get_action_dim()

    def get_obs_dim(self):
        return self.observer.get_obs_dim()

    def get_model_base_type(self):
        return self.dynamics.base_type

    def get_data_type(self):
        return self.data_type

    def set_state(self, x):
        assert x.shape == self.get_state_dim()
        self.state = copy.copy(x)

    def set_time(self, time):
        self.steps = time/self.dynamics.get_delta_t()

    def reset_time(self):
        self.steps = 0

    def set_cost_evaluator(self, cost_evaluator):
        self.cost_evaluator = cost_evaluator

    def propagate_robot(self, action):
        assert isinstance(action, np.ndarray), 'simulated robot has numpy.ndarray type action!'
        state_next = self.dynamics.propagate(self.state, action)
        self.set_state(state_next)
        self.steps += 1
        return state_next

    def evaluate_state_action_pair_cost(self, state, action):
        return self.cost_evaluator.evaluate(state, action)

    def take_action(self, action):
        assert isinstance(action, np.ndarray), 'simulated robot has numpy.ndarray type action!'
        state_next = None
        cost = 0
        for _ in range(self.steps_per_action):
            state_next = self.propagate_robot(action)
            cost += self.evaluate_state_action_pair_cost(state_next, action)
        assert state_next is not None, 'invalid state!'
        return state_next, cost

    def take_action_sequence(self, actions):
        assert isinstance(actions, np.ndarray), 'simulated robot has numpy.ndarray type action!'
        state_next = None
        cost = 0
        for action in actions:
            state_next = self.propagate_robot(action)
            cost += self.evaluate_state_action_pair_cost(state_next, action)
        assert state_next is None, 'invalid state!'
        return state_next, cost

    @property
    def base_type(self):
        return self.dynamics.base_type

    @property
    def delta_t(self):
        return self.dynamics.get_delta_t()

