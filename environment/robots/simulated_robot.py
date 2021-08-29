from robot_planning.factory.factory_from_config import factory_from_config
from robot_planning.factory.factories import dynamics_factory_base
from robot_planning.factory.factories import cost_evaluator_factory_base
from robot_planning.factory.factories import controller_factory_base
from robot_planning.factory.factories import renderer_factory_base
from robot_planning.factory.factories import observer_factory_base
import numpy as np
import copy
from copy import deepcopy
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
    def __init__(self, dynamics=None, start_state=None, steps_per_action=None, data_type=None, cost_evaluator=None, controller=None, renderer=None):
        Robot.__init__(self)
        self.dynamics = dynamics
        self.start_state = start_state
        self.state = start_state
        self.cost_evaluator = cost_evaluator
        self.data_type = data_type
        self.steps_per_action = steps_per_action
        self.steps = 0
        self.controller = controller
        self.renderer = renderer

    def initialize_from_config(self, config_data, section_name):
        Robot.initialize_from_config(self, config_data, section_name)
        dynamics_section_name = config_data.get(section_name, 'dynamics')
        self.dynamics = factory_from_config(dynamics_factory_base, config_data, dynamics_section_name)
        self.state = np.asarray(ast.literal_eval(config_data.get(section_name, 'start_state')))
        self.start_state = self.state
        self.data_type = config_data.get(section_name, 'data_type')
        if config_data.has_option(section_name, 'steps_per_action'):
            self.steps_per_action = config_data.getint(section_name, 'steps_per_action')
        else:
            self.steps_per_action = 1
        if config_data.has_option(section_name, 'controller'):
            controller_section_name = config_data.get(section_name, 'controller')
            self.controller = factory_from_config(controller_factory_base, config_data, controller_section_name)
        if config_data.has_option(section_name, 'renderer'):
            renderer_section_name = config_data.get(section_name, 'renderer')
            self.renderer = factory_from_config(renderer_factory_base, config_data, renderer_section_name)
        if config_data.has_option(section_name, 'cost_evaluator'): # controller may have a different cost evaluator from the robot, if we use the robot to train rl algorithms
            cost_evaluator_section_name = config_data.get(section_name, 'cost_evaluator')
            self.cost_evaluator = factory_from_config(cost_evaluator_factory_base, config_data, cost_evaluator_section_name)
        if config_data.has_option(section_name, 'observer'):
            observer_section_name = config_data.get(section_name, 'observer')
            self.observer = factory_from_config(observer_factory_base, config_data, observer_section_name)

    @property
    def delta_t(self):
        return self.dynamics.get_delta_t()

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

    def get_renderer(self):
        return self.renderer

    def set_state(self, x):
        assert x.shape == self.get_state_dim()
        self.state = copy.copy(x)

    def set_time(self, time):
        self.steps = time/self.dynamics.get_delta_t()

    def reset_time(self):
        self.steps = 0

    def reset_state(self, initial_state, random):
        if random:
            state_shape = self.dynamics.get_state_dim()
            state_bounds = self.dynamics.get_state_bounds()

            new_state = []
            for i in range(state_shape[0]):
                new_state.append(np.random.random()*2*state_bounds[i] - state_bounds[i])
            new_state = np.array(new_state)
            self.state = deepcopy(new_state)

        else:
            if initial_state is not None:
                self.state = initial_state
            else:
                self.state = deepcopy(self.start_state)

    def set_cost_evaluator(self, cost_evaluator):
        self.cost_evaluator = cost_evaluator

    def set_renderer(self, renderer):
        # the renderer is only responsible for renderering the robot itself and visualizing its controller info(such as MPPI)
        self.renderer = renderer
        if self.controller is not None:
            self.controller.set_renderer(renderer)

    def render_robot_state(self):
        if self.renderer is not None:
            self.renderer.render_states(state_list=[self.get_state()], kinematics_list=[self.controller.cost_evaluator.collision_checker.kinematics])

    def render_obstacles(self):
        if self.renderer is not None:
            obstacle_list = self.cost_evaluator.collision_checker.get_obstacle_list()
            self.renderer.render_obstacles(obstacle_list=obstacle_list, **{'color': "k"})

    def render_goal(self):
        if self.renderer is not None:
            goal = self.cost_evaluator.goal_checker.get_goal()
            self.renderer.render_goal(goal=goal, **{'color': "g"})

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
        self.render_robot_state()
        for _ in range(self.steps_per_action):
            state_next = self.propagate_robot(action)
            cost += self.evaluate_state_action_pair_cost(state_next, action)
        assert state_next is not None, 'invalid state!'
        return state_next, cost

    def take_action_sequence(self, actions):
        assert isinstance(actions, np.ndarray), 'simulated robot has numpy.ndarray type action!'
        state_next = None
        cost = 0
        self.render_robot_state()
        for action in actions:
            state_next = self.propagate_robot(action)
            cost += self.evaluate_state_action_pair_cost(state_next, action)
        assert state_next is None, 'invalid state!'
        return state_next, cost

    def propagate_robot_with_controller(self):
        action = self.controller.plan(state_cur=self.get_state())
        self.render_robot_state()
        state_next = self.dynamics.propagate(self.state, action)
        self.set_state(state_next)
        self.steps += 1
        return state_next

    def take_action_with_controller(self):
        state_next = None
        cost = 0
        action = self.controller.plan(state_cur=self.get_state())
        self.render_robot_state()
        self.render_goal()
        self.render_obstacles()
        for _ in range(self.steps_per_action):
            state_next = self.propagate_robot(action)
            cost += self.evaluate_state_action_pair_cost(state_next, action)
        assert state_next is not None, 'invalid state!'
        return state_next, cost
