from robot_planning.factory.factory_from_config import factory_from_config
from robot_planning.factory.factories import dynamics_factory_base
from robot_planning.factory.factories import cost_evaluator_factory_base
from robot_planning.factory.factories import controller_factory_base
from robot_planning.factory.factories import renderer_factory_base
from robot_planning.factory.factories import observer_factory_base
from robot_planning.environment.robots.base_robot import Robot
import numpy as np
import copy
from copy import deepcopy
import ast
import time


class SimulatedRobot(Robot):
    def __init__(self, robot_index=None, dynamics=None, start_state=None, steps_per_action=None, data_type=None,
                 cost_evaluator=None,
                 controller=None, renderer=None, observer=None):
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
        self.observer = observer
        self.robot_index = robot_index

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
        if config_data.has_option(section_name,
                                  'cost_evaluator'):  # controller may have a different cost evaluator from the robot, if we use the robot to train rl algorithms
            cost_evaluator_section_name = config_data.get(section_name, 'cost_evaluator')
            self.cost_evaluator = factory_from_config(cost_evaluator_factory_base, config_data,
                                                      cost_evaluator_section_name)
        if config_data.has_option(section_name, 'observer'):
            observer_section_name = config_data.get(section_name, 'observer')
            self.observer = factory_from_config(observer_factory_base, config_data, observer_section_name)
        if config_data.has_option(section_name, 'robot_index'):
            self.robot_index = int(config_data.get(section_name, 'robot_index'))

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
        self.steps = time / self.dynamics.get_delta_t()

    def set_goal(self, goal_state):
        self.controller.cost_evaluator.goal_checker.set_goal(goal_state)
        self.cost_evaluator.goal_checker.set_goal(goal_state)

    def reset_time(self):
        self.steps = 0

    def reset_state(self, initial_state=None, random=False):
        if random:
            state_shape = self.dynamics.get_state_dim()
            state_bounds = self.dynamics.get_state_bounds()

            new_state = []
            for i in range(state_shape[0]):
                new_state.append(np.random.random() * 2 * state_bounds[i] - state_bounds[i])
            new_state = np.array(new_state)
            self.state = deepcopy(new_state)

        else:
            if initial_state is not None:
                self.state = initial_state
            else:
                self.state = deepcopy(self.start_state)

    def reset_controller(self):
        if self.controller is not None:
            self.controller.reset()

    def set_cost_evaluator(self, cost_evaluator):
        self.cost_evaluator = cost_evaluator

    def set_renderer(self, renderer):
        # the renderer is only responsible for renderering the robot itself and visualizing its controller info(such as MPPI)
        self.renderer = renderer
        if self.controller is not None:
            self.controller.set_renderer(renderer)

    def render_robot_state(self):
        if self.renderer is not None:
            self.renderer.render_states(state_list=[self.get_state()],
                                        kinematics_list=[self.cost_evaluator.collision_checker.kinematics])

    def render_obstacles(self):
        if self.renderer is not None:
            obstacle_list = self.cost_evaluator.collision_checker.get_obstacle_list()
            self.renderer.render_obstacles(obstacle_list=obstacle_list, **{'color': "k"})

    def render_all(self):
        if self.robot_index is None or self.robot_index == 0:
            self.render_obstacles()
        self.render_goal()
        self.render_robot_state()
        if hasattr(self.renderer, 'isAgentsUpdated') and self.robot_index is not None:
            self.renderer.isAgentsUpdated[self.robot_index] = True
        self.renderer.show()
        self.renderer.clear()

    def render_goal(self):
        if self.renderer is not None:
            goal = self.cost_evaluator.goal_checker.get_goal()
            goal_color = self.cost_evaluator.goal_checker.get_goal_color()
            self.renderer.render_goal(goal=goal, color=goal_color)

    def propagate_robot(self, action):
        assert isinstance(action, np.ndarray), 'simulated robot has numpy.ndarray type action!'
        if self.renderer is not None:
            if self.renderer.active:
                self.render_all()
        state_next = self.dynamics.propagate(self.state, action)
        control_cost = 0
        self.set_state(state_next)
        self.steps += 1
        return state_next, control_cost

    def evaluate_state_action_pair_cost(self, state, action, state_next=None):
        cost = self.cost_evaluator.evaluate(state.reshape((-1, 1)), action.reshape((-1, 1)), dynamics=self.dynamics)
        # print('Robot index {}, cost = {}'.format(self.robot_index, cost))
        return cost

    def take_action(self, action):
        assert isinstance(action, np.ndarray), 'simulated robot has numpy.ndarray type action!'
        state_next = None
        cost = 0
        # self.render_robot_state()
        for _ in range(self.steps_per_action):
            state_next, control_cost = self.propagate_robot(action)
            cost += self.evaluate_state_action_pair_cost(state=state_next, action=action)
        assert state_next is not None, 'invalid state!'
        return state_next, cost

    def take_action_sequence(self, actions):
        assert isinstance(actions, np.ndarray), 'simulated robot has numpy.ndarray type action!'
        state_next = None
        cost = 0
        for action in actions:
            state_next, control_cost = self.propagate_robot(action)
            cost += self.evaluate_state_action_pair_cost(state=state_next, action=action)
        return state_next, cost

    def propagate_robot_with_controller(self, steps):
        state_next = None
        assert steps > 0
        for step in range(steps):
            action = self.controller.plan(state_cur=self.get_state())
            # self.render_robot_state()
            # state_next = self.dynamics.propagate(self.state, action)
            # self.set_state(state_next)
            state_next, control_cost = self.propagate_robot(action)
            self.steps += 1
        return state_next

    def take_action_with_controller(self):
        state_next = None
        cost = 0
        warm_start = True if self.steps == 0 else False

        action = self.controller.plan(state_cur=self.get_state(), warm_start=warm_start)
        for _ in range(self.steps_per_action):
            # print(self.get_state())
            state_next, control_cost = self.propagate_robot(action)
            cost += self.evaluate_state_action_pair_cost(state=state_next, action=action)
        assert state_next is not None, 'invalid state!'
        return state_next, cost
