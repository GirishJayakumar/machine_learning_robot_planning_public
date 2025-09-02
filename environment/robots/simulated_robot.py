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
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, WhiteKernel, Matern, ExpSineSquared
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
        self.opponent_inputs = np.zeros((1, 8))
        self.opponent_outputs = np.zeros((1, 8))
        noise_level = 0.01  # Initial noise level (this can be adjusted)
        noise_level_bounds = (1e-10, 0.1)  # Bounds on the noise level (this can also be adjusted)
        #hyperparameter tuning
        self.old_opp_state = None



    def initialize_from_config(self, config_data, section_name):
        Robot.initialize_from_config(self, config_data, section_name)
        dynamics_section_name = config_data.get(section_name, 'dynamics')
        self.dynamics = factory_from_config(dynamics_factory_base, config_data, dynamics_section_name)
        self.state = np.asarray(ast.literal_eval(config_data.get(section_name, 'start_state')))
        self.start_state = self.state
        self.cost_evaluator = config_data.get(section_name, 'cost_evaluator')
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
        self.opponent_inputs = np.zeros((1, 8))
        self.opponent_outputs = np.zeros((1, 8))
        # Adjusted bounds
        self.kernel = C(constant_value_bounds=(1e-6, 1e7)) * (RBF(length_scale_bounds=(1e-5, 1e9)) +
                                                              Matern(length_scale_bounds=(1e-5, 1e9)) +
                                                              ExpSineSquared(length_scale_bounds=(1e-5, 1e9), periodicity_bounds=(1e-5, 1e7))) + WhiteKernel(noise_level_bounds=(1e-13, 1e3))
        self.gp_model = GaussianProcessRegressor(kernel=self.kernel, alpha=0.0001, n_restarts_optimizer=10)

    #self.dynamics.track.localize
    def train_gp_model(self):
        self.gp_model.fit(self.opponent_inputs, self.opponent_outputs)

    def predict_opponent_state(self, map_opponent_state):
        # Predict the next state of the opponent using the GP model
        X = map_opponent_state.reshape(1, -1)
        Y_pred = self.gp_model.predict(X)
        opponent_state_pred = Y_pred.mean(axis=0)
        return opponent_state_pred

    def convertCartesianToMap(self, opp_state):
        new_opp_state = np.copy(opp_state)
        x_y_coords = [opp_state[-2], opp_state[-1]]
        M = np.array([[x_y_coords[0]], [x_y_coords[1]]])
        new_header, d_val, s_val = self.dynamics.track.localize(M, new_opp_state[-3])
        new_opp_state[-3], new_opp_state[-2], new_opp_state[-1] = new_header, d_val, s_val
        return new_opp_state
    @property
    def delta_t(self):
        return self.dynamics.get_delta_t()

    def get_state(self):
        return copy.copy(self.state)

    def get_time(self):
        return self.steps * self.dynamics.get_delta_t()

    def get_state_variance(self):
        # Return the current covariance matrix of the robot's state
        return self.covariance

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

    def propagate_robot(self, action, opp_state=None):
        assert isinstance(action, np.ndarray), 'simulated robot has numpy.ndarray type action!'
        if self.renderer is not None:
            if self.renderer.active:
                self.render_all()
        if opp_state is not None:
            opp_state_pred, _ = self.predict_gp(opp_state)
            state_next = self.dynamics.propagate(self.state, action, opp_state=opp_state_pred)
        else:
            state_next = self.dynamics.propagate(self.state, action)
        control_cost = 0
        self.set_state(state_next)
        self.steps += 1
        return state_next, control_cost

    def evaluate_state_action_pair_cost(self, state, action, state_next=None):
        #not needed
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

    # don't know if this method is necessary
    def predict_gp(self, obs):
        #not needed
        obs_2d = np.reshape(obs, (-1, 1))
        return self.gp.predict(obs_2d, return_std=True)

    def update_training_data(self, opp_state):
        # If no old state is stored, update it and return
        if self.old_opp_state is not None:
            self.opponent_inputs = np.vstack((self.opponent_inputs, self.old_opp_state))
            self.opponent_outputs = np.vstack((self.opponent_outputs, opp_state))
        self.old_opp_state = opp_state

    def take_action_with_controller(self, opp_state):
        if self.renderer is not None and self.renderer.active:
            self.render_all()
        state_cur = self.get_state()
        #map_opp_state = self.convertCartesianToMap(opp_state)
        # Predict opponent's state using GP model
        if self.gp_model is not None:
            # Add new state to training data
            self.update_training_data(opp_state)#map_opp_state)

            # Train GP model
            start_time = time.time()
            self.train_gp_model()

            # Predict the opponent's state
            opp_state_pred = self.predict_opponent_state(opp_state)#map_opp_state)
            print("opp_state_pred:", opp_state_pred)
            end_time = time.time()

            # Print debug info
            time_to_train_and_predict = end_time - start_time
            print(f"Time to train and predict: {time_to_train_and_predict:.4f}")
            for hp in self.kernel.hyperparameters:
                print('gp', hp)
            params = self.gp_model.kernel_
            print(params)

        # Plan control action
        action = self.controller.plan(state_cur=state_cur, opp_state=opp_state_pred)

        # Apply control action to propagate robot state
        state_next = self.dynamics.propagate(state_cur, action)

        # Evaluate cost
        cost = self.cost_evaluator.evaluate(state_cur, action, state_next)

        # Update robot state
        self.set_state(state_next)
        self.steps += 1

        return state_next, cost, opp_state_pred
