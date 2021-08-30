import copy

import numpy as np
import ast
from robot_planning.factory.factories import collision_checker_factory_base, goal_checker_factory_base
from robot_planning.factory.factory_from_config import factory_from_config


class CostEvaluator():
    def __init__(self, goal_checker=None, collision_checker=None):
        self.goal_checker = goal_checker
        self.collision_checker = collision_checker

    def initialize_from_config(self, config_data, section_name):
        raise NotImplementedError

    def evaluate(self, state_cur, dyna_obstacle_list=None):
        raise NotImplementedError

    def set_collision_checker(self, collision_checker=None):
        self.collision_checker = collision_checker

    def set_goal_checker(self, goal_checker=None):
        self.goal_checker = goal_checker


class QuadraticCostEvaluator(CostEvaluator):
    def __init__(self, goal_checker=None, collision_checker=None, Q=None, R=None, collision_cost=None, goal_cost=None):
        CostEvaluator.__init__(self, goal_checker, collision_checker)
        self.Q = Q
        self.R = R
        self.collision_cost = collision_cost
        self.goal_cost = goal_cost

    def initialize_from_config(self, config_data, section_name):
        if config_data.has_option(section_name, 'collision_cost'):
            self.collision_cost = config_data.getfloat(section_name, 'collision_cost') # collision_cost should be positive
        if config_data.has_option(section_name, 'goal_cost'):
            self.goal_cost = config_data.getfloat(section_name, 'goal_cost') # goal_cost should be negative
        if config_data.has_option(section_name, 'Q'):
            Q = np.asarray(ast.literal_eval(config_data.get(section_name, 'Q')))
            if Q.ndim == 1:
                self.Q = np.diag(Q)
            else:
                self.Q = Q
        if config_data.has_option(section_name, 'QN'):
            QN = np.asarray(ast.literal_eval(config_data.get(section_name, 'QN')))
            if QN.ndim == 1:
                self.QN = np.diag(QN)
            else:
                self.QN = QN
        else:
            self.QN = self.Q
        if config_data.has_option(section_name, 'R'):
            R = np.asarray(ast.literal_eval(config_data.get(section_name, 'R')))
            if R.ndim == 1:
                self.R = np.diag(R)
            else:
                self.R = R
        if config_data.has_option(section_name, 'goal_checker'):
            goal_checker_section_name = config_data.get(section_name, 'goal_checker')
            self.goal_checker = factory_from_config(goal_checker_factory_base, config_data,
                                                      goal_checker_section_name)
        if config_data.has_option(section_name, 'collision_checker'):
            collision_checker_section_name = config_data.get(section_name, 'collision_checker')
            self.collision_checker = factory_from_config(collision_checker_factory_base, config_data,
                                                          collision_checker_section_name)

    def evaluate(self, state_cur, actions=None, dyna_obstacle_list=None, dynamics=None):
        cost = (1/2) * (state_cur - self.goal_checker.goal_state).T @ self.Q @ (state_cur - self.goal_checker.goal_state)
        if actions is not None:
            cost += (1/2) * actions.T @ self.R @ actions
        if self.collision_checker.check(state_cur):  # True for collision, False for no collision
            if self.collision_cost is not None:
                cost += self.collision_cost
            else:
                cost += 1000  # default collision cost
        if self.goal_checker.check(state_cur):  # True for goal reached, False for goal not reached
            if self.goal_cost is not None:
                cost += self.goal_cost
            else:
                cost += -5000  # default goal cost
        return cost

    def evaluate_terminal_cost(self, state_cur, actions=None, dyna_obstacle_list=None, dynamics=None):
        # evaluate cost of the final step of the horizon
        cost = (1 / 2) * (state_cur - self.goal_checker.goal_state).T @ self.QN @ (
                    state_cur - self.goal_checker.goal_state)
        if actions is not None:
            cost += (1 / 2) * actions.T @ self.R @ actions
        if self.collision_checker.check(state_cur):  # True for collision, False for no collision
            if self.collision_cost is not None:
                cost += self.collision_cost
            else:
                cost += 1000  # default collision cost
        if self.goal_checker.check(state_cur):  # True for goal reached, False for goal not reached
            if self.goal_cost is not None:
                cost += self.goal_cost
            else:
                cost += -5000  # default goal cost
        return cost


class AutorallyMPPICostEvaluator(QuadraticCostEvaluator):
    def __init__(self, goal_checker=None, collision_checker=None, Q=None, R=None, collision_cost=None, goal_cost=None):
        QuadraticCostEvaluator.__init__(self, goal_checker, collision_checker, Q, R, collision_cost, goal_cost)

    def initialize_from_config(self, config_data, section_name):
        QuadraticCostEvaluator.initialize_from_config(self, config_data, section_name)

    def evaluate(self, state_cur, actions=None, dyna_obstacle_list=None, dynamics=None):
        state_cur = copy.deepcopy(state_cur)
        state_cur = self.global_to_local_coordinate_transform(state_cur, dynamics)
        cost = (1/2) * (state_cur - self.goal_checker.goal_state).T @ self.Q @ (state_cur - self.goal_checker.goal_state)
        if actions is not None:
            cost += (1/2) * actions.T @ self.R @ actions
        if self.collision_checker.check(state_cur):  # True for collision, False for no collision
            if self.collision_cost is not None:
                cost += self.collision_cost
            else:
                cost += 1000  # default collision cost
        return cost

    def evaluate_terminal_cost(self, state_cur, actions=None, dyna_obstacle_list=None, dynamics=None):
        state_cur = self.global_to_local_coordinate_transform(state_cur, dynamics)
        cost = (1/2) * (state_cur - self.goal_checker.goal_state).T @ self.QN @ (state_cur - self.goal_checker.goal_state)
        if self.collision_checker.check(state_cur):  # True for collision, False for no collision
            if self.collision_cost is not None:
                cost += self.collision_cost
            else:
                cost += 1000  # default collision cost
        return cost

    def global_to_local_coordinate_transform(self, state, dynamics):
        e_psi, e_y, s = dynamics.track.localize(np.array((state[-2], state[-1])), state[-3])
        state[5:] = np.vstack((e_psi, e_y, s)).reshape((3,))
        return state

