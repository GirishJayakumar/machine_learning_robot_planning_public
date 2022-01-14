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

    def evaluate(self, state_cur, state_next=None, dyna_obstacle_list=None):
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
            self.collision_cost = config_data.getfloat(section_name,
                                                       'collision_cost')  # collision_cost should be positive
        if config_data.has_option(section_name, 'goal_cost'):
            self.goal_cost = config_data.getfloat(section_name, 'goal_cost')  # goal_cost should be negative
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
        if state_cur.ndim == 1:
            state_cur = state_cur.reshape((-1, 1))
        cost = (1 / 2) * (state_cur - self.goal_checker.goal_state).T @ self.Q @ (
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

    def evaluate_terminal_cost(self, state_cur, actions=None, dyna_obstacle_list=None, dynamics=None):
        if state_cur.ndim == 1:
            state_cur = state_cur.reshape((-1, 1))
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

    def evaluate(self, state_cur, actions=None, noises=None, dyna_obstacle_list=None, dynamics=None):
        map_state = self.global_to_local_coordinate_transform(state_cur, dynamics)
        error_state_right = np.expand_dims((map_state - self.goal_checker.goal_state.reshape((-1, 1))).T, axis=2)
        error_state_left = np.expand_dims((map_state - self.goal_checker.goal_state.reshape((-1, 1))).T, axis=1)
        cost = (1 / 2) * error_state_left @ np.tile(np.expand_dims(self.Q, axis=0),
                                                    (state_cur.shape[1], 1, 1)) @ error_state_right
        if actions is not None:
            actions_left = np.expand_dims(actions.T, axis=1)
            actions_right = np.expand_dims(actions.T, axis=2)
            if noises is not None:
                noises_left = np.expand_dims(noises.T, axis=1)
                noises_right = np.expand_dims(noises.T, axis=2)
                cost += 1 / 2 * noises_left @ np.tile(np.expand_dims(self.R, axis=0),
                                                      (state_cur.shape[1], 1, 1)) @ noises_right
                cost += (actions_left - noises_left) @ np.tile(np.expand_dims(self.R, axis=0),
                                                               (state_cur.shape[1], 1, 1)) @ noises_right
            else:
                cost += (1 / 2) * actions_left @ np.tile(np.expand_dims(self.R, axis=0),
                                                         (state_cur.shape[1], 1, 1)) @ actions_right
        # collisions =  self.collision_checker.check(state_cur)  # True for collision, False for no collision
        # collisions = collisions.reshape((-1, 1, 1))
        # if self.collision_cost is not None:
        #     cost += collisions * self.collision_cost
        # else:
        #     cost += collisions * 1000  # default collision cost
        return cost

    def evaluate_terminal_cost(self, state_cur, actions=None, dyna_obstacle_list=None, dynamics=None):
        map_state = self.global_to_local_coordinate_transform(state_cur, dynamics)
        error_state_right = np.expand_dims((map_state - self.goal_checker.goal_state.reshape((-1, 1))).T, axis=2)
        error_state_left = np.expand_dims((map_state - self.goal_checker.goal_state.reshape((-1, 1))).T, axis=1)
        cost = (1 / 2) * error_state_left @ np.tile(np.expand_dims(self.QN, axis=0),
                                                    (state_cur.shape[1], 1, 1)) @ error_state_right
        # collisions = self.collision_checker.check(state_cur)  # True for collision, False for no collision
        # collisions = collisions.reshape((-1, 1, 1))
        # if self.collision_cost is not None:
        #     cost += collisions * self.collision_cost
        # else:
        #     cost += collisions * 1000  # default collision cost
        return cost

    def global_to_local_coordinate_transform(self, state, dynamics):
        e_psi, e_y, s = dynamics.track.localize(np.array((state[-2, :], state[-1, :])), state[-3, :])
        new_state = state.copy()
        new_state[5:, :] = np.vstack((e_psi, e_y, s))
        return new_state


class TerminalCostEvaluator(CostEvaluator):
    def __init__(self, goal_checker=None, collision_checker=None, Q=None, R=None, collision_cost=None, goal_cost=None):
        CostEvaluator.__init__(self, goal_checker, collision_checker)
        self.collision_cost = collision_cost
        self.goal_cost = goal_cost
        self.dense = None

    def initialize_from_config(self, config_data, section_name):
        if config_data.has_option(section_name, 'collision_cost'):
            self.collision_cost = config_data.getfloat(section_name,
                                                       'collision_cost')  # collision_cost should be positive
        if config_data.has_option(section_name, 'goal_cost'):
            self.goal_cost = config_data.getfloat(section_name, 'goal_cost')  # goal_cost should be negative
        if config_data.has_option(section_name, 'goal_checker'):
            goal_checker_section_name = config_data.get(section_name, 'goal_checker')
            self.goal_checker = factory_from_config(goal_checker_factory_base, config_data,
                                                    goal_checker_section_name)
        if config_data.has_option(section_name, 'collision_checker'):
            collision_checker_section_name = config_data.get(section_name, 'collision_checker')
            self.collision_checker = factory_from_config(collision_checker_factory_base, config_data,
                                                         collision_checker_section_name)
        if config_data.has_option(section_name, 'dense'):
            self.dense = config_data.getboolean(section_name, 'dense')

    def evaluate(self, state_cur, actions=None, dyna_obstacle_list=None):
        cost = 0
        if self.collision_checker.check(state_cur):  # True for collision, False for no collision
            if self.collision_cost is not None:
                cost += self.collision_cost
            else:
                cost += 1000  # default collision cost
        if self.dense:
            cost += 10 * self.goal_checker.dist(state_cur)

        if self.goal_checker.check(state_cur):  # True for goal reached, False for goal not reached
            if self.goal_cost is not None:
                cost += self.goal_cost
            else:
                cost += -5000  # default goal cost

        return cost


class AbstractCostEvaluator(CostEvaluator):
    def __init__(self, goal_checker=None, sub_goal_checker=None, collision_checker=None, non_achievable_cost=None, achievable_cost=None):
        CostEvaluator.__init__(self, goal_checker, collision_checker)
        self.non_achievable_cost = non_achievable_cost
        self.achievable_cost = achievable_cost
        self.dense = None
        self.sub_goal_checker = sub_goal_checker

    def initialize_from_config(self, config_data, section_name):
        goal_checker_section_name = config_data.get(section_name, 'goal_checker')
        self.goal_checker = factory_from_config(goal_checker_factory_base, config_data, goal_checker_section_name)
        self.non_achievable_cost = config_data.getfloat(section_name, 'non_achievable_cost')
        self.ultimate_goal_cost = config_data.getfloat(section_name, 'ultimate_goal_cost')
        if config_data.has_option(section_name, 'achievable_cost'):
            self.achievable_cost = config_data.getfloat(section_name, 'achievable_cost')
        else:
            self.achievable_cost = - self.non_achievable_cost

    def set_sub_goal_checker(self, sub_goal_checker):
        self.sub_goal_checker = sub_goal_checker

    def evaluate(self, state_cur, state_next=None, action=None):
        assert np.linalg.norm(action.reshape(self.sub_goal_checker.goal_state.shape) - self.sub_goal_checker.goal_state) < 1e-5
        cost = 0
        if self.sub_goal_checker.check(state_cur):
            cost += self.achievable_cost
        else:
            cost += self.non_achievable_cost
        if self.goal_checker.check(state_cur):
            cost += self.ultimate_goal_cost
        return cost
