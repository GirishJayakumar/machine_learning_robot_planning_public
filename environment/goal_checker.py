import numpy as np
import ast


class GoalChecker(object):
    def __init__(self, goal_state=None, kinematics=None):
        self.kinematics = kinematics # for more complicated collision checkers
        self.goal_state = goal_state

    def initialize_from_config(self, config_data, section_name):
        raise NotImplementedError

    def set_goal(self, goal_state):
        self.goal_state = goal_state

    def check(self, state_cur):
        raise NotImplementedError


class BicycleModelGoalChecker(GoalChecker):
    def __init__(self, goal=None, kinematics=None):
        GoalChecker.__init__(self, goal, kinematics)

    def initialize_from_config(self, config_data, section_name):
        self.goal_state = np.asarray(ast.literal_eval(config_data.get(section_name, 'goal_state')))
        self.goal_radius = config_data.getfloat(section_name, 'goal_radius')

    def check(self, state_cur):  # True for goal reached, False for goal not reached
        if np.linalg.norm(self.goal_state - state_cur) < self.goal_radius:
            return True
        return False
