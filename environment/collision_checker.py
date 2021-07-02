import numpy as np
import ast


class CollisionChecker(object):
    def __init__(self, obstacles=None, kinematics=None):
        self.kinematics = kinematics  # for more complicated collision checkers
        self.obstacles = obstacles

    def initialize_from_config(self, config_data, section_name):
        raise NotImplementedError

    def set_obstacles(self, obstacles):
        self.obstacles = obstacles

    def set_other_agents_list(self, other_agents_list):
        self.other_agents_list = other_agents_list

    def check(self, state_cur):
        raise NotImplementedError


class BicycleModelCollisionChecker(CollisionChecker): # TODO: This is currently a circular collision checker, should implement bicycle kinematics
    def __init__(self, obstacles=None, kinematics=None):
        CollisionChecker.__init__(self, obstacles, kinematics)

    def initialize_from_config(self, config_data, section_name):
        if config_data.has_option(section_name, 'obstacles'):
            self.obstacles = np.asarray(ast.literal_eval(config_data.get(section_name, 'obstacles')))
        if config_data.has_option(section_name, 'obstacles_radius'):
            self.obstacles_radius = np.asarray(ast.literal_eval(config_data.get(section_name, 'obstacles_radius')))
        self.other_agents_list = None

    def check(self, state_cur):  # True for collision, False for no collision
        for i in range(len(self.obstacles)):
            if np.linalg.norm(self.obstacles[i] - state_cur[:2]) < self.obstacles_radius[i]:
                return True
        if self.other_agents_list is not None:
            for agent in self.other_agents_list:
                if np.linalg.norm(agent.state - state_cur[:2]) < 1: #TODO: this is hack. Implement kinematics...
                    return True
        return False

