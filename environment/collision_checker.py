import numpy as np
import ast
from robot_planning.factory.factory_from_config import factory_from_config
from robot_planning.factory.factories import kinematics_factory_base
from robot_planning.environment.kinematics.simulated_kinematics import PointKinematics
from robot_planning.environment.kinematics.simulated_kinematics import BicycleModelKinematics


class CollisionChecker(object):
    def __init__(self, obstacles=None, kinematics=None, field_boundary=None):
        self.kinematics = kinematics  # for more complicated collision checkers
        self.obstacles = obstacles
        self.field_boundary = field_boundary

    def initialize_from_config(self, config_data, section_name):
        pass

    def set_obstacles(self, obstacles):
        self.obstacles = obstacles

    def set_other_agents_list(self, other_agents_list):
        self.other_agents_list = other_agents_list

    def check(self, state_cur, check_other_agents=False):
        raise NotImplementedError


class PointCollisionChecker(CollisionChecker):
    def __init__(self, obstacles=None, kinematics=None):
        CollisionChecker.__init__(self, obstacles, kinematics)

    def initialize_from_config(self, config_data, section_name):
        kinematics_section_name = config_data.get(section_name, 'kinematics')
        self.kinematics = factory_from_config(kinematics_factory_base, config_data, kinematics_section_name)
        assert isinstance(self.kinematics, PointKinematics), 'The PointCollisionChecker should have PointKinematics'
        if config_data.has_option(section_name, 'obstacles'):
            self.obstacles = np.asarray(ast.literal_eval(config_data.get(section_name, 'obstacles')))
        if config_data.has_option(section_name, 'field_boundary'):
            self.field_boundary = np.asarray(ast.literal_eval(config_data.get(section_name, 'field_boundary')))
        if config_data.has_option(section_name, 'obstacles_radius'):
            self.obstacles_radius = np.asarray(ast.literal_eval(config_data.get(section_name, 'obstacles_radius')))
        if len(self.obstacles) is not len(self.obstacles_radius):
            raise ValueError('the numbers of obstacles and radii do not match')
        self.other_agents_list = None


    def get_obstacle_list(self):
        obstacle_list = []
        for i in range(len(self.obstacles)):
            obstacle_list.append((self.obstacles[i][0], self.obstacles[i][1], self.obstacles_radius[i]))
        return obstacle_list

    def check(self, state_cur, check_other_agents=False):  # True for collision, False for no collision
        state_cur = np.squeeze(state_cur)
        for i in range(len(self.obstacles)):
            if self.obstacles_radius[i] == 0:
                continue
            if np.linalg.norm(self.obstacles[i] - state_cur[:2]) < self.obstacles_radius[i] + self.kinematics.get_radius():
                return True
        if self.other_agents_list is not None and check_other_agents:
            for agent in self.other_agents_list:
                if np.linalg.norm(agent.state[:2] - state_cur[:2]) < \
                        agent.cost_evaluator.collision_checker.kinematics.get_radius() + self.kinematics.get_radius():
                    return True

        if self.field_boundary is not None:
            pos_cur = [state_cur[0], state_cur[1]]
            if abs(pos_cur[0]) > self.field_boundary[0] - 0.5 or abs(pos_cur[1]) > self.field_boundary[1] -0.5:
                return True
        return False


class BicycleModelCollisionChecker(CollisionChecker):
    def __init__(self, obstacles=None, kinematics=None):
        CollisionChecker.__init__(self, obstacles, kinematics)

    def initialize_from_config(self, config_data, section_name):
        kinematics_section_name = config_data.get(section_name, 'kinematics')
        self.kinematics = factory_from_config(kinematics_factory_base, config_data, kinematics_section_name)
        assert isinstance(self.kinematics, BicycleModelKinematics), 'The BicycleModelCollisionChecker should have BicycleModelKinematics'
        if config_data.has_option(section_name, 'obstacles'):
            self.obstacles = np.asarray(ast.literal_eval(config_data.get(section_name, 'obstacles')))
        if config_data.has_option(section_name, 'obstacles_radius'):
            self.obstacles_radius = np.asarray(ast.literal_eval(config_data.get(section_name, 'obstacles_radius')))
        if config_data.has_option(section_name, 'agent_safety_distance'):
            self.agent_safety_distance = config_data.getfloat(section_name, 'agent_safety_distance')
        self.other_agents_list = None

    def check(self, state_cur, check_other_agents=False):  # True for collision, False for no collision
        state_cur = np.squeeze(state_cur)
        vertex_list = self.kinematics.compute_rectangle_vertices_from_state(state_cur)
        for vertex in vertex_list:
            for i in range(len(self.obstacles)):
                if np.linalg.norm(self.obstacles[i] - vertex) < self.obstacles_radius[i]:
                    return True
        if self.other_agents_list is not None and check_other_agents:
            for agent in self.other_agents_list:
                if np.linalg.norm(agent.state[:2] - state_cur[:2]) < self.agent_safety_distance: #TODO: this is hack. Need to use a generic representation for agent kinematics
                    return True
        return False


class AutorallyCollisionChecker(PointCollisionChecker):
    def __init__(self, obstacles=None, kinematics=None):
        CollisionChecker.__init__(self, obstacles, kinematics)

    def initialize_from_config(self, config_data, section_name):
        PointCollisionChecker.initialize_from_config(self, config_data, section_name)
        self.track_width = config_data.getfloat(section_name, "track_width")
        kinematics_section_name = config_data.get(section_name, 'kinematics')
        self.kinematics = factory_from_config(kinematics_factory_base, config_data, kinematics_section_name)

    def check(self, state_cur, check_other_agents=False):
        if state_cur.ndim == 1:
            if state_cur[-2] < -self.track_width or self.track_width < state_cur[-2]:
                return True
            else:
                return False
        else:
            collisions = np.where((state_cur[-2, :] < -self.track_width) | (self.track_width < state_cur[-2, :]), 1, 0)
            return collisions
