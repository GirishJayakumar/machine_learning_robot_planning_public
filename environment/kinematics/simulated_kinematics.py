import numpy as np
import copy


class Kinematics(object):
    def __init__(self, kinematics_type=None):
        self.kinematics_type = kinematics_type

    def initialize_from_config(self, config_data, section_name):
        self.kinematics_type = config_data.get(section_name, 'type')


class PointKinematics(Kinematics):
    def __init__(self, radius=None):
        Kinematics.__init__(self)
        self.radius = radius

    def initialize_from_config(self, config_data, section_name):
        self.radius = config_data.getfloat(section_name, 'radius')

    def get_radius(self):
        return copy.copy(self.radius)


class BicycleModelKinematics(Kinematics):
    def __init__(self, length=None, width=None):
        Kinematics.__init__(self)
        self.length = length
        self.width = width

    def initialize_from_config(self, config_data, section_name):
        self.length = config_data.getfloat(section_name, 'length')
        self.width = config_data.getfloat(section_name, 'width')

    def compute_rectangle_vertices_from_state(self, state_cur):
        #  assuming the first two dimensions of the state are the x and y coordinates, and the third is yaw angle
        alpha = np.arctan(self.width/self.length)
        phi = state_cur[2]
        half_diagonal_line_length = np.linalg.norm(np.asarray([self.length/2, self.width]))
        front_vertex_left = np.asarray([state_cur[0] + half_diagonal_line_length * np.cos(phi+alpha), state_cur[1] + half_diagonal_line_length * np.sin(phi+alpha)]).reshape((2, 0))
        front_vertex_right = np.asarray([state_cur[0] + half_diagonal_line_length * np.cos(phi-alpha), state_cur[1] + half_diagonal_line_length * np.sin(phi-alpha)]).reshape((2, 0))
        rear_vertex_left = np.asarray([state_cur[0] - half_diagonal_line_length * np.cos(phi-alpha), state_cur[1] - half_diagonal_line_length * np.sin(phi-alpha)]).reshape((2, ))
        rear_vertex_right = np.asarray([state_cur[0] - half_diagonal_line_length * np.cos(phi+alpha), state_cur[1] - half_diagonal_line_length * np.sin(phi+alpha)]).reshape((2, ))
        
        return [front_vertex_left, front_vertex_right, rear_vertex_left, rear_vertex_right]


