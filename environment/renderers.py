import numpy as np
import ast
from robot_planning.factory.factory_from_config import factory_from_config
from robot_planning.factory.factories import kinematics_factory_base
import matplotlib.pyplot as plt


class Renderer():
    def __init__(self):
        pass

    def initialize_from_config(self, config_data, section_name):
        pass

    def create_figure(self):
        pass

    def render_all(self):
        pass

    def render_agents(self):
        pass

    def render_obstacles(self):
        pass

    def render_trajectories(self):
        pass


class MatplotlibRenderer(Renderer):
    def __init__(self, xaxis_range=None, yaxis_range=None, auto_range=None, figure_size=None, figure_dpi=None):
        Renderer.__init__(self)
        self.xaxis_range = xaxis_range
        self.yaxis_range = yaxis_range
        self.auto_range = auto_range
        self.figure_size = figure_size
        self.figure_dpi = figure_dpi
        self._figure = None
        self._axis = None

    def initialize_from_config(self, config_data, section_name):
        Renderer.initialize_from_config(self, config_data, section_name)
        self.xaxis_range = np.asarray(ast.literal_eval(config_data.get(section_name, 'xaxis_range')), dtype=np.float64)
        self.yaxis_range = np.asarray(ast.literal_eval(config_data.get(section_name, 'yaxis_range')), dtype=np.float64)
        self.figure_size = np.asarray(ast.literal_eval(config_data.get(section_name, 'figure_size')), dtype=np.int)
        self.figure_dpi = config_data.getint(section_name, 'figure_dpi')
        if config_data.has_option(section_name, 'auto_range'):
            self.auto_range = ast.literal_eval(config_data.get(section_name, 'auto_range'))
        else:
            self.auto_range = False
        self.create_figure()

    def create_figure(self):
        self._figure = plt.figure(figsize=(self.figure_size[0], self.figure_size[1]), dpi=self.figure_dpi)
        self._axis = self._figure.add_subplot(1, 1, 1)
        plt.figure(self._figure.number)

    def show(self):
        if not self.auto_range:
            self._axis.axis([self.xaxis_range[0], self.xaxis_range[1], self.yaxis_range[0],  self.yaxis_range[1]])
        plt.grid(True)
        plt.pause(0.01)

    def clear(self):
        plt.cla()


class MPPIMatplotlibRenderer(MatplotlibRenderer):
    def __init__(self, xaxis_range=None, yaxis_range=None, auto_range=None, figure_size=None, figure_dpi=None):
        MatplotlibRenderer.__init__(self, xaxis_range, yaxis_range, auto_range, figure_size, figure_dpi)

    def initialize_from_config(self, config_data, section_name):
        MatplotlibRenderer.initialize_from_config(self, config_data, section_name)

    def render_states(self, state_list=None, kinematics=None, **kwargs):
        for i in range(len(state_list)):
            state = state_list[i]
            circle = plt.Circle((state[0], state[1]), kinematics.radius, **kwargs)
            self._axis.add_artist(circle)

    def render_obstacles(self, obstacle_list=None, **kwargs):
        for (ox, oy, size) in obstacle_list:
            circle = plt.Circle((ox, oy), size, **kwargs)
            self._axis.add_artist(circle)

    def render_goal(self, goal=None, **kwargs):
        x = goal[0]
        y = goal[1]
        radius = goal[2]
        circle = plt.Circle((x, y), radius, **kwargs)
        self._axis.add_artist(circle)

    def render_trajectories(self, trajectory_list=None, **kwargs):
        for trajectory in trajectory_list:
            previous_state = trajectory[:, 0]
            for i in range(1, trajectory.shape[1]):
                state = trajectory[:, i]
                line, = self._axis.plot([state[0], previous_state[0]], [state[1], previous_state[1]], **kwargs)
                previous_state = state


class AutorallyMatplotlibRenderer(MatplotlibRenderer):
    def __init__(self, xaxis_range=None, yaxis_range=None, auto_range=None, figure_size=None, figure_dpi=None):
        MatplotlibRenderer.__init__(self, xaxis_range, yaxis_range, auto_range, figure_size, figure_dpi)

    def initialize_from_config(self, config_data, section_name):
        MatplotlibRenderer.initialize_from_config(self, config_data, section_name)
        map_path = config_data.get(section_name, 'map_path')
        self.map = np.load(map_path)

    def render_states(self, state_list=None, kinematics=None, **kwargs):
        for i in range(len(state_list)):
            state = state_list[i]
            circle = plt.Circle((state[-2], state[-1]), kinematics.radius, **kwargs)
            self._axis.add_artist(circle)

    def render_obstacles(self, obstacle_list=None, **kwargs):
        self._axis.plot(self.map['X_in'], self.map['Y_in'], 'k')
        self._axis.plot(self.map['X_out'], self.map['Y_out'], 'k')
        return

    def render_goal(self, goal=None, **kwargs):
        # x = goal[0]
        # y = goal[1]
        # radius = goal[2]
        # circle = plt.Circle((x, y), radius, **kwargs)
        # self._axis.add_artist(circle)
        return

    def render_trajectories(self, trajectory_list=None, **kwargs):
        for trajectory in trajectory_list:
            previous_state = trajectory[:, 0]
            for i in range(1, trajectory.shape[1]):
                state = trajectory[:, i]
                line, = self._axis.plot([state[-1], previous_state[-1]], [state[-2], previous_state[-2]], **kwargs)
                previous_state = state
