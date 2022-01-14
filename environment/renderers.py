import numpy as np
import ast
from robot_planning.factory.factory_from_config import factory_from_config
from robot_planning.factory.factories import kinematics_factory_base
import matplotlib.pyplot as plt
from robot_planning.utils import AUTORALLY_DYNAMICS_DIR
import os
import imageio
import moviepy.video.io.ImageSequenceClip


class Renderer():
    def __init__(self):
        pass

    def initialize_from_config(self, config_data, section_name):
        pass

    def create_figure(self):
        pass

    def close_figure(self):
        pass

    def render_all(self):
        pass

    def render_agents(self):
        pass

    def render_states(self):
        pass

    def render_obstacles(self):
        pass

    def render_trajectories(self):
        pass

    def show(self):
        pass

    def clear(self):
        pass


class MatplotlibRenderer(Renderer):
    def __init__(self, xaxis_range=None, yaxis_range=None, auto_range=None, figure_size=None, figure_dpi=None,
                 active=True, save_animation=False):
        Renderer.__init__(self)
        self.xaxis_range = xaxis_range
        self.yaxis_range = yaxis_range
        self.auto_range = auto_range
        self.figure_size = figure_size
        self.figure_dpi = figure_dpi
        self._figure = None
        self._axis = None
        self.active = active
        self.save_animation = save_animation
        self.save_dir = None
        self.frame = 0

    def initialize_from_config(self, config_data, section_name):
        Renderer.initialize_from_config(self, config_data, section_name)
        self.xaxis_range = np.asarray(ast.literal_eval(config_data.get(section_name, 'xaxis_range')), dtype=np.float64)
        self.yaxis_range = np.asarray(ast.literal_eval(config_data.get(section_name, 'yaxis_range')), dtype=np.float64)
        self.figure_size = np.asarray(ast.literal_eval(config_data.get(section_name, 'figure_size')), dtype=int)
        self.figure_dpi = config_data.getint(section_name, 'figure_dpi')
        if config_data.has_option(section_name, 'auto_range'):
            self.auto_range = ast.literal_eval(config_data.get(section_name, 'auto_range'))
        else:
            self.auto_range = False
        self.create_figure()

    def set_save_dir(self, save_dir):
        assert save_dir is not None
        if not os.path.isdir(save_dir):
            os.mkdir(save_dir)
        self.save_dir = save_dir

    def create_figure(self):
        if self.active:
            self._figure = plt.figure(figsize=(self.figure_size[0], self.figure_size[1]), dpi=self.figure_dpi)
            self._axis = self._figure.add_subplot(1, 1, 1)
            plt.figure(self._figure.number)

    def close_figure(self):
        plt.close(self._figure)

    def set_range(self):
        if not self.auto_range:
            self._axis.axis([self.xaxis_range[0], self.xaxis_range[1], self.yaxis_range[0], self.yaxis_range[1]])
        plt.grid(True)

    def show(self):
        assert self.active
        self.set_range()
        plt.pause(0.01)
        if self.save_animation:
            self.save()
        self.frame += 1

    def clear(self):
        plt.cla()

    def save(self, save_path_name=None):
        assert self.active
        if save_path_name is None:
            assert self.save_dir is not None
            save_path_name = self.save_dir / 'frame{}.png'.format(self.frame)
        self.set_range()
        plt.savefig(save_path_name)

    def activate(self):
        self.active = True
        self.frame = 0

    def deactivate(self):
        self.active = False
        self.frame = 0

    def render_gif(self, duration=0.5):
        frames = [0 for _ in range(5)]
        for frame in range(self.frame):
            frames.append(frame)
        frames += [self.frame - 1 for _ in range(5)]

        images = []
        for frame in frames:
            file_name = self.save_dir / 'frame{}.png'.format(frame)
            images.append(imageio.imread(file_name))

        gif_dir = self.save_dir / 'movie.gif'
        imageio.mimsave(gif_dir, images, duration=duration)

    def render_mp4(self, duration=0.5):
        image_folder = self.save_dir
        fps = int(1 / duration)

        frames = [0 for _ in range(5)]
        for frame in range(self.frame):
            frames.append(frame)
        frames += [self.frame - 1 for _ in range(5)]

        image_files = [os.path.join(image_folder, 'frame{}.png'.format(frame))
                       for frame in frames]
        clip = moviepy.video.io.ImageSequenceClip.ImageSequenceClip(image_files, fps=fps)
        clip.write_videofile(str(self.save_dir / 'movie.mp4'))



class MPPIMatplotlibRenderer(MatplotlibRenderer):
    def __init__(self, xaxis_range=None, yaxis_range=None, auto_range=None, figure_size=None, figure_dpi=None):
        MatplotlibRenderer.__init__(self, xaxis_range, yaxis_range, auto_range, figure_size, figure_dpi)

    def initialize_from_config(self, config_data, section_name):
        MatplotlibRenderer.initialize_from_config(self, config_data, section_name)

    def render_states(self, state_list=None, kinematics=None, **kwargs):
        if self.active:
            for i in range(len(state_list)):
                state = state_list[i]
                circle = plt.Circle((state[0], state[1]), kinematics.radius, **kwargs)
                self._axis.add_artist(circle)

    def render_obstacles(self, obstacle_list=None, **kwargs):
        if self.active:
            for (x, y, size) in obstacle_list:
                circle = plt.Circle((x, y), size, **kwargs)
                self._axis.add_artist(circle)

    def render_goal(self, goal=None, **kwargs):
        if self.active:
            x = goal[0]
            y = goal[1]
            radius = goal[2]
            circle = plt.Circle((x, y), radius, **kwargs)
            self._axis.add_artist(circle)

    def render_trajectories(self, trajectory_list=None, **kwargs):
        if self.active:
            for trajectory in trajectory_list:
                previous_state = trajectory[:, 0]
                for i in range(1, trajectory.shape[1]):
                    state = trajectory[:, i]
                    line, = self._axis.plot([state[0], previous_state[0]], [state[1], previous_state[1]], **kwargs)
                    previous_state = state


class EnvMatplotlibRenderer(MatplotlibRenderer):
    def __init__(self, xaxis_range=None, yaxis_range=None, auto_range=None, figure_size=None, figure_dpi=None):
        MatplotlibRenderer.__init__(self, xaxis_range, yaxis_range, auto_range, figure_size, figure_dpi)

    def initialize_from_config(self, config_data, section_name):
        MatplotlibRenderer.initialize_from_config(self, config_data, section_name)

    def render_states(self, state_list=None, kinematics_list=None, **kwargs):
        if self.active:
            for i in range(len(state_list)):
                state = state_list[i]
                kinematics = kinematics_list[i]
                circle = plt.Circle((state[0], state[1]), kinematics.radius, **kwargs)
                self._axis.add_artist(circle)

    def render_obstacles(self, obstacle_list=None, **kwargs):
        if self.active:
            for (ox, oy, size) in obstacle_list:
                circle = plt.Circle((ox, oy), size, **kwargs)
                self._axis.add_artist(circle)

    def render_goal(self, goal=None, **kwargs):
        if self.active:
            x = goal[0]
            y = goal[1]
            radius = goal[2]
            circle = plt.Circle((x, y), radius, **kwargs)
            self._axis.add_artist(circle)

    def render_trajectories(self, trajectory_list=None, **kwargs):
        if self.active:
            for trajectory in trajectory_list:
                previous_state = trajectory[:, 0]
                for i in range(1, trajectory.shape[1]):
                    state = trajectory[:, i]
                    line, = self._axis.plot([state[0], previous_state[0]], [state[1], previous_state[1]], **kwargs)
                    previous_state = state


class CSSMPCMatplotlibRenderer(MatplotlibRenderer):
    def __init__(self, xaxis_range=None, yaxis_range=None, auto_range=None, figure_size=None, figure_dpi=None):
        MatplotlibRenderer.__init__(self, xaxis_range, yaxis_range, auto_range, figure_size, figure_dpi)

    def initialize_from_config(self, config_data, section_name):
        MatplotlibRenderer.initialize_from_config(self, config_data, section_name)

    def render_states(self, state_list=None, kinematics=None, **kwargs):
        for i in range(len(state_list)):
            state = state_list[i]
            circle = plt.Circle((state[-1], state[-2]), kinematics.radius, **kwargs)
            self._axis.add_artist(circle)

    def render_obstacles(self, obstacle_list=None, **kwargs):
        # for (ox, oy, size) in obstacle_list:
        #     circle = plt.Circle((ox, oy), size, **kwargs)
        #     self._axis.add_artist(circle)
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


class AutorallyMatplotlibRenderer(MatplotlibRenderer):
    def __init__(self, xaxis_range=None, yaxis_range=None, auto_range=None, figure_size=None, figure_dpi=None,
                 trajectories_rendering=True):
        MatplotlibRenderer.__init__(self, xaxis_range, yaxis_range, auto_range, figure_size, figure_dpi)
        self.trajectories_rendering = trajectories_rendering
        self.path_rendering = False
        self.path = np.zeros((3, 0))
        self.cbar = None

    def initialize_from_config(self, config_data, section_name):
        if config_data.has_option(section_name, 'trajectories_rendering'):
            self.trajectories_rendering = config_data.getboolean(section_name, 'trajectories_rendering')
        MatplotlibRenderer.initialize_from_config(self, config_data, section_name)
        if config_data.has_option(section_name, 'map_file'):
            map_file = config_data.get(section_name, 'map_file')
            self.map = np.load(AUTORALLY_DYNAMICS_DIR + '/' + map_file)
        if config_data.has_option(section_name, 'path_rendering'):
            self.path_rendering = config_data.get(section_name, 'path_rendering')

    def render_states(self, state_list=None, kinematics=None, **kwargs):
        for i in range(len(state_list)):
            state = state_list[i]
            circle = plt.Circle((state[-2], state[-1]), kinematics.radius, **kwargs)
            self._axis.add_artist(circle)
        if self.path_rendering:
            self.path = np.append(self.path, np.vstack((state[0], state[6], state[7])), axis=1)
            pcm = self._axis.scatter(self.path[1, :], self.path[2, :], c=self.path[0, :], marker='.')
            if self.path.shape[1] < 2:
                self.cbar = plt.colorbar(pcm)
                self.cbar.set_label('speed (m/s)')
            else:
                self.cbar.update_normal(pcm)

    def render_obstacles(self, obstacle_list=None, **kwargs):
        self._axis.plot(self.map['X_in'], self.map['Y_in'], 'k')
        self._axis.plot(self.map['X_out'], self.map['Y_out'], 'k')
        return

    def render_goal(self, goal=None, **kwargs):
        x = goal[0]
        y = goal[1]
        radius = goal[2]
        circle = plt.Circle((x, y), radius, **kwargs, zorder=0)
        self._axis.add_artist(circle)
        return

    def render_trajectories(self, trajectory_list=None, **kwargs):
        trajectory_list = np.asarray(trajectory_list)
        if self.trajectories_rendering is True:
            previous_state = trajectory_list[:, :, 0]
            for i in range(1, trajectory_list.shape[2]):
                state = trajectory_list[:, :, i]
                self._axis.plot([state[:, -2], previous_state[:, -2]], [state[:, -1], previous_state[:, -1]], **kwargs)
                previous_state = state
        else:
            pass
