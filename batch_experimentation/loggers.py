import copy
import csv
import numpy as np
import os
from robot_planning.utils import EXPERIMENT_ROOT_DIR
from robot_planning.factory.factories import collision_checker_factory_base, goal_checker_factory_base
from robot_planning.factory.factory_from_config import factory_from_config


class Logger(object):
    def __init__(self, experiment_root_dir=None, experiment_name=None):
        self.experiment_root_dir = None
        self.experiment_name = None
        self.experiments_folder_name = None

    def initialize_from_config(self, config_data, section_name):
        self.experiment_root_dir = EXPERIMENT_ROOT_DIR
        if config_data.has_option(section_name, 'experiment_name'):
            self.experiment_name = config_data.get(section_name, 'experiment_name')

    def save_fig(self):
        raise NotImplementedError

    def set_experiment_name(self, experiment_name):
        self.experiment_name = experiment_name

    def create_save_dir(self):
        experiments_dir = self.experiment_root_dir + "/" + self.experiments_folder_name
        if not os.path.isdir(experiments_dir):
            os.mkdir(experiments_dir)
        current_experiment_dir = self.experiment_root_dir + "/" + self.experiments_folder_name + "/" + self.experiment_name
        if not os.path.isdir(current_experiment_dir):
            os.mkdir(current_experiment_dir)
        return experiments_dir, current_experiment_dir


class MPPILogger(Logger):
    def __init__(self, experiment_dir=None):
        Logger.__init__(self, experiment_dir)
        self.experiments_folder_name = "MPPI_experiments"

    def initialize_from_config(self, config_data, section_name):
        Logger.initialize_from_config(self, config_data, section_name)

    def save_fig(self, renderer=None, time=None):
        self.create_save_dir()
        time = str(np.around(time, decimals=2))
        save_path_name = self.experiment_root_dir + "/" + self.experiments_folder_name + "/" + self.experiment_name + "/" + time + ".png"
        renderer.save(save_path_name)


class AutorallyLogger(Logger):
    def __init__(self, experiment_dir=None, collision_checker=None, goal_checker=None):
        Logger.__init__(self, experiment_dir)
        self.experiments_folder_name = "Autorally_experiments"
        self.number_of_collisions = 0
        self.number_of_laps = 0
        self.number_of_failure = 0
        self.in_obstacle = False
        self.around_goal_position = True
        self.collision_checker = collision_checker
        self.goal_checker = goal_checker

    def initialize_from_config(self, config_data, section_name):
        Logger.initialize_from_config(self, config_data, section_name)
        if config_data.has_option(section_name, 'goal_checker'):
            goal_checker_section_name = config_data.get(section_name, 'goal_checker')
            self.goal_checker = factory_from_config(goal_checker_factory_base, config_data,
                                                      goal_checker_section_name)
        if config_data.has_option(section_name, 'collision_checker'):
            collision_checker_section_name = config_data.get(section_name, 'collision_checker')
            self.collision_checker = factory_from_config(collision_checker_factory_base, config_data,
                                                          collision_checker_section_name)
        _, current_experiment_dir = self.create_save_dir()
        self.log_file_path = current_experiment_dir + "/" + self.experiment_name + "_log_file.csv"
        with open(self.log_file_path, 'w') as csvfile:
            csvwriter = csv.writer(csvfile)
            # TODO: extend the list if necessary
            csvwriter.writerow(['robot time', 'state', 'lap_num', 'collision_num', 'controller_failure_num'])

    def set_agent(self, agent):
        self.agent = agent
        self.in_obstacle = self.collision_checker.check(agent.get_state())
        self.around_goal_position = self.goal_checker.check(agent.get_state())

    def calculate_number_of_collisions(self, state, dynamics, collision_checker):
        #This state is in cartesian coordinates, needs to be converted to map coordinates
        state = self.global_to_local_coordinate_transform(state, dynamics)
        if collision_checker.check(state) and self.in_obstacle is False: # the limit should not be hard-coded
            self.in_obstacle = True
            self.number_of_collisions += 1
        if not collision_checker.check(state) and self.in_obstacle is True:
            self.in_obstacle = False

    def calculate_number_of_laps(self, state, dynamics, goal_checker):
        # This state is in cartesian coordinates, needs to be converted to map coordinates
        # state = self.global_to_local_coordinate_transform(state, dynamics)
        if goal_checker.check(state) and self.around_goal_position is False: # the limit should not be hard-coded
            self.around_goal_position = True
            self.number_of_laps += 1
        if not goal_checker.check(state) and self.around_goal_position is True:
            self.around_goal_position = False

    def add_number_of_failure(self):
        self.number_of_failure += 1

    def log(self):
        # TODO: extend info if necessary
        info = [self.agent.get_time(), self.agent.get_state(), self.number_of_laps, self.number_of_collisions, self.number_of_failure]
        with open(self.log_file_path, 'a') as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow(info)

    def global_to_local_coordinate_transform(self, state, dynamics):
        state = copy.deepcopy(state)
        e_psi, e_y, s = dynamics.track.localize(np.array((state[-2], state[-1])), state[-3])
        state[5:] = np.vstack((e_psi, e_y, s)).reshape((3,))
        return state

    def get_num_of_collisions(self):
        return copy.deepcopy(self.number_of_collisions)

    def get_num_of_laps(self):
        return copy.deepcopy(self.number_of_laps)

    def get_num_of_failures(self):
        return copy.deepcopy(self.number_of_failure)


class AutorallyMPPILogger(AutorallyLogger):
    def __init__(self, experiment_dir=None, collision_checker=None, goal_checker=None):
        AutorallyLogger.__init__(self, experiment_dir)
        self.experiments_folder_name = "Autorally_MPPI_experiments"

    def initialize_from_config(self, config_data, section_name):
        AutorallyLogger.initialize_from_config(self, config_data, section_name)


class AutorallyCSSMPCLogger(AutorallyLogger):
    def __init__(self, experiment_dir=None, collision_checker=None, goal_checker=None):
        AutorallyLogger.__init__(self, experiment_dir)
        self.experiments_folder_name = "Autorally_CSSMPC_experiments"

    def initialize_from_config(self, config_data, section_name):
        AutorallyLogger.initialize_from_config(self, config_data, section_name)