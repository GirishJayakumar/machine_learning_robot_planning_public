import numpy as np
import os
from robot_planning.utils import EXPERIMENT_ROOT_DIR


class Logger(object):
    def __init__(self, experiment_root_dir=None, experiment_name=None):
        self.experiment_root_dir = None
        self.experiment_name = None

    def initialize_from_config(self, config_data, section_name):
        pass

    def save_fig(self):
        raise NotImplementedError


class MPPILogger(Logger):
    def __init__(self, experiment_dir=None):
        Logger.__init__(self, experiment_dir)
        self.MPPI_experiments_folder_name = "MPPI_experiments"

    def initialize_from_config(self, config_data, section_name):
        self.experiment_root_dir = EXPERIMENT_ROOT_DIR
        if config_data.has_option(section_name, 'experiment_name'):
            self.experiment_name = config_data.get(section_name, 'experiment_name')

    def set_experiment_name(self, experiment_name):
        self.experiment_name = experiment_name

    def save_fig(self, renderer=None, time=None):
        self.create_save_dir()
        time = str(np.around(time, decimals=2))
        save_path_name = self.experiment_root_dir + "/" + self.MPPI_experiments_folder_name + "/" + self.experiment_name + "/" + time + ".png"
        renderer.save(save_path_name)

    def create_save_dir(self):
        MPPI_experiments_dir = self.experiment_root_dir + "/" + self.MPPI_experiments_folder_name
        if not os.path.isdir(MPPI_experiments_dir):
            os.mkdir(MPPI_experiments_dir)
        MPPI_current_experiment_dir = self.experiment_root_dir + "/" + self.MPPI_experiments_folder_name + "/" + self.experiment_name
        if not os.path.isdir(MPPI_current_experiment_dir):
            os.mkdir(MPPI_current_experiment_dir)




