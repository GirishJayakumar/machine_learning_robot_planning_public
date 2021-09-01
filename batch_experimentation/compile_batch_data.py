import numpy as np
import os
import matplotlib.pyplot as plt
from robot_planning.utils import AUTORALLY_DYNAMICS_DIR


class BatchDataParser:
    def __init__(self):
        self.states = []
        self.crashes = []

    def parse_datas(self):
        batch_code = 'batch_00'
        directory = 'experiments/Autorally_experiments/' + batch_code + '/'
        for run_file in os.listdir(directory):
            run_data = np.load(directory + run_file)
            self.states.append(run_data['states'])
            self.crashes.append(run_data['crash'])

    def print_crashes(self):
        num_crashes = sum(self.crashes)
        num_runs = len(self.crashes)
        print(num_crashes, ' crashes out of ', num_runs, ' runs = ', num_crashes/num_runs)

    def plot_trajectories(self):
        plt.figure()
        map_file = 'environment/dynamics/autorally_dynamics/CCRF_2021-01-10.npz'
        map = np.load(map_file)
        plt.plot(map['X_in'], map['Y_in'], 'k')
        plt.plot(map['X_out'], map['Y_out'], 'k')
        for states in self.states:
            plt.scatter(states[6, :], states[7, :], c=states[0, :], marker='.')
        plt.show()


if __name__ == '__main__':
    data_parser = BatchDataParser()
    data_parser.parse_datas()
    data_parser.print_crashes()
    data_parser.plot_trajectories()
