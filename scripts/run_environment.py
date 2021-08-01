try:
    import ConfigParser
except ImportError:
    import configparser as ConfigParser
from robot_planning.environment.environment import Environment
import numpy as np


def main():
    config_path = "configs/run_environment.cfg"
    config_data = ConfigParser.ConfigParser()
    config_data.read(config_path)
    environment = Environment()
    environment.initialize_from_config(config_data, 'environment')

    actions = [np.asarray([0, 1]), np.asarray([0, -1])]

    for i in range(4):
        states, observations, costs = environment.step(actions)
        print("t={}".format(i))
        print("states:   ", states)
        print("observations:   ", observations)
        print("costs:   ", costs)

    environment.reset()
    print('******************')
    print('environment reset!')
    print('******************')

    for i in range(4):
        states, observations, costs = environment.step(actions)
        print("t={}".format(i))
        print("states:   ", states)
        print("observations:   ", observations)
        print("costs:   ", costs)


if __name__ == '__main__':
    main()
    print('done!')
