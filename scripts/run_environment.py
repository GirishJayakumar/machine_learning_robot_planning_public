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

    actions = np.asarray([[0, 1], [0, -1]]).T
    for i in range(100):
        states, costs = environment.step(actions)
        print(states, "    ", costs)

if __name__ == '__main__':
    main()
