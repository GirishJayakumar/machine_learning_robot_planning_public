try:
    import ConfigParser
except ImportError:
    import configparser as ConfigParser
from robot_planning.environment.dynamics.bicycle_dynamics import BicycleDynamics
import numpy as np

def main():
    config_path = "configs/run_rrt.cfg"
    config_data = ConfigParser.ConfigParser()
    config_data.read(config_path)
    agent1 = BicycleDynamics()
    agent1.initialize_from_config(config_data, 'agent1')

    action = np.asarray([0, 1]).reshape((2,))
    for i in range(100):
        state_next, cost = agent1.propagate(action)
        print(state_next, "    ", cost)
    


if __name__ == '__main__':
    main()
