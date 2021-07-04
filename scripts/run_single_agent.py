try:
    import ConfigParser
except ImportError:
    import configparser as ConfigParser
from robot_planning.environment.robots.simulated_robot import SimulatedRobot
from robot_planning.factory.factories import robot_factory_base
from robot_planning.factory.factory_from_config import factory_from_config
import numpy as np


def main():
    config_path = "configs/run_single_agent.cfg"
    config_data = ConfigParser.ConfigParser()
    config_data.read(config_path)
    agent1 = factory_from_config(robot_factory_base, config_data, 'agent1')

    action = np.asarray([0, 1]).reshape((2,))
    for i in range(100):
        state_next, cost = agent1.take_action(action)
        print(state_next, "    ", cost)
    

if __name__ == '__main__':
    main()
