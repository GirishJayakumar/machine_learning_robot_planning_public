try:
    import ConfigParser
except ImportError:
    import configparser as ConfigParser
from robot_planning.environment.environment import Environment
import numpy as np


def main():
    config_path = "configs/environment_configs/point_environment.cfg"
    config_data = ConfigParser.ConfigParser()
    config_data.read(config_path)
    environment = Environment()
    environment.initialize_from_config(config_data, 'environment')

    actions = [np.asarray([1, 0]), np.asarray([0.5, -0.5])]

    for i in range(50):
        states, observations, costs, rl_rewards = environment.step(actions)
        print("t={}".format(i))
        print("states:   ", states)
        print("observations:   ", observations)
        print("costs:   ", costs)
        environment.render()
        environment.renderer.show()
        environment.renderer.clear()

    environment.reset()
    print('******************')
    print('environment reset!')
    print('******************')

    for i in range(50):
        states, observations, costs, rl_rewards = environment.step(actions)
        print("t={}".format(i))
        print("states:   ", states)
        print("observations:   ", observations)
        print("costs:   ", costs)
        environment.render()
        environment.renderer.show()
        environment.renderer.clear()


if __name__ == '__main__':
    main()
    print('done!')
