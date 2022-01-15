try:
    import ConfigParser
except ImportError:
    import configparser as ConfigParser

from robot_planning.trainers.trainer import Trainer
from robot_planning.environment.environment import Environment
from robot_planning.factory.factories import robot_factory_base
from robot_planning.factory.factory_from_config import factory_from_config
import numpy as np
import unittest


class TestRunEnvironment(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        print("setUpClass")

    @classmethod
    def tearDownClass(cls):
        print("tearDownClass")

    def test_run_environment(self):
        config_path = 'configs/test_run_environment.cfg'
        config_data = ConfigParser.ConfigParser()
        config_data.read(config_path)
        environment = Environment()
        environment.initialize_from_config(config_data, 'environment')

        actions = [np.asarray([0, 1]), np.asarray([0, -1])]

        for i in range(4):
            states, observations, costs, rl_rewards = environment.step(actions)
            print("t={}".format(i))
            print("states:   ", states)
            print("observations:   ", observations)
            print("costs:   ", costs)

        environment.reset()
        print('******************')
        print('environment reset!')
        print('******************')

        for i in range(4):
            states, observations, costs, rl_rewards = environment.step(actions)
            print("t={}".format(i))
            print("states:   ", states)
            print("observations:   ", observations)
            print("costs:   ", costs)

    def test_run_single_agent(self):
        config_path = "configs/test_run_single_agent.cfg"
        config_data = ConfigParser.ConfigParser()
        config_data.read(config_path)
        agent1 = factory_from_config(robot_factory_base, config_data, 'agent1')

        action = np.asarray([0, 1]).reshape((2,))
        for i in range(20):
            state_next, cost = agent1.take_action(action)
            print(state_next, "    ", cost)

    def test_run_trainer(self):
        config_path = "configs/test_run_trainer.cfg"
        config_data = ConfigParser.ConfigParser()
        config_data.read(config_path)
        trainer = Trainer()
        trainer.initialize_from_config(config_data=config_data, section_name='trainer')
        print('trainer initialized!')
        trainer.train()
        print('done!')


if __name__ == '__main__':
    unittest.main()