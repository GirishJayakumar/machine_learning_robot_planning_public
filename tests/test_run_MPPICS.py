try:
    import ConfigParser
except ImportError:
    import configparser as ConfigParser
from robot_planning.environment.robots.simulated_robot import SimulatedRobot
from robot_planning.factory.factories import robot_factory_base
from robot_planning.factory.factories import renderer_factory_base
from robot_planning.factory.factory_from_config import factory_from_config
import unittest
import random
import numpy as np


class TestRunMPPI(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        print("setUpClass")

    @classmethod
    def tearDownClass(cls):
        print("tearDownClass")

    def test_run_MPPICS(self):
        config_path = "configs/test_run_MPPICS.cfg"
        config_data = ConfigParser.ConfigParser()
        config_data.read(config_path)
        agent1 = factory_from_config(robot_factory_base, config_data, 'agent1')
        renderer1 = factory_from_config(renderer_factory_base, config_data, 'renderer1')
        agent1.set_renderer(renderer=renderer1)
        actions = agent1.controller.plan(agent1.state).T
        state_next, cost = agent1.take_action_sequence(actions)
        renderer1.show()
        renderer1.close()



if __name__ == '__main__':
    unittest.main()
