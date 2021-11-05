try:
    import ConfigParser
except ImportError:
    import configparser as ConfigParser
from robot_planning.environment.robots.simulated_robot import SimulatedRobot
from robot_planning.factory.factories import robot_factory_base
from robot_planning.factory.factories import renderer_factory_base
from robot_planning.factory.factory_from_config import factory_from_config
import numpy as np
import unittest
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'


class TestRunCSSMPC(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        print("setUpClass")

    @classmethod
    def tearDownClass(cls):
        print("tearDownClass")

    def test_run_CSSMPC(self):
        config_path = "configs/test_run_CSSMPC.cfg"
        config_data = ConfigParser.ConfigParser()
        config_data.read(config_path)
        agent1 = factory_from_config(robot_factory_base, config_data, 'agent1')
        renderer1 = factory_from_config(renderer_factory_base, config_data, 'renderer1')
        agent1.set_renderer(renderer=renderer1)
        while not np.linalg.norm(agent1.cost_evaluator.goal_checker.goal_state[-2:] - agent1.state[-2:]) < 100:
            state_next, cost = agent1.take_action_with_controller()
            renderer1.show()
            renderer1.clear()
            print(state_next, "    ", cost)


if __name__ == '__main__':
    unittest.main()
