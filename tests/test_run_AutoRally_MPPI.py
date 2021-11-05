try:
    import ConfigParser
except ImportError:
    import configparser as ConfigParser
import os
from robot_planning.environment.robots.simulated_robot import SimulatedRobot
from robot_planning.factory.factories import robot_factory_base
from robot_planning.factory.factories import renderer_factory_base
from robot_planning.factory.factories import logger_factory_base
from robot_planning.factory.factory_from_config import factory_from_config
from robot_planning.factory.factories import goal_checker_factory_base
import unittest
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


class TestRunAutorallyMPPI(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        print("setUpClass")

    @classmethod
    def tearDownClass(cls):
        print("tearDownClass")

    def test_run_autorally_MPPI(self):
        config_path = "configs/test_run_Autorally_MPPI.cfg"
        config_data = ConfigParser.ConfigParser()
        config_data.read(config_path)
        agent1 = factory_from_config(robot_factory_base, config_data, 'agent1')
        renderer1 = factory_from_config(renderer_factory_base, config_data, 'renderer1')
        logger = factory_from_config(logger_factory_base, config_data, 'logger')
        logger.set_agent(agent=agent1)
        agent1.set_renderer(renderer=renderer1)
        goal_checker_for_checking_vehicle_position = factory_from_config(goal_checker_factory_base, config_data, 'my_goal_checker_for_checking_vehicle_position')
        while not logger.goal_checker.check(agent1.state.reshape((-1, 1))):
            state_next, cost = agent1.take_action_with_controller()
            logger.calculate_number_of_laps(state_next, dynamics=agent1.dynamics, goal_checker=agent1.cost_evaluator.goal_checker)
            logger.calculate_number_of_collisions(state_next, dynamics=agent1.dynamics, collision_checker=agent1.cost_evaluator.collision_checker)
            renderer1.render_goal(goal_checker_for_checking_vehicle_position.get_goal())
            logger.log()
            if logger.crash == 1:
                raise RuntimeError
            renderer1.show()
            renderer1.clear()
            print(state_next, "    ", cost)


if __name__ == '__main__':
    unittest.main()

