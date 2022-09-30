try:
    import ConfigParser
except ImportError:
    import configparser as ConfigParser
from robot_planning.environment.robots.simulated_robot import SimulatedRobot
from robot_planning.factory.factories import robot_factory_base
from robot_planning.factory.factories import renderer_factory_base
from robot_planning.factory.factory_from_config import factory_from_config
from robot_planning.factory.factories import logger_factory_base
import numpy as np
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'


def main():
    try:
        config_path = "configs/run_Autorally_CSSMPC.cfg"
        config_data = ConfigParser.ConfigParser()
        config_data.read(config_path)
        agent1 = factory_from_config(robot_factory_base, config_data, 'agent1')
        renderer1 = factory_from_config(renderer_factory_base, config_data, 'renderer1')
        logger = factory_from_config(logger_factory_base, config_data, 'logger')
        agent1.set_renderer(renderer=renderer1)
        logger.set_agent(agent=agent1)
        while not agent1.cost_evaluator.goal_checker.check(agent1.state):
            try:
                state_next, cost = agent1.take_action_with_controller()
            except RuntimeError:
                logger.add_number_of_failure()
                agent1.reset_state()
                agent1.reset_time()
            logger.calculate_number_of_laps(state_next, dynamics=agent1.dynamics, goal_checker=agent1.cost_evaluator.goal_checker)
            logger.calculate_number_of_collisions(state_next, dynamics=agent1.dynamics, collision_checker=agent1.cost_evaluator.collision_checker)
            logger.log()
            print("state: ", state_next)
            print("number of laps: ", logger.get_num_of_laps(), "number of collisions: ",
                  logger.get_num_of_collisions(), "number of controller failures: ", logger.get_num_of_failures())
    finally:
        logger.shutdown()

if __name__ == '__main__':
    main()
