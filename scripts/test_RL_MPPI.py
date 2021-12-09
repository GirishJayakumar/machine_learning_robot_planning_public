try:
    import ConfigParser
except ImportError:
    import configparser as ConfigParser
from robot_planning.environment.robots.simulated_robot import SimulatedRobot
from robot_planning.factory.factories import robot_factory_base
from robot_planning.factory.factories import renderer_factory_base
from robot_planning.factory.factories import logger_factory_base
from robot_planning.factory.factory_from_config import factory_from_config
import numpy as np


def main():
    config_path = "configs/run_RL_MPPI.cfg"
    config_data = ConfigParser.ConfigParser()
    config_data.read(config_path)
    agent1 = factory_from_config(robot_factory_base, config_data, 'agent1')
    renderer1 = factory_from_config(renderer_factory_base, config_data, 'renderer1')
    logger = factory_from_config(logger_factory_base, config_data, 'logger')
    agent1.set_renderer(renderer=renderer1)
    while not agent1.cost_evaluator.goal_checker.check(agent1.state):
        state_next, cost = agent1.take_action_with_controller()
        renderer1.show()
        time = agent1.get_time()
        logger.save_fig(renderer=renderer1, time=time)
        renderer1.clear()
        print(state_next, "    ", cost)
        agent1.set_goal(np.array([10.0 + 2 * time, 0.0, 0.0, 0.0, 1.0]))


if __name__ == '__main__':
    main()
