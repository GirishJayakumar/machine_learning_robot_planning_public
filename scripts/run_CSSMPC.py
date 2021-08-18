try:
    import ConfigParser
except ImportError:
    import configparser as ConfigParser
from robot_planning.environment.robots.simulated_robot import SimulatedRobot
from robot_planning.factory.factories import robot_factory_base
from robot_planning.factory.factories import renderer_factory_base
from robot_planning.factory.factory_from_config import factory_from_config
import numpy as np

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'



def main():
    # config_path = "configs/batch_configs/exp_obs1x2.0_obs1y0.0_obs1r1.5_obs2x6.0_obs2y0.0_obs2r1.5_gamma1.0_alpha0.0_Numtraj100.0.cfg"
    config_path = "configs/run_CSSMPC.cfg"
    config_data = ConfigParser.ConfigParser()
    config_data.read(config_path)
    agent1 = factory_from_config(robot_factory_base, config_data, 'agent1')
    renderer1 = factory_from_config(renderer_factory_base, config_data, 'renderer1')
    agent1.set_renderer(renderer=renderer1)
    while not 0:
        state_next, cost = agent1.take_action_with_controller()
        renderer1.show()
        renderer1.clear()
        print(state_next, "    ", cost)


if __name__ == '__main__':
    main()
