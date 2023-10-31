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
import matplotlib.pyplot as plt


os.environ['KMP_DUPLICATE_LIB_OK']='True'


def main():
    try:
        config_path = "configs/run_Autorally_CSSMPC.cfg"
        config_data = ConfigParser.ConfigParser()
        config_data.read(config_path)
        agent1 = factory_from_config(robot_factory_base, config_data, 'agent1')
        agent2 = factory_from_config(robot_factory_base, config_data, 'agent2')
        renderer1 = factory_from_config(renderer_factory_base, config_data, 'renderer1')
        logger = factory_from_config(logger_factory_base, config_data, 'logger')
        agent1.set_renderer(renderer=renderer1)
        agent2.set_renderer(renderer=renderer1)
        logger.set_agent(agent=agent1)
        while not agent1.cost_evaluator.goal_checker.check(agent1.state):
            try:
                state_next, cost = agent1.take_action_with_controller(agent2.state)
                opponent_state_prediction = agent1.predict_opponent_state(agent2.state)
                multi_step_prediction = [opponent_state_prediction]
                #multi step prediction
                for _ in range(1):
                    opponent_state_prediction = agent1.predict_opponent_state(opponent_state_prediction)
                    multi_step_prediction.append(opponent_state_prediction)
                state_next2, cost = agent2.take_action_with_controller(agent1.state)
                logger.calculate_number_of_laps(state_next, dynamics=agent1.dynamics,
                                                goal_checker=agent1.cost_evaluator.goal_checker)
                logger.calculate_number_of_collisions(state_next, dynamics=agent1.dynamics,
                                                      collision_checker=agent1.cost_evaluator.collision_checker)
                print("state1: ", state_next)
                print("state2: ", state_next2)
                print("opp_pred: ", multi_step_prediction)
                #look at renderers file for axis method
                #use opponent outputs and predicted outputs (based on opponent input data)
                #The optimal value found for dimension 0 of parameter k2__length_scale is close to the specified lower bound 1e-05. Decreasing the bound and calling fit again may find a better value.
                for curr_pred in multi_step_prediction:
                    renderer1.showOpponentPrediction(curr_pred[-2], curr_pred[-1])
                renderer1.showCurrent(state_next2[-2], state_next2[-1])
            except RuntimeError:
                logger.add_number_of_failure()
                agent1.reset_state()
                agent1.reset_time()
                agent2.reset_state()
                agent2.reset_time()
            logger.log()
            print("number of laps: ", logger.get_num_of_laps(), "number of collisions: ",
                  logger.get_num_of_collisions(), "number of controller failures: ", logger.get_num_of_failures())
    finally:
        logger.shutdown()

if __name__ == '__main__':
    main()
