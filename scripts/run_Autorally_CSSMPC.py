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
from Plotter import Plotter

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

        all_predictions = []
        all_states_next2 = []
        all_times = []
        while not agent1.cost_evaluator.goal_checker.check(agent1.state):
        #for _ in range(30):
            try:
                state_next, cost, opponent_state_prediction = agent1.take_action_with_controller(agent2.state)
                multi_step_prediction = [opponent_state_prediction]
                state_next2, cost, opponent_state_prediction2 = agent2.take_action_with_controller(agent1.state)
                #multi step prediction
                """for _ in range(1):
                    opponent_state_prediction = agent1.predict_opponent_state(opponent_state_prediction)
                    multi_step_prediction.append(opponent_state_prediction)"""
                logger.calculate_number_of_laps(state_next, dynamics=agent1.dynamics,
                                                goal_checker=agent1.cost_evaluator.goal_checker)
                logger.calculate_number_of_collisions(state_next, dynamics=agent1.dynamics,
                                                      collision_checker=agent1.cost_evaluator.collision_checker)
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
            all_predictions.append([opponent_state_prediction[-2], opponent_state_prediction[-1]])
            """map_state_next2 = np.copy(state_next2)
            x_y_coords = [state_next2[-2], state_next2[-1]]
            M = np.array([[x_y_coords[0]], [x_y_coords[1]]])
            new_header, d_val, s_val = agent2.dynamics.track.localize(M, map_state_next2[-3])
            map_state_next2[-3], map_state_next2[-2], map_state_next2[-1] = new_header, d_val, s_val"""
            all_states_next2.append([state_next2[-2], state_next2[-1]])
            #all_times.append(_ + 1)
            #print("allPreds: ", all_predictions)
            #print("allStates: ", all_states_next2)
            print("number of laps: ", logger.get_num_of_laps(), "number of collisions: ",
                  logger.get_num_of_collisions(), "number of controller failures: ", logger.get_num_of_failures())
        plotter = Plotter()
        plotter.plot_prediction_and_state(all_predictions, all_states_next2)
        #plotter.plot_s_and_time(all_predictions, all_states_next2, all_times)
        #plotter.plot_d_and_time(all_predictions, all_states_next2, all_times)

        plotter.show()
    finally:
        logger.shutdown()

if __name__ == '__main__':
    main()
