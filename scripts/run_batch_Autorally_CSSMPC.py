from robot_planning.factory.factories import robot_factory_base
from robot_planning.factory.factories import renderer_factory_base
from robot_planning.factory.factory_from_config import factory_from_config
from robot_planning.scripts.generate_batch_configs import generate_batch_configs
from robot_planning.factory.factories import logger_factory_base
try:
    import ConfigParser
except ImportError:
    import configparser as ConfigParser
import ast


def run_batch_Autorally_CSSMPC(run_batch_Autorally_CSSMPC_config_path, batch_config_folder_path):
    config_data = ConfigParser.ConfigParser()
    config_data.read(run_batch_Autorally_CSSMPC_config_path)
    experiment_names = ast.literal_eval(config_data.get('run_batch_Autorally_CSSMPC', 'experiment_names'))
    for experiment_name in experiment_names:
        print("Running " + experiment_name)
        single_experiment_config_path = batch_config_folder_path + '/' + experiment_name + '.cfg'
        single_experiment_config_data = ConfigParser.ConfigParser()
        single_experiment_config_data.read(single_experiment_config_path)
        run_single_Autorally_CSSMPC(config_data=single_experiment_config_data, experiment_name=experiment_name)


def run_single_Autorally_CSSMPC(config_data, experiment_name):
        agent1 = factory_from_config(robot_factory_base, config_data, 'agent1')
        renderer1 = factory_from_config(renderer_factory_base, config_data, 'renderer1')
        logger = factory_from_config(logger_factory_base, config_data, 'logger')
        agent1.set_renderer(renderer=renderer1)
        logger.set_agent(agent=agent1)
        logger.set_experiment_name(experiment_name)
        while logger.get_num_of_laps() < 1:
            try:
                state_next, cost = agent1.take_action_with_controller()
            except:
                logger.add_number_of_failure()
                agent1.reset_state()
                agent1.reset_time()
            logger.calculate_number_of_laps(state_next, dynamics=agent1.dynamics,
                                            goal_checker=agent1.cost_evaluator.goal_checker)
            logger.calculate_number_of_collisions(state_next, dynamics=agent1.dynamics,
                                                  collision_checker=agent1.cost_evaluator.collision_checker)
            renderer1.show()
            renderer1.clear()
            logger.log()
            print("state: ", state_next)
            print("number of laps: ", logger.get_num_of_laps(), "number of collisions: ",
                  logger.get_num_of_collisions(), "number of controller failures: ", logger.get_num_of_failures())


if __name__ == '__main__':
    generate_batch_config_path = "configs/Autorally_CSSMPC_generate_batch_configs.cfg"
    template_config_path = "configs/run_Autorally_CSSMPC.cfg"
    generate_batch_configs(generate_batch_config_path, template_config_path)
    run_batch_Autorally_CSSMPC_config_path = "configs/run_batch_Autorally_CSSMPC.cfg"
    batch_config_folder_path = "configs/batch_configs"
    run_batch_Autorally_CSSMPC(run_batch_Autorally_CSSMPC_config_path, batch_config_folder_path)