try:
    import ConfigParser
except ImportError:
    import configparser as ConfigParser
from robot_planning.factory.factories import robot_factory_base
from robot_planning.factory.factories import renderer_factory_base
from robot_planning.factory.factory_from_config import factory_from_config
from robot_planning.scripts.generate_batch_configs import generate_batch_configs
from robot_planning.batch_experimentation.loggers import MPPILogger
from robot_planning.factory.factories import logger_factory_base
import ast


def run_batch_MPPI(run_batch_MPPI_config_path, batch_config_folder_path):
    config_data = ConfigParser.ConfigParser()
    config_data.read(run_batch_MPPI_config_path)
    experiment_names = ast.literal_eval(config_data.get('run_batch_MPPI', 'experiment_names'))
    for experiment_name in experiment_names:
        print("Running " + experiment_name)
        single_experiment_config_path = batch_config_folder_path + '/' + experiment_name + '.cfg'
        single_experiment_config_data = ConfigParser.ConfigParser()
        single_experiment_config_data.read(single_experiment_config_path)
        agent1 = factory_from_config(robot_factory_base, single_experiment_config_data, 'agent1')
        agent1.set_renderer(renderer=None)
        while not agent1.cost_evaluator.goal_checker.check(agent1.state):
            state_next, cost = agent1.take_action_with_controller()
        print("goal reached!")


def run_batch_MPPI_with_rendering_and_saving(run_batch_MPPI_config_path, batch_config_folder_path):
    config_data = ConfigParser.ConfigParser()
    config_data.read(run_batch_MPPI_config_path)
    experiment_names = ast.literal_eval(config_data.get('run_batch_MPPI', 'experiment_names'))
    for experiment_name in experiment_names:
        print("Running " + experiment_name)
        single_experiment_config_path = batch_config_folder_path + '/' + experiment_name + '.cfg'
        single_experiment_config_data = ConfigParser.ConfigParser()
        single_experiment_config_data.read(single_experiment_config_path)

        logger = factory_from_config(logger_factory_base, single_experiment_config_data, 'logger')
        renderer1 = factory_from_config(renderer_factory_base, single_experiment_config_data, 'renderer1')
        agent1 = factory_from_config(robot_factory_base, single_experiment_config_data, 'agent1')
        agent1.set_renderer(renderer=renderer1)
        logger.set_experiment_name(experiment_name)
        while not agent1.cost_evaluator.goal_checker.check(agent1.state):
            state_next, cost = agent1.take_action_with_controller()
            time = agent1.get_time()
            logger.save_fig(renderer=renderer1, time=time)
            renderer1.clear()
        print("goal reached!")


if __name__ == '__main__':
    generate_batch_config_path = "configs/MPPI_generate_batch_configs.cfg"
    template_config_path = "configs/run_MPPI.cfg"
    generate_batch_configs(generate_batch_config_path, template_config_path)
    run_batch_MPPI_config_path = "configs/run_batch_MPPI.cfg"
    batch_config_folder_path = "configs/batch_configs"
    run_batch_MPPI_with_rendering_and_saving(run_batch_MPPI_config_path, batch_config_folder_path)
