from robot_planning.factory.factories import configs_generator_factory_base
from robot_planning.factory.factory_from_config import factory_from_config
import configparser as ConfigParser


def generate_batch_configs(config_path, template_config_path):
    config_data = ConfigParser.ConfigParser()
    config_data.read(config_path)
    template_config_data = ConfigParser.ConfigParser()
    template_config_data.read(template_config_path)
    generator = factory_from_config(configs_generator_factory_base, config_data, 'generate_batch_configs')
    generator.generate_batch_experiment_configs(template_config=template_config_data)


if __name__ == '__main__':
    config_path = "configs/Autorally_CSSMPC_generate_batch_configs.cfg"
    template_config_path = "configs/run_Autorally_CSSMPC.cfg"
    generate_batch_configs(config_path, template_config_path)
