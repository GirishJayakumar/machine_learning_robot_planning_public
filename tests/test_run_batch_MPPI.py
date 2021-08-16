try:
    import ConfigParser
except ImportError:
    import configparser as ConfigParser
from robot_planning.factory.factories import robot_factory_base
from robot_planning.factory.factories import renderer_factory_base
from robot_planning.factory.factory_from_config import factory_from_config
from robot_planning.scripts.generate_batch_configs import generate_batch_configs
import ast
from robot_planning.scripts.generate_batch_configs import generate_batch_configs
from robot_planning.scripts.run_batch_MPPI import run_batch_MPPI
from robot_planning.scripts.run_batch_MPPI import run_batch_MPPI_with_rendering_and_saving
import unittest


class TestRunBatchMPPI(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        print("setUpClass")

    @classmethod
    def tearDownClass(cls):
        print("tearDownClass")

    def test_run_batch_MPPI(self):
        generate_batch_config_path = "configs/test_generate_batch_configs.cfg"
        template_config_path = "configs/test_run_MPPI.cfg"
        generate_batch_configs(generate_batch_config_path, template_config_path)
        run_batch_MPPI_config_path = "configs/run_batch_MPPI.cfg"
        batch_config_folder_path = "configs/batch_configs"
        run_batch_MPPI(run_batch_MPPI_config_path, batch_config_folder_path)

    def test_run_batch_MPPI_with_rendering_and_saving(self):
        generate_batch_config_path = "configs/test_generate_batch_configs.cfg"
        template_config_path = "configs/test_run_MPPI.cfg"
        generate_batch_configs(generate_batch_config_path, template_config_path)
        run_batch_MPPI_config_path = "configs/run_batch_MPPI.cfg"
        batch_config_folder_path = "configs/batch_configs"
        run_batch_MPPI_with_rendering_and_saving(run_batch_MPPI_config_path, batch_config_folder_path)


if __name__ == '__main__':
    unittest.main()
