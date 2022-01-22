try:
    import ConfigParser
except ImportError:
    import configparser as ConfigParser

from robot_planning.trainers.trainer import Trainer
import unittest


class TestRunEnvironment(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        print("setUpClass")

    @classmethod
    def tearDownClass(cls):
        print("tearDownClass")

    def test_run_trainer(self):
        config_path = "configs/test_run_single_abstract_agent_trainer.cfg"
        config_data = ConfigParser.ConfigParser()
        config_data.read(config_path)
        trainer = Trainer()
        trainer.initialize_from_config(config_data=config_data, section_name='trainer')
        print('trainer initialized!')

        # test training
        trainer.train()

        # test evaluator
        evaluator = Trainer()
        evaluator.initialize_from_config(config_data=config_data, section_name='trainer')
        evaluator.load_model(episode=1, training=False)
        evaluator.evaluate(n_eval_episodes=1, eval_episode_length=5, visualize=True, save_animation=True)

        print('done')


if __name__ == '__main__':
    unittest.main()