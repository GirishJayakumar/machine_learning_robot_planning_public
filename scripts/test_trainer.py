try:
    import ConfigParser
except ImportError:
    import configparser as ConfigParser

from robot_planning.trainers.trainer import Trainer


def main():
    config_path = "configs/test_trainer.cfg"
    config_data = ConfigParser.ConfigParser()
    config_data.read(config_path)
    trainer = Trainer()
    trainer.initialize_from_config(config_data=config_data, section_name='trainer')
    print('trainer initialized!')

    # test training
    trainer.train()

    # test loading and train
    trainer.load_model(episode=8, training=True)
    trainer.train()

    # test evaluater
    evaluator = Trainer()
    evaluator.initialize_from_config(config_data=config_data, section_name='trainer')
    evaluator.load_model(episode=9, training=False)
    evaluator.evaluate(n_eval_episodes=10, eval_episode_length=100, visualize=True)


    print('done!')


if __name__ == '__main__':
    main()
