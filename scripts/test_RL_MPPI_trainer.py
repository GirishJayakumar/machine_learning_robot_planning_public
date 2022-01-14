try:
    import ConfigParser
except ImportError:
    import configparser as ConfigParser

from robot_planning.trainers.trainer import Trainer


def main():
    config_path = "configs/test_RL_MPPI_trainer.cfg"
    config_data = ConfigParser.ConfigParser()
    config_data.read(config_path)
    trainer = Trainer()
    trainer.initialize_from_config(config_data=config_data, section_name='trainer')
    print('trainer initialized!')

    # # test visualize
    # trainer.evaluate(visualize=True, n_eval_episodes=1)

    # test training
    # trainer.train()

    # test loading and train
    # trainer.load_model(episode=1999, training=True)
    # trainer.train()

    # test evaluater
    evaluator = Trainer()
    evaluator.initialize_from_config(config_data=config_data, section_name='trainer')
    evaluator.load_model(episode=250, training=False)
    evaluator.evaluate(n_eval_episodes=1, eval_episode_length=30, visualize=True, save_animation=True)


    print('done!')


if __name__ == '__main__':
    main()
