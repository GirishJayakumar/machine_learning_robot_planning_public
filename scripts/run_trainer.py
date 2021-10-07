try:
    import ConfigParser
except ImportError:
    import configparser as ConfigParser

from robot_planning.trainers.trainer import Trainer

EVALUATE_ONLY = True

def main():
    config_path = "configs/run_trainer.cfg"
    config_data = ConfigParser.ConfigParser()
    config_data.read(config_path)

    if not EVALUATE_ONLY:
        trainer = Trainer()
        trainer.initialize_from_config(config_data=config_data, section_name='trainer')
        print('trainer initialized!')

        # train
        trainer.train()

    # evaluate
    evaluator = Trainer()
    evaluator.initialize_from_config(config_data=config_data, section_name='trainer')
    evaluator.load_model(episode=400, training=False)
    evaluator.evaluate(n_eval_episodes=50, eval_episode_length=200, visualize=True)


    print('done!')


if __name__ == '__main__':
    main()
