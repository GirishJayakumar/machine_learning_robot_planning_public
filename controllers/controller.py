import numpy as np

class Controller(object):
    def __init__(self, dynamics_model=None, collision_checker=None, goal_checker=None, cost=None, gui=None):
        self.dynamics_model = dynamics_model
        self.collision_checker = collision_checker
        self.goal_checker = goal_checker
        self.cost = cost
        self.gui = gui

    def initialize_from_config(self, config_data, section_name):
        self.collision_checker = factory_from_config(collision_checker_factory_base, config_data,
                                                     collision_checker_section_name)
        self.goal_checker = factory_from_config(goal_evaluator_factory_base, config_data, goal_evaluator_section_name)


    def plan(self):
        raise NotImplementedError