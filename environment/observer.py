import numpy as np


class Observer(object):
    def __init__(self):
        pass

    def initialize_from_config(self, config_data, section_name):
        raise NotImplementedError

    def generate_observation(self, states):
        raise NotImplementedError


class FullStateObserver(Observer):
    def __init__(self):
        super(FullStateObserver).__init__()

    def initialize_from_config(self, config_data, section_name):
        pass

    def generate_observation(self, states):
        obs = np.concatenate(states)
        return obs
