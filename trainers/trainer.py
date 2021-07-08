import torch

class Trainer(object):
    def __init__(self, rl_framework=None, env=None):
        self.rl_framework = rl_framework
        self.env = env

    def run(self):
        pass

