import torch
from torch.nn import Module

def np2tensor(np_array, requires_grad=False):
    tensor = torch.tensor(np_array, dtype=torch.float32, requires_grad=requires_grad)
    return tensor

def hard_update(target_net: Module, source_net: Module):
    for target_param, source_param in zip(target_net.parameters(), source_net.parameters()):
        target_param.data.copy_(source_param.data)


def soft_update(target_net: Module, source_net: Module, tau):
    for target_param, source_param in zip(target_net.parameters(), source_net.parameters()):
        target_param.data.copy_(target_param.data * (1 - tau) + source_param.data * tau)