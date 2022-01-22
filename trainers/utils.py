import torch
from torch.nn import Module
from typing import List


def np2tensor(np_array, requires_grad=False):
    tensor = torch.tensor(np_array, dtype=torch.float32, requires_grad=requires_grad)
    return tensor


def list_np2list_tensor(list_np: List, requires_grad=False):
    list_tensor = [torch.tensor(list_np[i], dtype=torch.float32, requires_grad=requires_grad) for i in
                   range(len(list_np))]
    return list_tensor


def hard_update(target_net: Module, source_net: Module):
    for target_param, source_param in zip(target_net.parameters(), source_net.parameters()):
        target_param.data.copy_(source_param.data)


def soft_update(target_net: Module, source_net: Module, tau):
    for target_param, source_param in zip(target_net.parameters(), source_net.parameters()):
        target_param.data.copy_(target_param.data * (1 - tau) + source_param.data * tau)

def print_table(header, data):
    from tabulate import tabulate
    assert len(header) == len(data[0])
    print(tabulate(data, headers=header, floatfmt=".3f"))