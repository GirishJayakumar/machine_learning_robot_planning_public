import torch

def np2tensor(np_array, requires_grad=False):
    tensor = torch.tensor(np_array, dtype=torch.float32, requires_grad=requires_grad)
    return tensor