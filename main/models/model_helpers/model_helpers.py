import torch.nn as nn

def get_activation(name):
    """Helper to map config string to PyTorch class"""
    if name == 'ReLU': return nn.ReLU()
    if name == 'SiLU': return nn.SiLU()
    if name == 'Tanh': return nn.Tanh()
    if name == 'LeakyReLU': return nn.LeakyReLU()
    raise ValueError(f"Unknown activation: {name}")