from typing import Union

import torch
from torch import nn

Activation = Union[str, nn.Module]


_str_to_activation = {
    'relu': nn.ReLU(),
    'tanh': nn.Tanh(),
    'leaky_relu': nn.LeakyReLU(),
    'sigmoid': nn.Sigmoid(),
    'selu': nn.SELU(),
    'softplus': nn.Softplus(),
    'identity': nn.Identity(),
}


def build_mlp(
        input_size: int,
        output_size: int,
        n_layers: int,
        size: int,
        activation: Activation = 'tanh',
        output_activation: Activation = 'identity',
) -> nn.Module:
    """
    Builds a feedforward neural network (MLP).
    """
    # Map string to activation instances
    if isinstance(activation, str):
        activation = _str_to_activation.get(activation, nn.Tanh())
    if isinstance(output_activation, str):
        output_activation = _str_to_activation.get(output_activation, nn.Identity())

    layers = []

    # First layer
    layers.append(nn.Linear(input_size, size))
    layers.append(activation)  # instance, no ()

    # Hidden layers
    for _ in range(n_layers - 1):
        layers.append(nn.Linear(size, size))
        layers.append(activation)  # instance, no ()

    # Output layer
    layers.append(nn.Linear(size, output_size))
    layers.append(output_activation)  # instance, no ()

    return nn.Sequential(*layers)


device = None


def init_gpu(use_gpu=True, gpu_id=0):
    global device
    if torch.cuda.is_available() and use_gpu:
        device = torch.device("cuda:" + str(gpu_id))
        print("Using GPU id {}".format(gpu_id))
    else:
        device = torch.device("cpu")
        print("GPU not detected. Defaulting to CPU.")


def set_device(gpu_id):
    torch.cuda.set_device(gpu_id)


def from_numpy(*args, **kwargs):
    return torch.from_numpy(*args, **kwargs).float().to(device)


def to_numpy(tensor):
    return tensor.to('cpu').detach().numpy()
