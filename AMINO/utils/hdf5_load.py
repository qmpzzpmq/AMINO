import numpy as np

import torch
from torch.nn.parameter import Parameter

def weight_check_return(weight, readin, device):
    readin_tensor = torch.as_tensor(readin)
    assert weight.shape == readin_tensor.shape, f"assign weight shape unequal"
    return Parameter(readin_tensor.to(device))

def bn2d_load(weight, layer, device=None):
    device = layer.weight.device if device is None else device
    with torch.no_grad():
        layer.weight = weight_check_return(layer.weight, weight["beta:0"], device)
        layer.bias = weight_check_return(layer.bias, weight["gamma:0"], device)
        layer.running_mean = weight_check_return(
            layer.running_mean, weight["moving_mean:0"], device
        )
        layer.running_var = weight_check_return(
            layer.running_var, weight["moving_variance:0"], device
        )

def conv2d_load(weight, layer, device=None):
    device = layer.weight.device if device is None else device
    with torch.no_grad():
        layer.weight = weight_check_return(
            layer.weight, np.array(weight["kernel:0"]).transpose(1,0), device
        )
        layer.bias = weight_check_return(layer.bias, weight["bias:0"], device)
