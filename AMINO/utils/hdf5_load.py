import torch
from torch.nn.parameter import Parameter

def bn2d_load(weight, layer, device=None):
    device = layer.weight.device if device is None else device
    with torch.no_grad():
        layer.weight = Parameter(torch.as_tensor(weight["beta:0"]).to(device))
        layer.bias = Parameter(torch.as_tensor(weight["gamma:0"]).to(device))
        layer.running_mean = Parameter(
            torch.as_tensor(weight["moving_mean:0"]).to(device)
        )
        layer.running_var = Parameter(
            torch.as_tensor(weight['moving_variance:0']).to(device)
        )

def conv2d_load(weight, layer, device=None):
    device = layer.weight.device if device is None else device
    with torch.no_grad():
        layer.weight = Parameter(torch.as_tensor(weight['kernel:0']).to(device))
        layer.bias = Parameter(torch.as_tensor(weight['bias:0']).to(device))
