import torch
import torch.nn as nn

class SIMPLE_POOLER(nn.Module):
    def __init__(
        self,
        pooling_method="mean",
    ):
        super().__init__()
        if pooling_method == "mean":
            self.method = torch.mean
        elif pooling_method == "max":
            self.metthod = torch.max

    def forward(self, xs, xs_len):
        # xs: (B, T, C)
        xs = self.method(xs, dim=1)
        # xs: (B, C)
        return xs, xs_len