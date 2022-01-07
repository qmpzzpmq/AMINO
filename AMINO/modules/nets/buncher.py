import torch.nn as nn

class SIMPLE_LINEAR_BUNCHER(nn.Module):
    def __init__(
        self,
        num_classes,
        num_hidden=256,
        num_layer = 1,
    ):
        super().__init__()
        num_dim = [num_hidden] * num_layer + [num_classes]
        self.net = nn.Sequential(*[
            nn.Linear(num_dim[i], num_dim[i+1]) for i in range(len(num_dim)-1)
        ])

    def forward(self, xs, xs_len):
        return self.net(xs), xs_len