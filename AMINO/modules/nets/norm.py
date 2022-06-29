import torch.nn as nn

class AUDIO_NORM(nn.BatchNorm2d):
    def __init__(self, num_features):
        super().__init__(num_features=num_features)

    def forward(self, x): 
        return nn.BatchNorm2d.forward(
            self, x.transpose(1, 3), 
        ).transpose(1, 3)