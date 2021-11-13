import torch.nn as nn

from AMINO.utils.dynamic_import import dynamic_import

def init_layer(layer):
    """Initialize a Linear or Convolutional layer. """
    nn.init.xavier_uniform_(layer.weight)
 
    if hasattr(layer, 'bias'):
        if layer.bias is not None:
            layer.bias.data.fill_(0.)

def init_bn(bn):
    """Initialize a Batchnorm layer. """
    bn.bias.data.fill_(0.)
    bn.weight.data.fill_(1.)

class ConvBlock(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            conv_class="torch.nn:Conv2d",
            pool_size=(2, 2),
            pool_class="torch.nn:AvgPool2d",    
        ):
        super().__init__()
        conv_class = dynamic_import(conv_class)
        pool_class = dynamic_import(pool_class)
        self.net = nn.Sequential(
            conv_class(
                in_channels=in_channels, 
                out_channels=out_channels,
                kernel_size=(3, 3), stride=(1, 1),
                    padding=(1, 1), bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True),
            conv_class(
                in_channels=out_channels, 
                out_channels=out_channels,
                kernel_size=(3, 3), stride=(1, 1),
                padding=(1, 1), bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True),
            pool_class(pool_size),
        )
        self.init_weight()
        
    def init_weight(self):
        for layer in self.net:
            if type(layer) == nn.Conv2d:
                init_layer(layer)
                continue
            elif type(layer) == nn.BatchNorm2d:
                init_bn(layer)
                continue

    def forward(self, x):
        return self.net(x)

class depthwise_separable_conv(nn.Module):
    def __init__(self, nin, nout):
        super(depthwise_separable_conv, self).__init__()
        self.depthwise = nn.Conv2d(nin, nin, kernel_size=3, padding=1, groups=nin)
        self.pointwise = nn.Conv2d(nin, nout, kernel_size=1)

    def forward(self, x):
        out = self.depthwise(x)
        out = self.pointwise(out)
        return out

class conv_autoencoder(nn.Module):
    def __init__(
        self,
        channels=[64, 128, 256, 512],
        enc_dropout_p=0.2,
        dec_dropout_p=0.2,
    ):
        super().__init__()
        channels = [1] + channels
        num_layer = len(channels) - 1
        layers = []
        for i in range(num_layer):
            layers.append(
                nn.Conv2d(
                    channels[i],
                    channels[i+1],
                    (3, 3), padding=(1,1),
                )
            )
            layers.append(nn.Dropout(enc_dropout_p))
        self.enc = nn.Sequential(*layers)
        layers = []
        for i in range(num_layer):
            layers.append(
                nn.Conv2d(
                    channels[num_layer-i],
                    channels[num_layer-i-1],
                    (3, 3), padding=(1,1),
                )
            )
            layers.append(nn.Dropout(dec_dropout_p))
        self.dec = nn.Sequential(*layers)

    def forward(self, x):
        # x: (B, C, T, F)
        h = self.enc(x)
        y = self.dec(h)
        return y