import torch
import torch.nn as nn
import torch.nn.functional as F

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
        conv_clas = dynamic_import(conv_class)
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

class simple_autoencoder(nn.Module):
    def __init__(
        self,
        feature_dim=257,
        enc_embed_dim=256,
        enc_num_layers=5,
        enc_drop_out=0.3,
        dec_embed_dim=256,
        dec_num_layers=5,
        dec_drop_out=0.3,
    ):
        super().__init__()
        enc_dim = [feature_dim] + [enc_embed_dim] * enc_num_layers
        dec_dim = [dec_embed_dim] * dec_num_layers + [feature_dim]

        enc_layers = []
        for i in range(len(enc_dim)-1):
            enc_layers.append(nn.Dropout(p=enc_drop_out, inplace=False))
            enc_layers.append(nn.Linear(enc_dim[i], enc_dim[i+1]))
            enc_layers.append(nn.BatchNorm2d(1))
            enc_layers.append(nn.ReLU(inplace=True))
        dec_layers = []
        for i in range(len(dec_dim)-1):
            dec_layers.append(nn.Dropout(p=dec_drop_out, inplace=False))
            dec_layers.append(nn.Linear(dec_dim[i], dec_dim[i+1]))
            dec_layers.append(nn.BatchNorm2d(1))
            dec_layers.append(nn.ReLU(inplace=True))
        self.enc = nn.Sequential(*enc_layers)
        self.dec = nn.Sequential(*dec_layers)
        # self.weight_init()

    def weight_init(self):
        for layers in [self.enc, self.dec]:
            for layer in layers:
                if type(layer) == nn.Linear:
                    nn.init.kaiming_uniform_(layer.weight, nonlinearity='relu')
                    nn.init.normal_(layer.bias)

    def forward(self, x):
        # x: (B, C, T, F)
        h = self.enc(x)
        y = self.dec(h)
        return y

class conv_autoencoder(nn.Module):
    def __init__(
        self,
        channels=[64, 128, 256, 512],
        enc_dropout_p=0.2,
        dec_dropout_p=0.2,
        resume_from_cnn10=None,
    ):
        super().__init__()
        layers = []
        channels = [1] + channels
        num_layer = len(channels) - 1
        for i in range(num_layer):
            layers.append(
                ConvBlock(
                    in_channels=channels[i],
                    out_channels=channels[i + 1],
                )
            )
            layers.append(
                nn.Dropout(enc_dropout_p)
            )
        self.enc = nn.Sequential(*layers)
        layers = []
        for i in reversed(range(num_layer)):
            layers.append(
                ConvBlock(
                    in_channels=channels[i + 1],
                    out_channels=channels[i],
                    conv_class="torch.nn:ConvTranspose2d",
                )
            )
            layers.append(
                nn.Dropout(dec_dropout_p)
            )
        self.dec = nn.Sequential(*layers)
        if resume_from_cnn10 is not None:
            self.resume_from_cnn10(resume_from_cnn10)

    def forward(self, x):
        # x: (B, C, T, F)
        h = self.enc(x)
        y = self.dec(h)
        return y
    
    def resume_from_cnn10(self, path):
        #self == model
        model_dict = self.state_dict()
        pretrained_dict = torch.load(path)
        pretrained_model_dict = pretrained_dict.get("model")
        pretrained_model_dict = {k: v for k, v in pretrained_model_dict.items() if k in model_dict}
        model_dict.update(pretrained_model_dict)
        self.load_state_dict(model_dict)
