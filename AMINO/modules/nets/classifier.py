import torch.nn as nn

from AMINO.utils.init_object import init_object

class AMINO_CLASSIFIER(nn.Module):
    def __init__(
        self,
        encoder,
        decoders,
    ):
        super().__init__()
        self.encoder = init_object(encoder)
        # decoder could be ’AMINO.modules.nets.classifier:AMINO_GP_DECODER‘
        # and ‘’
        self.decoder = nn.ModuleList([
            init_object(decoder) for decoder in decoders
        ])
    def forward(self, xs, xs_len):
        # xs: (B, T, F)
        hs, hs_len = self.encoder(xs, xs_len)
        ys_list = []
        ys_len_list = []
        for decoder in self.decoders:
            ys, ys_len = decoder(hs, hs_len)
            ys_list.append(ys)
            ys_len_list.append(ys_len)
        return ys_list, ys_len_list

class AMINO_GP_DECODER(nn.Module):
    def __init__(
        self,
        buncher,
        pooler,
    ):
        super().__init__()
        self.buncher = init_object(buncher)
        self.pooler = init_object(pooler)
        
    def forward(self, hs, hs_len):
        # hs: (B, T, H)
        cs, cs_len = self.buncher(hs, hs_len)
        # cs: (B, C)
        ys, ys_len = self.pooler(cs, cs_len)
        # ys: (B, T, C)
        return ys, ys_len

class AMINO_AUTOENCDOER_DECODER(nn.Module):
    def __init__(
        self,
        decoder,
    ):
        super().__init__()
        self.decoder = init_object(decoder)

    def forward(self, xs, xs_len):
        return self.decoder(xs, xs_len)
