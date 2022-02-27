import torch.nn as nn

from AMINO.utils.init_object import init_object

class AMINO_ENC_DECS(nn.Module):
    def __init__(
        self,
        encoder,
        decoders,
    ):
        super().__init__()
        self.encoder = init_object(encoder)
        # decoder could be ’AMINO.modules.nets.decoder:AMINO_GP_DECODER‘
        # or "AMINO.modules.nets.decoder:AMINO_AUTOENCODER_DECODER"
        module_dict = {}
        for k, v in decoders.items():
            module_dict[k] = init_object(v)
        self.decoders = nn.ModuleDict(module_dict)

    def forward(self, xs, xs_len):
        # xs: (B, T, F)
        zs, zs_len, hs, hs_len = self.encoder(xs, xs_len)
        ys_dict = dict()
        ys_len_dict = dict()
        for key, decoder in self.decoders.items():
            ys, ys_len = decoder(hs, hs_len)
            ys_dict[key] = ys
            ys_len_dict[key] = ys_len
        return zs, zs_len, ys_dict, ys_len_dict
