import logging

import torch

def pad_tensor(vec, pad, dim):
    """
    args:
        vec - tensor to pad
        pad - the size to pad to
        dim - dimension to pad
    return:
        a new tensor padded to 'pad' in dimension 'dim'
    """
    pad_size = list(vec.shape)
    pad_size[dim] = pad - vec.size(dim)
    return torch.cat(
        [vec, torch.zeros(
            *pad_size, dtype=vec.dtype, device=vec.device)]
        , dim=dim
    )

def singlepadcollate(batch, dim=-1):
    each_data_len = [x.shape[dim] for x in batch]
    max_len = max(each_data_len)
    padded_each_data = [pad_tensor(x, max_len, dim) for x in batch]
    data = torch.stack(padded_each_data, dim=0)
    data_len = torch.tensor(each_data_len)
    return data, data_len

class Pad_tensor(object):
    def __init__(self, pad, dim):
        self.pad = pad
        self.dim = dim

    def __call__(self, vec):
        return pad_tensor(vec, self.pad, self.dim)

class SinglePadCollate(object):
    def __init__(self, dim=0):
        self.dim = dim

    def __call__(self, batch):
        return singlepadcollate(batch, self.dim)

class MulPadCollate(object):
    def __init__(self, pad_choices, dim=-1):
        self.dim = dim
        self.pad_choices = pad_choices

    def __call__(self, batch):
        # data extract
        datas = list()
        datas_len = list()
        batch = [x for x in batch if x is not None]
        if len(batch) == 0:
            return None
        for i, pad_choice in enumerate(self.pad_choices):
            each_data = [x[i] for x in batch]
            if pad_choice == "pad":
                each_data_len = [x.shape[self.dim] for x in each_data]
                max_len = max(each_data_len)
                padded_each_data = [pad_tensor(x, max_len, self.dim) for x in each_data]
                datas.append(torch.stack(padded_each_data, dim=0))
                datas_len.append(torch.tensor(each_data_len))
            elif pad_choice == "unpad":
                datas.append(torch.stack(each_data))
                datas_len.append(
                    torch.tensor([x.shape[self.dim] for x in each_data])
                )
            elif pad_choice == "onehot":
                datas.append(torch.stack(each_data))
                datas_len.append(
                    torch.ones(len(each_data), dtype=torch.int)
                )
            else:
                raise ValueError(f"{pad_choice} is not implement")
        return datas, datas_len

class AMINOPadCollate(MulPadCollate):
    def __init__(self, pad_choices=["pad", "unpad"]):
        super().__init__(pad_choices, dim=-1)
    def __call__(self, batch):
        if ( outs := MulPadCollate.__call__(self, batch) ) is None:
            return None
        datas, datas_len = outs
        return {
            'feature': {'data': datas[0], 'len': datas_len[0]},
            'label': {'data': datas[1], 'len': datas_len[1]},
        }
