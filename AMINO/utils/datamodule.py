
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
        each_data_len = [x.shape[self.dim] for x in batch]
        max_len = max(each_data_len)
        padded_each_data = [pad_tensor(x, max_len, self.dim) for x in batch]
        data = torch.stack(padded_each_data, dim=0)
        data_len = torch.tensor(each_data_len)
        return data, data_len

class MulPadCollate(object):
    def __init__(self, pad_choices, dim=0):
        super().__init__()
        self.dim = dim
        self.pad_choices = pad_choices

    def __call__(self, batch):
        # data extract
        datas = list()
        datas_len = list()
        for i, pad_choice in enumerate(self.pad_choices):
            if pad_choice:
                each_data = [x[i] for x in batch]
                each_data_len = [x.shape[self.dim] for x in each_data]
                max_len = max(each_data_len)
                padded_each_data = [pad_tensor(x, max_len, self.dim) for x in each_data]
                datas.append(torch.stack(padded_each_data, dim=0))
                datas_len.append(torch.tensor(each_data_len))
            else:
                each_data = [x[i] for x in batch]
                datas.append(torch.tensor(each_data))
                datas_len.append(
                    torch.tensor([x.shape[self.dim] for x in each_data])
                )
        return datas, datas_len