from omegaconf import OmegaConf
def element_cat(*x):
    return list(x)

def register_OmegaConf_resolvers():
    OmegaConf.register_new_resolver("nfft2fea_dim", lambda x: int(x / 2 + 1))
    OmegaConf.register_new_resolver("product", lambda x, y: x * y)
    OmegaConf.register_new_resolver("list_reversed", lambda x: list(reversed(x)))
    OmegaConf.register_new_resolver("plus", lambda x, y: x + y)
    OmegaConf.register_new_resolver("extract_last", lambda x: x[-1])
    OmegaConf.register_new_resolver("cat", element_cat)