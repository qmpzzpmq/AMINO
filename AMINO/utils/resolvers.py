from omegaconf import OmegaConf

def register_OmegaConf_resolvers():
    OmegaConf.register_new_resolver("nfft2fea_dim", lambda x: int(x / 2 + 1))
    OmegaConf.register_new_resolver("product", lambda x, y: x * y)
    OmegaConf.register_new_resolver("list_reversed", lambda x: list(reversed(x)))