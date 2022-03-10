import numpy as np
import torch
import torchaudio

def torch_version_print():
    print(f"torch vision: {torch.__version__}")
    print(f"torchaudio vision: {torchaudio.__version__}")
    print(f"numpy vision: {np.__version__}")

def DDP_test(ddp_backend="nccl"):
    gpu_avail = torch.cuda.is_available()
    print(f"gpu available: {gpu_avail}")
    if gpu_avail:
        print(f"GPU num: {torch.cuda.device_count()}")
        print(f"cuda version: {torch.version.cuda}")
        torch.distributed.init_process_group(ddp_backend)
        print(f"world size: {torch.distributed.get_world_size()}")
        for idx in range(torch.cuda.device_count()):
            a = torch.rand([5, 3]).to(device=torch.device('cuda', idx)).cpu().numpy()
            print(f"GPU {idx} succeed")
    else:
        a = torch.rand([5, 3]).cpu().numpy()

def main():
    torch_version_print()
    DDP_test()

if __name__ == "__main__":
    main()
