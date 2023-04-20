import os
import torch
import datetime
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler

def local_ddp_init(model, current_rank, backend="nccl", timeout=datetime.timedelta(seconds=180)):
    dist.init_process_group(
        backend=backend,
        timeout=timeout,
        rank=current_rank,
    )
    device_name = 'cuda' + str(current_rank)
    alloc_device = torch.device(alloc_device)
    model.to(alloc_device)
    print(f'Rank {device_name} is using device {device_name}')
    model = torch.nn.parallel.DistributedDataParallel(
        model,
        device_ids=[current_rank])
    return model, alloc_device