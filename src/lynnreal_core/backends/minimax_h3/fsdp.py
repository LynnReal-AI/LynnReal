"""Process-group initialization used by H3 inference."""
from datetime import timedelta
import torch
import torch.distributed as dist
from .distributed import init_sequence_parallel

def initialize_distributed(*, sp_size: int, local_rank: int) -> tuple[int, int]:
    """Initialize NCCL and H3's SP/DP process groups under torchrun."""

    import os

    world = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    torch.cuda.set_device(local_rank)
    if world > 1 and not dist.is_initialized():
        dist.init_process_group(backend="nccl", init_method="env://", timeout=timedelta(hours=1))
    init_sequence_parallel(sp_size)
    return rank, world
