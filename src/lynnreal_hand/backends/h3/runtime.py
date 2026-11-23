"""Activate the bundled, pinned H3 extensions without modifying site-packages."""
from pathlib import Path
import importlib
import importlib.abc
import importlib.util
import sys

HERE = Path(__file__).resolve().parent
_active = False


class H3ModuleFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if not fullname.startswith('diffusers.'): return None
        candidate = HERE/'overlay'/Path(*fullname.split('.'))
        if candidate.with_suffix('.py').is_file():
            return importlib.util.spec_from_file_location(fullname, candidate.with_suffix('.py'))
        if (candidate/'__init__.py').is_file():
            return importlib.util.spec_from_file_location(fullname, candidate/'__init__.py',
                                                         submodule_search_locations=[str(candidate)])
        return None


def activate():
    global _active
    if _active: return
    runtime = HERE/'_runtime'
    sys.path[:0] = [str(runtime), str(runtime/'native'), str(HERE.parents[1]/'models')]
    sys.meta_path.insert(0, H3ModuleFinder())
    import diffusers
    exports = {
        'AutoencoderKLMiniMaxH3':'models.autoencoders.autoencoder_kl_minimax_h3',
        'AutoencoderKLMiniMaxH3Audio':'models.autoencoders.autoencoder_kl_minimax_h3_audio',
        'MiniMaxH3Scheduler':'schedulers.scheduling_minimax_h3',
    }
    for name, module in exports.items():
        setattr(diffusers, name, getattr(importlib.import_module('diffusers.'+module), name))
    from geometry import install_geometry
    install_geometry()
    _active = True


def initialize(*, sp_size=1):
    import os
    import torch
    activate()
    local_rank = int(os.environ.get('LOCAL_RANK', '0'))
    if not torch.cuda.is_available(): raise RuntimeError('H3 execution requires a CUDA Hopper GPU')
    from lynnreal_core.backends.minimax_h3.fsdp import initialize_distributed
    initialize_distributed(sp_size=sp_size, local_rank=local_rank)
    device = torch.device('cuda', local_rank)
    from h3_fa3_policy import require_h3_fa3_training_backend
    from fa3_autograd_policy import install, verify_gpu
    require_h3_fa3_training_backend('_flash_3', device)
    install(); verify_gpu(device)
    return device


def configure_native_vae(vae):
    from diffusers.models.attention_dispatch import AttentionBackendName
    count = 0
    for module in vae.modules():
        processor = getattr(module, 'processor', None)
        if processor is not None and hasattr(processor, '_attention_backend'):
            processor._attention_backend = AttentionBackendName.NATIVE
            count += 1
    if not count: raise RuntimeError('Native VAE attention processors were not found')
    return vae
