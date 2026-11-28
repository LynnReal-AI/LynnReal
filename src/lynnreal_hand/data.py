"""Raw example manifests and tensor-only conditioning caches."""
from pathlib import Path
from types import SimpleNamespace
import json
from lynnreal_core.backends.minimax_h3.geometry import latent_aligned_pixel_indices

NATIVE_LENGTHS = {124: 37, 158: 47, 243: 72, 260: 77}


def native_length(frames, stage):
    if type(frames) is not int or frames < 1:
        raise ValueError('The input frame count must be a positive integer')
    if stage == 'stage1':
        return len(latent_aligned_pixel_indices(frames, allow_partial=True))
    if frames not in NATIVE_LENGTHS:
        raise ValueError(f'Use a source-native Fast span from {list(NATIVE_LENGTHS)}')
    return NATIVE_LENGTHS[frames]


def load_example(path, *, verify_media=False, stage='stage3'):
    path = Path(path).resolve(); value = json.loads(path.read_text())
    for name in ('image', 'hand_video', 'camera', 'target_video', 'cache'):
        if value.get(name): value[name] = str((path.parent/value[name]).resolve())
    native_length(value['frames'], stage)
    for name in ('image', 'hand_video', 'camera'):
        if not Path(value[name]).is_file(): raise FileNotFoundError(value[name])
    if verify_media:
        from .media import probe
        info = probe(value['hand_video'])
        if info['frames'] != value['frames']:
            raise ValueError('Example manifest and hand video frame counts differ')
        if abs(info['fps']-float(value['fps'])) > 1e-4:
            raise ValueError('Example frame rate differs from the source video')
    value['_manifest_path'] = str(path)
    return value


def save_cache(tensors, metadata, directory):
    from safetensors.torch import save_file
    directory = Path(directory); directory.mkdir(parents=True, exist_ok=True)
    save_file({k: v.detach().cpu().contiguous() for k, v in tensors.items()}, str(directory/'conditions.safetensors'))
    (directory/'sample.json').write_text(json.dumps(metadata, indent=2)+'\n')
    return directory


def load_cache(directory):
    import torch
    from safetensors.torch import load_file
    directory = Path(directory).resolve(); meta = json.loads((directory/'sample.json').read_text())
    tensors = load_file(str(directory/'conditions.safetensors'), device='cpu')
    shape = tuple(meta['target_shape']); length = native_length(meta['pixel_frames'], meta.get('variant'))
    if len(shape) != 4 or shape[0] != 24 or shape[1] != length:
        raise ValueError('Invalid native target geometry')
    for key in ('prompt_embeds', 'text_token_tags', 'anchor_latents', 'hand_latents', 'camera_viewmats', 'camera_K'):
        if key not in tensors: raise ValueError(f'Condition cache lacks {key}')
    if tuple(tensors['hand_latents'].shape) != shape:
        raise ValueError('Hand conditions must retain the full native span and canvas')
    tail = tensors.get('base_hand_tail')
    if tail is not None:
        from .inference.base_windows import HISTORY_LATENTS, WINDOW_FRAMES, WINDOW_LATENTS, CHUNK_FRAMES, window_spans
        frames = meta['pixel_frames']
        end = window_spans(frames)[-1][0] // CHUNK_FRAMES * HISTORY_LATENTS + WINDOW_LATENTS
        if (meta.get('variant') != 'stage1' or frames <= WINDOW_FRAMES
                or tuple(tail.shape) != (shape[0], end-length, *shape[2:])):
            raise ValueError('Invalid Base hand-conditioning tail')
    if tensors['camera_viewmats'].shape != (length,4,4) or tensors['camera_K'].shape != (length,3,3):
        raise ValueError('Camera poses and intrinsics must be aligned to native latent frames')
    # Geometry placeholders never expose paired future RGB to generation.
    tensors.pop('target_latents', None)
    tensors.pop('teacher_hand_latents', None)
    target = torch.full(shape, float('nan'), dtype=torch.bfloat16)
    result = SimpleNamespace(**tensors, **{k:v for k,v in meta.items() if k not in tensors},
                             _cache_dir=str(directory))
    result.target_latents = target
    for key in ('source_video','source_hand_video'):
        if getattr(result,key,None):setattr(result,key,str((directory/getattr(result,key)).resolve()))
    result.task = meta.get('task', 'i2v_cam_hand')
    result.sample_id = meta['sample_id']; result.caption = meta['caption']
    result.pixel_frames = meta['pixel_frames']; result.validation = False
    result.source_frame_indices = tensors.get('source_frame_indices', torch.arange(result.pixel_frames))
    result.HAND_ROI = tensors.get('HAND_ROI')
    return result
