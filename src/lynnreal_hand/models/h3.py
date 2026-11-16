"""Load the Base or Fast inference transformer."""
from pathlib import Path


def _install_inference_kernels(model, *, role):
    from exact_kernels.install import install as install_exact
    # Exact block kernels require the original H3 attention processor. Install
    # them before the streaming processor replaces it, as in the source engine.
    install_exact(model)
    if role == 'student':
        from inference_core import install_student_attention
        install_student_attention(model)


def build_model(config, device, *, role='base'):
    if role not in ('base', 'student'):
        raise ValueError('Only Base and Fast inference models are included')
    from ..backends.h3.runtime import activate
    activate()
    from h3_components.loader import load_transformer
    model_cfg = dict(config['model'])
    if not (Path(model_cfg['checkpoint_path'])/'transformer/config.json').is_file():
        raise FileNotFoundError('Place the selected transformer weights under checkpoints; see docs/weights.md')
    modules = load_transformer(model_cfg, device=device)
    model = modules.transformer
    expected = 33122992896 + (sum(p.numel() for p in model.delta_embedding.parameters())
                            if model.uses_anyflow else 0)
    packaged_adaln = model.config.get('adaln_schedule') is not None
    if packaged_adaln:
        hidden, time_dim = model.config.hidden_size, model.config.time_embed_dim
        expected -= model.config.num_layers * (time_dim + 1) * 6 * hidden * 3
    if sum(p.numel() for p in model.parameters()) != expected:
        raise ValueError('Transformer parameters do not match the declared model layout')
    if role == 'student':
        if not model.uses_anyflow:
            raise ValueError('Fast requires a causal transformer with streaming time conditioning')
    elif model.uses_anyflow:
        raise ValueError('Base inference requires the Base transformer')
    _install_inference_kernels(model, role=role)
    model.eval().requires_grad_(False)
    memory = None
    if packaged_adaln:
        from ..inference.adaln import validate_packaged_schedule
        memory = validate_packaged_schedule(model, config, device)
    elif config.get('sampling', {}).get('precompute_adaln', False):
        from ..inference.adaln import specialize_adaln
        memory = specialize_adaln(model, config, device)
    return model, memory
