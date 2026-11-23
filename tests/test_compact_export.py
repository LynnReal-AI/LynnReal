"""A compact checkpoint strict-loads without restoring learned AdaLN projections."""
import copy
import importlib.util
import json
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize('component', ['base', 'fast'])
def test_compact_export_strict_roundtrip_and_schedule_values(tmp_path, component):
    from lynnreal_hand.backends.h3.runtime import activate
    activate()
    from lynnreal_core.backends.minimax_h3.model import SolarMiniMaxH3Transformer3DModel as Model
    from lynnreal_hand.inference.adaln import schedule_rows, select_adaln, validate_packaged_schedule
    spec = importlib.util.spec_from_file_location('compact_export', ROOT/'scripts/export_compact.py')
    exporter = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(exporter)
    architecture = dict(num_attention_heads=2, attention_head_dim=16, hidden_size=32,
        num_layers=2, num_refiner_layers=1, ffn_dim=64, in_channels=4,
        audio_in_channels=4, text_dim=16, freq_dim=8,
        time_embed_hidden_dim=32, time_embed_dim=16)
    if component=='fast':
        architecture.update(lynnreal_variant='fast', anyflow_gate=.25)
    model = Model.from_config(architecture).to(torch.bfloat16)
    for name in ['proj_in', 'audio_proj_in', 'time_embedder', 'proj_out', 'audio_proj_out']:
        getattr(model, name).float()
    if component=='fast':model.delta_embedding.float()
    model.eval().requires_grad_(False)
    source, target = tmp_path/'source', tmp_path/'compact'
    model.save_pretrained(source/'transformer', safe_serialization=True, max_shard_size='20KB')
    config = json.loads((ROOT/f'configs/{component}.json').read_text())
    config_path = tmp_path/'config.json'
    config_path.write_text(json.dumps(config))
    exporter.repack_weights(source, target, max_shard_bytes=25000)
    exporter.export_tables(source, config_path, target, 'cpu')
    index = json.loads((target/'transformer/diffusion_pytorch_model.safetensors.index.json').read_text())
    assert not any('.adaln_proj.linear.' in name for name in index['weight_map'])
    restored = Model.strict_from_pretrained(target/'transformer', torch_dtype=torch.bfloat16,
                                           local_files_only=True, low_cpu_mem_usage=True)
    restored.eval().requires_grad_(False)
    validate_packaged_schedule(restored, config, 'cpu')
    assert sum(p.numel() for p in restored.parameters()) < sum(p.numel() for p in model.parameters())
    with torch.no_grad():
        for times, targets in schedule_rows(config, 'cpu'):
            t, r = torch.tensor(times), torch.tensor(targets) if targets is not None else None
            select_adaln(restored, t, r)
            embedding = model._time_condition(t, r, parameter_dtype=lambda m: next(m.parameters()).dtype)
            for original, compact in zip(model.transformer_blocks, restored.transformer_blocks):
                for expected, actual in zip(original.adaln_proj(embedding), compact.adaln_proj(embedding)):
                    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    if component=='base':
        changed = copy.deepcopy(config)
        changed['sampling']['schedule_points'] += 1
        with pytest.raises(ValueError, match='sampling schedule differs'):
            validate_packaged_schedule(restored, changed, 'cpu')
