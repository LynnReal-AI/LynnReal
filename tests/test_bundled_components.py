"""Complete snapshots resolve companions locally and preserve custom paths."""
from pathlib import Path

import pytest

from lynnreal_hand.config import load_config

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize('component', ['base', 'fast'])
def test_complete_components_are_preferred_over_legacy_shared_paths(tmp_path, component):
    directory = tmp_path/component
    for name in ['vae/config.json', 'text_encoder/config.json',
                 'tokenizer/tokenizer_config.json', 'processor/preprocessor_config.json']:
        file = directory/name
        file.parent.mkdir(parents=True, exist_ok=True)
        file.write_text('{}')
    for name in ['config.json', 'weights.safetensors', 'SHA256SUMS.json']:
        file = directory/'decoder'/name
        file.parent.mkdir(exist_ok=True)
        file.write_text('{}')
    config = load_config(ROOT/f'configs/{component}.json', weights_root=tmp_path)
    assert config['conditioning']['checkpoint_path'] == str(directory)
    if component == 'fast':
        assert config['decoder'] == str(directory/'decoder')
        assert config['hand_encoder'] == str(directory/'hand_encoder')


def test_incomplete_snapshot_keeps_the_legacy_conditioner_path(tmp_path):
    (tmp_path/'base/vae').mkdir(parents=True)
    config = load_config(ROOT/'configs/base.json', weights_root=tmp_path)
    assert config['conditioning']['checkpoint_path'] == str(tmp_path/'foundation')
