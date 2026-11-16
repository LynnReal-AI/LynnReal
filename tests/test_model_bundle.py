"""Companion packaging includes tokenizer assets without touching transformers."""
import importlib.util
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('publish_hf_bundle', ROOT/'scripts/publish_hf_bundle.py')
bundler = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bundler)


def sources(tmp_path):
    native = tmp_path/'official'
    entries = ['vae/config.json', 'text_encoder/config.json', 'tokenizer/vocab.json',
               'tokenizer/merges.txt', 'tokenizer/tokenizer_config.json',
               'processor/preprocessor_config.json', 'processor/tokenizer_config.json',
               'processor/chat_template.json', 'scheduler/scheduler_config.json']
    for name in entries:
        path = native/name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('{}')
    for name in ['vae', 'text_encoder']:
        (native/name/'model.safetensors').write_bytes(b'weight fixture')
        (native/name/'model.safetensors.index.json').write_text(json.dumps(
            dict(weight_map={'tensor': 'model.safetensors'})))
    decoder = tmp_path/'decoder'
    decoder.mkdir()
    for name in bundler.DECODER_FILES:
        (decoder/name).write_bytes(b'{}')
    cards = tmp_path/'cards'
    for name in ['base', 'fast', 'vae']:
        (cards/name).mkdir(parents=True)
        (cards/name/'README.md').write_text('# Public model\n')
    return native, decoder, cards


@pytest.mark.parametrize('component', ['base', 'fast'])
def test_bundle_is_self_contained_and_excludes_existing_generation_weights(tmp_path, component):
    native, decoder, cards = sources(tmp_path)
    destination, names = bundler.prepare(component, native, decoder, cards, tmp_path/'staging')
    assert 'tokenizer/vocab.json' in names and 'processor/chat_template.json' in names
    assert 'scheduler/scheduler_config.json' in names
    assert not any(name.startswith(('transformer/', 'hand_encoder/')) for name in names)
    for name in ['vae/model.safetensors', 'text_encoder/model.safetensors']:
        assert (destination/name).is_symlink()
        assert (destination/name).read_bytes() == (native/name).read_bytes()
    from lynnreal_hand.config import load_config
    config = load_config(destination/'model_index.json')
    assert config['conditioning']['checkpoint_path'] == str(destination)
    if component == 'fast':
        assert config['decoder'] == str(destination/'decoder')
        assert config['hand_encoder'] == str(destination/'hand_encoder')
        assert all(f'decoder/{name}' in names for name in bundler.DECODER_FILES)
    else:
        assert not any(name.startswith('decoder/') for name in names)


def test_incomplete_official_weights_are_rejected_before_preparing_upload(tmp_path):
    native, decoder, _ = sources(tmp_path)
    (native/'text_encoder/model.safetensors').unlink()
    with pytest.raises(FileNotFoundError, match='Missing tensor shard'):
        bundler.companion_files(native, decoder, 'base')


def test_uploads_clear_proxy_settings_without_changing_internal_bypasses(monkeypatch):
    keys = ('HTTP_PROXY', 'HTTPS_PROXY', 'ALL_PROXY', 'http_proxy', 'https_proxy', 'all_proxy')
    for name in keys:
        monkeypatch.setenv(name, 'http://proxy.invalid')
    monkeypatch.setenv('NO_PROXY', 'localhost,.internal')
    bundler.disable_upload_proxies()
    assert not any(name in os.environ for name in keys)
    assert os.environ['NO_PROXY'] == 'localhost,.internal'


def test_native_and_decoder_tensors_are_copied_from_pinned_hub_sources(tmp_path):
    hub = pytest.importorskip('huggingface_hub')
    import inspect
    if 'src_repo_id' not in inspect.signature(hub.CommitOperationCopy).parameters:
        pytest.skip('Cross-repository copying needs a newer Hub client')
    native, decoder, cards = sources(tmp_path)
    destination, names = bundler.prepare('fast', native, decoder, cards, tmp_path/'staging')
    manifest = dict(components=dict(
        foundation=dict(repo_id='owner/official', revision='a'*40),
        vae=dict(repo_id='owner/decoder', revision='b'*40)))
    class API:
        def model_info(self, repo, **kwargs):
            source = native if repo=='owner/official' else decoder
            assert kwargs['revision'] == ('a' if source==native else 'b')*40
            return SimpleNamespace(siblings=[SimpleNamespace(rfilename=str(p.relative_to(source)),
                size=p.stat().st_size) for p in source.rglob('*') if p.is_file()])
    operations = bundler.upload_operations('fast', destination, names, manifest, API())
    tensors = [op for op in operations if op.path_in_repo.endswith('.safetensors')]
    assert tensors and all(isinstance(op, hub.CommitOperationCopy) for op in tensors)
    decoder_copy = next(op for op in operations if op.path_in_repo=='decoder/weights.safetensors')
    assert decoder_copy.src_repo_id=='owner/decoder'
    assert decoder_copy.src_path_in_repo=='weights.safetensors'
    assert {op.path_in_repo for op in operations} == set(names)


def test_new_recipes_refresh_integrity_without_rehashing_unchanged_tensor_files(tmp_path):
    import hashlib
    (tmp_path/'config.json').write_text('{"conditioning":"local"}')
    (tmp_path/'model_index.json').write_text('{"component":"fast"}')
    (tmp_path/'README.md').write_text('# Public model')
    prior = {'config.json': 'old', 'transformer/weights.safetensors': 'preserved'}
    updated = bundler.refresh_entry_checksums(prior, tmp_path,
        ['config.json', 'model_index.json', 'README.md'])
    assert updated['config.json'] == hashlib.sha256((tmp_path/'config.json').read_bytes()).hexdigest()
    assert updated['model_index.json'] == hashlib.sha256((tmp_path/'model_index.json').read_bytes()).hexdigest()
    assert updated['transformer/weights.safetensors'] == 'preserved'
    assert 'README.md' not in updated
    assert prior['config.json'] == 'old'
