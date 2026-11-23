"""Portable package configuration and cached, pinned Hub downloads."""
from pathlib import Path
import importlib.util
import json

import pytest
import huggingface_hub
from huggingface_hub.errors import EntryNotFoundError

from lynnreal_hand.config import INFERENCE_PACKAGE_SCHEMA, INFERENCE_PIPELINES, load_config

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('download_weights', ROOT/'scripts/download_weights.py')
downloader = importlib.util.module_from_spec(spec)
spec.loader.exec_module(downloader)


def package_config(directory, component='fast'):
    directory.mkdir(parents=True, exist_ok=True)
    value = json.loads((ROOT/f'configs/{component}.json').read_text())
    value.update(schema=INFERENCE_PACKAGE_SCHEMA, component=component)
    value['model']['checkpoint_path'] = '${component_dir}'
    path = directory/'config.json'
    path.write_text(json.dumps(value))
    module, klass = INFERENCE_PIPELINES[component]
    (directory/'model_index.json').write_text(json.dumps(dict(_class_name=klass,
        _module=module, _library_name='lynnreal-hand', component=component, config='config.json')))
    return path


@pytest.mark.parametrize('component', ['base', 'fast'])
def test_package_configuration_survives_an_arbitrary_checkout_directory(tmp_path, component):
    directory = tmp_path/'weights'/'arbitrary-checkout-name'
    path = package_config(directory, component)
    config = load_config(path)
    assert config['model']['checkpoint_path'] == str(directory)
    assert config['conditioning']['checkpoint_path'] == str(directory.parent/'foundation')
    relocated = load_config(path, weights_root=tmp_path/'shared-components')
    assert relocated['model']['checkpoint_path'] == str(directory)
    assert relocated['conditioning']['checkpoint_path'] == str(tmp_path/'shared-components/foundation')
    if component == 'fast':
        assert config['decoder'] == str(directory.parent/'vae')
        assert config['sampling']['steps'] == 3
        assert config['sampling']['window'] == 2


def test_download_validates_configuration_before_large_weights_without_fetching_it_twice(tmp_path, monkeypatch):
    path = package_config(tmp_path/'fast')
    calls = []
    def config_download(**kw):
        calls.append(('index' if kw['filename']=='model_index.json' else 'configuration', kw))
        return str(path.parent/kw['filename'])
    def snapshot(**kw):
        calls.append(('weights', kw))
        return str(path.parent)
    monkeypatch.setattr(huggingface_hub, 'hf_hub_download', config_download)
    monkeypatch.setattr(huggingface_hub, 'snapshot_download', snapshot)
    result = downloader.download_component('fast', 'owner/fast', 'a'*40, path.parent)
    assert result == path
    assert [kind for kind, kw in calls] == ['index', 'configuration', 'weights']
    assert calls[1][1]['filename'] == 'config.json'
    assert calls[2][1]['ignore_patterns'] == ['config.json', 'model_index.json']
    assert all(kw['revision'] == 'a'*40 and not kw.get('force_download') for _, kw in calls)


def test_wrong_component_stops_before_downloading_weights(tmp_path, monkeypatch):
    path = package_config(tmp_path/'fast')
    monkeypatch.setattr(huggingface_hub, 'hf_hub_download', lambda **kw: str(path.parent/kw['filename']))
    def unexpected(**kw):
        pytest.fail('The 33B weights must not download after a component mismatch')
    monkeypatch.setattr(huggingface_hub, 'snapshot_download', unexpected)
    with pytest.raises(ValueError, match='inference component'):
        downloader.download_component('base', 'owner/fast', 'a'*40, path.parent)


@pytest.mark.parametrize('component', ['base', 'fast'])
def test_runtime_index_is_a_usable_entry_and_must_match_its_recipe(tmp_path, component):
    path = package_config(tmp_path/component, component)
    index_path = path.parent/'model_index.json'
    config = load_config(index_path)
    assert config['component'] == component
    assert config['_config_path'] == str(path)
    assert config['model']['checkpoint_path'] == str(path.parent)
    index = json.loads(index_path.read_text())
    index['_class_name'] = 'WrongPipeline'
    index_path.write_text(json.dumps(index))
    with pytest.raises(ValueError, match='model index'):
        load_config(path)


def test_legacy_revision_uses_original_snapshot_layout(tmp_path, monkeypatch):
    def missing(**kw):
        raise EntryNotFoundError('Legacy release has no root inference config')
    calls = []
    monkeypatch.setattr(huggingface_hub, 'hf_hub_download', missing)
    monkeypatch.setattr(huggingface_hub, 'snapshot_download', lambda **kw: calls.append(kw))
    assert downloader.download_component('base', 'owner/base', 'b'*40, tmp_path) is None
    assert calls[0]['ignore_patterns'] is None


def test_dry_run_uses_manifest_revision_and_performs_no_download(tmp_path, monkeypatch, capsys):
    path = tmp_path/'checkpoints/manifest.json'
    path.parent.mkdir()
    path.write_text(json.dumps(dict(components=dict(fast=dict(repo_id='owner/fast',
        revision='c'*40, local_dir='checkpoints/fast')))))
    def unexpected(*args, **kwargs):
        pytest.fail('A dry run must not contact the Hub')
    monkeypatch.setattr(downloader, 'download_component', unexpected)
    monkeypatch.chdir(tmp_path.parent)
    downloader.main(['--manifest', str(path), '--component', 'fast', '--dry-run'])
    plan = json.loads(capsys.readouterr().out)
    assert plan['revision'] == 'c'*40
    assert plan['local_dir'] == str(tmp_path/'checkpoints/fast')
