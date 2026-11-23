"""Prepare complete model companions and publish reviewed model cards.

Only native conditioners, the Fast decoder, and small public entry files are
uploaded. Existing transformer and hand-encoder weights are preserved. Model
sources are explicit arguments; temporary staging stays outside source control.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
NATIVE_FOLDERS = ('vae', 'text_encoder', 'tokenizer', 'processor', 'scheduler')
DECODER_FILES = ('config.json', 'weights.safetensors', 'SHA256SUMS.json')
CARD_FORBIDDEN = re.compile(
    r'(?i)\b(?:live|ema)\d+\b|\b[A-Z]{2}\d{4,}\b|'
    r'author.selected|selected.*checkpoint|\b[0-9a-f]{40,64}\b')


def disable_upload_proxies():
    """Use direct Hub and artifact transfers instead of shell proxy settings."""
    for name in ('HTTP_PROXY', 'HTTPS_PROXY', 'ALL_PROXY',
                 'http_proxy', 'https_proxy', 'all_proxy'):
        os.environ.pop(name, None)


def validate_native(source):
    required = ('vae/config.json', 'text_encoder/config.json', 'tokenizer/vocab.json',
                'tokenizer/merges.txt', 'tokenizer/tokenizer_config.json',
                'processor/preprocessor_config.json', 'processor/tokenizer_config.json',
                'scheduler/scheduler_config.json')
    for name in required:
        if not (source/name).is_file():
            raise FileNotFoundError(f'Missing official component file: {name}')
    for folder in ('vae', 'text_encoder'):
        indexes = list((source/folder).glob('*.safetensors.index.json'))
        if not indexes:
            raise ValueError(f'Missing sharded tensor index: {folder}')
        for index in indexes:
            mapping = json.loads(index.read_text())['weight_map']
            if not mapping:
                raise ValueError(f'Empty tensor index: {folder}')
            for name in set(mapping.values()):
                if Path(name).is_absolute() or '..' in Path(name).parts:
                    raise ValueError('Tensor indexes must use relative package paths')
                if not (index.parent/name).is_file():
                    raise FileNotFoundError(f'Missing tensor shard in {folder}: {name}')


def companion_files(native_source, decoder_source, component):
    if component == 'vae':
        return {}
    validate_native(native_source)
    result = {}
    for folder in NATIVE_FOLDERS:
        for file in sorted((native_source/folder).rglob('*')):
            if (file.is_file() and not any(part in ('.cache', '.git', '__pycache__')
                                           for part in file.relative_to(native_source).parts)
                    and not file.name.endswith(('.lock', '.incomplete'))):
                result[str(file.relative_to(native_source))] = file.resolve()
    if component == 'fast':
        for name in DECODER_FILES:
            source = decoder_source/name
            if not source.is_file():
                raise FileNotFoundError(f'Missing Fast decoder file: {name}')
            result[f'decoder/{name}'] = source.resolve()
    return result


def prepare(component, native_source, decoder_source, cards_root, staging_root):
    from importlib.util import module_from_spec, spec_from_file_location
    spec = spec_from_file_location('bundle_config', ROOT/'src/lynnreal_hand/config.py')
    config_module = module_from_spec(spec)
    spec.loader.exec_module(config_module)
    files = companion_files(native_source, decoder_source, component)
    card = (cards_root/component/'README.md').read_text()
    if CARD_FORBIDDEN.search(card) or str(ROOT.parent) in card:
        raise ValueError(f'Model card contains internal release details: {component}')
    destination = staging_root/component
    destination.mkdir(parents=True, exist_ok=True)
    for name, source in files.items():
        target = destination/name
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.is_symlink():
            target.unlink()
        elif target.exists():
            raise FileExistsError(f'Use a fresh staging directory: {target.name}')
        target.symlink_to(source)
    (destination/'README.md').write_text(card)
    names = list(files) + ['README.md']
    if component != 'vae':
        config = json.loads((ROOT/f'configs/{component}.json').read_text())
        config.update(schema=config_module.INFERENCE_PACKAGE_SCHEMA, component=component)
        config['model']['checkpoint_path'] = '${component_dir}'
        config['conditioning']['checkpoint_path'] = '${component_dir}'
        if component == 'fast':
            config['hand_encoder'] = '${component_dir}/hand_encoder'
            config['decoder'] = '${component_dir}/decoder'
        (destination/'config.json').write_text(json.dumps(config, indent=2)+'\n')
        module, klass = config_module.INFERENCE_PIPELINES[component]
        index = dict(_class_name=klass, _module=module, _library_name='lynnreal-hand',
                     component=component, config='config.json',
                     transformer=['lynnreal_core.backends.minimax_h3.model', 'SolarMiniMaxH3Transformer3DModel'])
        (destination/'model_index.json').write_text(json.dumps(index, indent=2)+'\n')
        config_module.load_config(destination/'model_index.json')
        names += ['config.json', 'model_index.json']
    return destination, names


def upload_operations(component, destination, names, manifest, api):
    """Reuse matching Hub files server-side when the installed client supports it."""
    import inspect
    from huggingface_hub import CommitOperationAdd, CommitOperationCopy
    if 'src_repo_id' not in inspect.signature(CommitOperationCopy).parameters:
        return [CommitOperationAdd(path_in_repo=name, path_or_fileobj=destination/name)
                for name in names]
    sources = {}
    operations = []
    for name in names:
        source_component, source_name = None, None
        if any(name.startswith(folder+'/') for folder in NATIVE_FOLDERS):
            source_component, source_name = 'foundation', name
        elif component == 'fast' and name.startswith('decoder/'):
            source_component, source_name = 'vae', name.removeprefix('decoder/')
        if source_component is not None:
            source = manifest['components'][source_component]
            if source_component not in sources:
                info = api.model_info(source['repo_id'], revision=source['revision'], files_metadata=True)
                sources[source_component] = {file.rfilename: file for file in info.siblings}
            original = sources[source_component].get(source_name)
            if original is not None and original.size == (destination/name).stat().st_size:
                operations.append(CommitOperationCopy(
                    src_path_in_repo=source_name, path_in_repo=name,
                    src_revision=source['revision'], src_repo_id=source['repo_id'], src_repo_type='model'))
                continue
        operations.append(CommitOperationAdd(path_in_repo=name, path_or_fileobj=destination/name))
    return operations


def refresh_entry_checksums(checksums, destination, names):
    """Keep runtime entry-file checksums aligned with the published recipes."""
    result = dict(checksums)
    for name in ('config.json', 'model_index.json', 'README.md'):
        if name in names and (name != 'README.md' or name in result):
            result[name] = hashlib.sha256((destination/name).read_bytes()).hexdigest()
    return result


def publish(component, destination, names, manifest):
    disable_upload_proxies()
    from huggingface_hub import HfApi, hf_hub_download
    api = HfApi()
    entry = manifest['components'][component]
    repo = entry['repo_id']
    info = api.model_info(repo)
    existing = {file.rfilename for file in info.siblings}
    missing = set(entry['required']) - existing
    if missing:
        raise ValueError(f'The existing {component} model package is incomplete')
    if 'SHA256SUMS.json' in existing:
        prior = json.loads(Path(hf_hub_download(repo, 'SHA256SUMS.json', revision=info.sha)).read_text())
        updated = refresh_entry_checksums(prior, destination, names)
        (destination/'SHA256SUMS.json').write_text(json.dumps(updated, indent=2)+'\n')
        if 'SHA256SUMS.json' not in names:
            names.append('SHA256SUMS.json')
    result = api.create_commit(
        repo_id=repo, repo_type='model', parent_commit=info.sha,
        operations=upload_operations(component, destination, names, manifest, api),
        commit_message='Update model package', commit_description='')
    for name in ['README.md'] + ([] if component == 'vae' else ['config.json', 'model_index.json']):
        remote = Path(hf_hub_download(repo, name, revision=result.oid))
        if remote.read_bytes() != (destination/name).read_bytes():
            raise ValueError(f'Published entry verification failed: {component}/{name}')
    published = api.model_info(repo, revision=result.oid)
    if not set(names) <= {file.rfilename for file in published.siblings}:
        raise ValueError(f'Published component inventory is incomplete: {component}')
    return result.oid


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--native-source', type=Path, required=True)
    parser.add_argument('--decoder-source', type=Path, required=True)
    parser.add_argument('--cards-root', type=Path, default=ROOT/'checkpoints/hf_repos')
    parser.add_argument('--staging-root', type=Path, default=ROOT/'checkpoints/hf_repos/bundled')
    parser.add_argument('--components', nargs='+', choices=('base', 'fast', 'vae'), default=['base', 'fast', 'vae'])
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--prepare', action='store_true')
    mode.add_argument('--publish', action='store_true')
    args = parser.parse_args()
    manifest_path = ROOT/'checkpoints/manifest.template.json'
    manifest = json.loads(manifest_path.read_text())
    for component in dict.fromkeys(args.components):
        destination, names = prepare(component, args.native_source, args.decoder_source,
                                     args.cards_root, args.staging_root)
        report = dict(component=component, files=len(names),
                      companion_bytes=sum((destination/name).stat().st_size for name in names),
                      published=False)
        if args.publish:
            revision = publish(component, destination, names, manifest)
            manifest['components'][component]['revision'] = revision
            if component in ('base', 'fast'):
                manifest['components'][component]['required'] += [name for name in (
                    'vae/config.json', 'text_encoder/config.json',
                    'tokenizer/tokenizer_config.json', 'processor/preprocessor_config.json',
                    'scheduler/scheduler_config.json')
                    if name not in manifest['components'][component]['required']]
            if component == 'fast':
                manifest['components']['hand_encoder']['revision'] = revision
                manifest['components']['fast']['required'] += [f'decoder/{name}' for name in DECODER_FILES
                    if f'decoder/{name}' not in manifest['components']['fast']['required']]
            manifest_path.write_text(json.dumps(manifest, indent=2)+'\n')
            report['published'] = True
        print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
