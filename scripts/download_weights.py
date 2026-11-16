"""Download a pinned component and validate its real inference configuration."""
from pathlib import Path
import argparse
import json
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))


def download_component(component, repo_id, revision, destination, *, allow_patterns=None):
    from huggingface_hub import hf_hub_download, snapshot_download
    from huggingface_hub.errors import EntryNotFoundError
    from lynnreal_hand import __version__
    from lynnreal_hand.config import INFERENCE_PACKAGE_SCHEMA, load_config, validate_model_index

    options = dict(repo_id=repo_id, revision=revision, local_dir=destination,
                   library_name='lynnreal-hand', library_version=__version__, token=False)
    inference_config = None
    model_index = None
    if component in ('base', 'fast'):
        try:
            model_index = Path(hf_hub_download(filename='model_index.json', **options))
        except EntryNotFoundError:
            pass
        else:
            validate_model_index(json.loads(model_index.read_text()), component)
        try:
            path = Path(hf_hub_download(filename='config.json', **options))
        except EntryNotFoundError:
            # Releases predating the root inference configuration remain usable.
            if model_index is not None:
                raise ValueError('The indexed package is missing its inference configuration')
        else:
            document = json.loads(path.read_text())
            if document.get('schema') != INFERENCE_PACKAGE_SCHEMA:
                raise ValueError('The repository does not contain a LynnReal-Hand inference package')
            config = load_config(model_index or path)
            if config['component'] != component:
                raise ValueError('The requested component differs from the downloaded configuration')
            inference_config = path
    # The root configuration has already been fetched and checked. Download it
    # once; the nested transformer/config.json remains part of the snapshot.
    already_fetched = []
    if inference_config:
        already_fetched.append('config.json')
    if model_index:
        already_fetched.append('model_index.json')
    snapshot_download(**options, allow_patterns=allow_patterns, ignore_patterns=already_fetched or None)
    return inference_config


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, default=ROOT/'checkpoints/manifest.template.json')
    parser.add_argument('--component', required=True)
    parser.add_argument('--repo-id')
    parser.add_argument('--revision', help='Defaults to the immutable revision in the release manifest')
    parser.add_argument('--weights-root', type=Path, help='Alternative to the default checkpoints directory')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args(argv)
    manifest_path = args.manifest.resolve()
    components = json.loads(manifest_path.read_text())['components']
    if args.component not in components:
        parser.error(f'Unknown component: {args.component}')
    item = components[args.component]
    repo_id = args.repo_id or item['repo_id']
    revision = args.revision or item.get('revision')
    if repo_id.startswith('YOUR_ORG/') or not re.fullmatch(r'[0-9a-fA-F]{40}', revision or ''):
        parser.error('Set a published repository ID and immutable revision')
    destination = Path(item['local_dir'])
    if args.weights_root:
        destination = args.weights_root.resolve()/destination.relative_to('checkpoints')
    if not destination.is_absolute():
        destination = manifest_path.parent.parent/destination
    print(json.dumps(dict(repo_id=repo_id, revision=revision, local_dir=str(destination),
                          component=args.component, allow_patterns=item.get('allow_patterns'),
                          authentication_required=False, upload=False), indent=2))
    if not args.dry_run:
        path = download_component(args.component, repo_id, revision, destination,
                                  allow_patterns=item.get('allow_patterns'))
        print(json.dumps(dict(downloaded=True, inference_config=str(path) if path else None), indent=2))


if __name__ == '__main__':
    main()
