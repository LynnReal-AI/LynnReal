"""Portable configuration with explicit model and output locations."""
from pathlib import Path
import json
from string import Template

INFERENCE_PACKAGE_SCHEMA = 'lynnreal-hand.inference.v1'
INFERENCE_PIPELINES = {
    'base': ('lynnreal_hand.inference.base', 'BasePipeline'),
    'fast': ('lynnreal_hand.inference.fast', 'FastPipeline'),
}


def validate_model_index(index, component):
    expected = INFERENCE_PIPELINES.get(component)
    if (expected is None or index.get('_library_name') != 'lynnreal-hand' or
            index.get('config') != 'config.json' or index.get('component') != component or
            (index.get('_module'), index.get('_class_name')) != expected):
        raise ValueError('The model index does not match the inference component')


def load_config(path, *, weights_root=None, output_dir=None):
    path = Path(path).resolve()
    document = json.loads(path.read_text())
    index = None
    if path.name == 'model_index.json':
        index = document
        validate_model_index(index, index.get('component'))
        path = path.parent/index['config']
        document = json.loads(path.read_text())
    root = path.parent.parent
    packaged = document.get('schema') == INFERENCE_PACKAGE_SCHEMA
    if index is not None and not packaged:
        raise ValueError('The model index must reference an inference package configuration')
    if packaged:
        index_path = path.parent/'model_index.json'
        if index is not None or index_path.is_file():
            index = index if index is not None else json.loads(index_path.read_text())
            validate_model_index(index, document.get('component'))
    default_weights = root if packaged else root/'checkpoints'
    variables = {'weights_root': str(Path(weights_root or default_weights).resolve()),
                 'component_dir': str(path.parent),
                 'repo_root': str(root), 'output_dir': str(Path(output_dir or root/'outputs').resolve())}
    def resolve(value):
        if isinstance(value, dict): return {k: resolve(v) for k, v in value.items()}
        if isinstance(value, list): return [resolve(v) for v in value]
        if isinstance(value, str): return Template(value).substitute(variables)
        return value
    config = resolve(document)
    # A complete model snapshot carries its own official conditioners. Preserve
    # explicit custom paths and support older snapshots with shared components.
    component_dir = Path(config['model']['checkpoint_path'])
    shared = Path(variables['weights_root'])/'foundation'
    native_entries = ('vae/config.json', 'text_encoder/config.json',
                      'tokenizer/tokenizer_config.json', 'processor/preprocessor_config.json')
    if (Path(config['conditioning']['checkpoint_path']) == shared and
            all((component_dir/name).is_file() for name in native_entries)):
        config['conditioning']['checkpoint_path'] = str(component_dir)
    if (config.get('decoder') == str(Path(variables['weights_root'])/'vae') and
            all((component_dir/'decoder'/name).is_file()
                for name in ('config.json', 'weights.safetensors', 'SHA256SUMS.json'))):
        config['decoder'] = str(component_dir/'decoder')
    config['_config_path'] = str(path)
    validate_config(config)
    return config


def validate_config(config):
    if not isinstance(config.get('sampling', {}).get('precompute_adaln', False), bool):
        raise ValueError('precompute_adaln must be a boolean')
    if 'schema' in config:
        if config['schema'] != INFERENCE_PACKAGE_SCHEMA:
            raise ValueError('Unsupported inference package schema')
        stage = {'base': 'stage1', 'fast': 'stage3'}.get(config.get('component'))
        if stage is None or config.get('stage') != stage:
            raise ValueError('The package component and inference stage do not match')
    if config['stage'] not in ('stage1', 'stage3'):
        raise ValueError('stage must be stage1 or stage3')
    if config['model'].get('attention_backend') != '_flash_3':
        raise ValueError('H3 execution requires explicit _flash_3; fallback is disabled')
    h, w = config['video']['height'], config['video']['width']
    if w % 32 or (h if h % 16 == 0 else h+4) % 32:
        raise ValueError('The native H3 canvas must have even latent spatial dimensions')
    if config['stage'] == 'stage3':
        if config['sampling']['steps'] != 3 or config['sampling']['window'] != 2:
            raise ValueError('The released Fast contract is 3-step, W2')
        if (h, w) not in ((540, 960), (768, 1344)):
            raise ValueError('Fast supports 960x540 or native 1344x768 output')
    if any(key in config for key in ('training','losses','image_branch')):
        raise ValueError('This distribution provides inference only; training recipes are not included')
    return config


def canvas(config):
    h, w = config['video']['height'], config['video']['width']
    return (544 if h == 540 else h), w
