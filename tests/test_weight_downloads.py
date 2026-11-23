"""Public download plans must select complete Fast assets and scoped conditioners."""
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def plan(component, *extra):
    result = subprocess.run([sys.executable, 'scripts/download_weights.py',
        '--component', component, '--dry-run', *extra], cwd=ROOT,
        text=True, capture_output=True, check=True)
    return json.loads(result.stdout)


def test_fast_and_encoder_share_the_published_snapshot(tmp_path):
    fast = plan('fast', '--weights-root', str(tmp_path))
    encoder = plan('hand_encoder', '--weights-root', str(tmp_path))
    assert fast['repo_id'] == encoder['repo_id'] == 'shaohao011/LynnReal-Hand-Fast-preview'
    assert fast['revision'] == encoder['revision']
    assert fast['local_dir'] == encoder['local_dir'] == str(tmp_path/'fast')
    assert encoder['allow_patterns'] == ['hand_encoder/**']


def test_foundation_download_excludes_original_transformers():
    foundation = plan('foundation')
    assert foundation['repo_id'] == 'MiniMaxAI/MiniMax-H3'
    assert set(foundation['allow_patterns']) == {'vae/**','text_encoder/**','tokenizer/**','processor/**'}
    assert foundation['authentication_required'] is False


def test_bad_revision_fails_before_network():
    result = subprocess.run([sys.executable, 'scripts/download_weights.py',
        '--component', 'fast', '--revision', 'main', '--dry-run'],
        cwd=ROOT, text=True, capture_output=True)
    assert result.returncode == 2 and 'immutable' in result.stderr
