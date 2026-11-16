"""Frame-preserving video IO; no frame repetition or time interpolation."""
from pathlib import Path
import json
import subprocess
import numpy as np


def probe(path):
    value = json.loads(subprocess.check_output(['ffprobe', '-v', 'error', '-count_frames',
        '-select_streams', 'v:0', '-show_entries', 'stream=width,height,avg_frame_rate,nb_read_frames',
        '-of', 'json', str(path)]))['streams'][0]
    a, b = map(int, value['avg_frame_rate'].split('/'))
    return dict(width=int(value['width']), height=int(value['height']),
                frames=int(value['nb_read_frames']), fps=a/b)


def read_video(path, *, frames, height, width):
    # Match the source field of view before adapting its spatial canvas.
    vf = f'scale=1344:768:force_original_aspect_ratio=increase,crop=1344:768,scale={width}:{height}:flags=lanczos'
    raw = subprocess.check_output(['ffmpeg', '-v', 'error', '-threads', '2', '-filter_threads', '1',
        '-i', str(path), '-vf', vf, '-frames:v', str(frames), '-vsync', '0',
        '-pix_fmt', 'rgb24', '-f', 'rawvideo', 'pipe:1'])
    expected = frames*height*width*3
    if len(raw) != expected:
        raise ValueError(f'{path}: requested {frames} genuine frames, received {len(raw)//(height*width*3)}')
    return np.frombuffer(raw, np.uint8).reshape(frames, height, width, 3).copy()


def write_video(frames, path, fps):
    if hasattr(frames, 'detach'): frames = frames.detach().cpu().numpy()
    frames = np.asarray(frames)
    if frames.dtype != np.uint8:
        frames = (frames.clip(0, 1)*255).round().astype(np.uint8)
    if frames.ndim != 4 or frames.shape[-1] != 3:
        raise ValueError('Expected [frames,height,width,3] images')
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    h, w = frames.shape[1:3]
    p = subprocess.Popen(['ffmpeg', '-v', 'error', '-y', '-threads', '2', '-f', 'rawvideo',
        '-pix_fmt', 'rgb24', '-s', f'{w}x{h}', '-r', str(fps), '-i', 'pipe:0', '-an',
        '-c:v', 'libx264', '-crf', '18', '-pix_fmt', 'yuv420p', str(path)], stdin=subprocess.PIPE)
    try: p.stdin.write(frames.tobytes())
    finally: p.stdin.close()
    if p.wait(): raise RuntimeError(f'Video encoding failed: {path}')
    return path
