"""Phase-aligned Base windows on the native H3 video timeline."""
from lynnreal_core.backends.minimax_h3.geometry import DEFAULT_GEOMETRY, latent_aligned_pixel_indices


WINDOW_FRAMES = 240
HISTORY_LATENTS = DEFAULT_GEOMETRY.latent_frames_per_chunk
CHUNK_FRAMES = DEFAULT_GEOMETRY.pixel_frames_per_chunk
# The 240-frame training span has 72 latents, including its two-latent tail.
ROPE_GAP = 405.0
WINDOW_LATENTS = len(latent_aligned_pixel_indices(WINDOW_FRAMES, allow_partial=True))


def window_spans(frames):
    """Return (start, retained end) in RGB frames for each full input window."""
    if frames <= WINDOW_FRAMES:
        return [(0, frames)]
    last_start = (frames - WINDOW_FRAMES + CHUNK_FRAMES - 1) // CHUNK_FRAMES
    max_stride = (WINDOW_FRAMES - CHUNK_FRAMES) // CHUNK_FRAMES
    count = (last_start + max_stride - 1) // max_stride
    starts = [(i * last_start // count) * CHUNK_FRAMES for i in range(count + 1)]
    ends = [start + CHUNK_FRAMES for start in starts[1:]] + [frames]
    return list(zip(starts, ends, strict=True))
