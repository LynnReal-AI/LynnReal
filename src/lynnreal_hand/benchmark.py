"""Warm CPU-RGB-ready response and true successive-delivery throughput."""
import numpy as np


def summarize(records):
    rows = [row for row in records if not row['warmup'] and row['chunk'] >= 2
            and row['latent_frames'] == 5]
    if not rows:
        raise ValueError('No formal steady full chunks were measured')
    if any(row['RGB_frames'] != 17 or row['mtp_ms'] <= 0 or row['output_interval_ms'] <= 0 for row in rows):
        raise ValueError('Invalid full-chunk timing record')
    values = [row['mtp_ms'] for row in rows]
    return dict(measured_chunks=len(rows), emitted_frames=sum(row['RGB_frames'] for row in rows),
        MTP_mean_ms=float(np.mean(values)), MTP_median_ms=float(np.median(values)),
        MTP_p95_ms=float(np.percentile(values,95)), minimum_observed_ms=min(values),
        continuous_FPS=1000*sum(row['RGB_frames'] for row in rows)/sum(row['output_interval_ms'] for row in rows),
        mean_output_interval_ms=float(np.mean([row['output_interval_ms'] for row in rows])),
        mean_commit_after_RGB_ms=float(np.mean([row['commit_after_RGB_ms'] for row in rows])),
        endpoint='synchronized pinned CPU uint8 RGB-ready; current commit follows ready',
        steady_filter='formal, chunk>=2, 5 native latent frames / 17 RGB frames')
