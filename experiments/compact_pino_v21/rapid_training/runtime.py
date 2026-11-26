"""Estimate a long-run budget from real runner timings, not the GPU model name."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .prepare import read_json
from ..training.runner import _json_write


def estimate(run_directory, *, total_steps=10000, warmup=3, protocol=None):
    root = Path(run_directory).resolve(strict=True)
    rows = [json.loads(line) for line in (root/'steps.jsonl').read_text().splitlines() if line.strip()]
    committed = [row for row in rows if row['committed']]
    if len(committed) <= warmup:
        raise ValueError('need committed updates beyond the declared warmup')
    values = np.asarray([row['seconds'] for row in committed[warmup:]], dtype=float)
    if not np.isfinite(values).all() or np.any(values <= 0):
        raise ValueError('positive finite measured step timings required')
    evaluations = [read_json(path) for path in sorted(root.glob('development-step-*.json'))]
    evaluation_times = [item['evaluation_seconds'] for item in evaluations
                        if isinstance(item.get('evaluation_seconds'), (float, int))]
    current = committed[-1]['global_step']
    if total_steps < current:
        raise ValueError('target steps precede the observed run')
    eval_count = None
    if protocol:
        eval_count = sum(current < step <= total_steps for step in read_json(protocol)['evaluation_steps'])
    median_evaluation = float(np.median(evaluation_times)) if evaluation_times else None
    remaining = total_steps-current
    p50, p95, p99 = (float(np.percentile(values, q)) for q in (50, 95, 99))
    peaks = [row['cuda_memory_bytes']['max_allocated'] for row in rows
             if isinstance(row.get('cuda_memory_bytes'), dict)]
    evaluation_projection = (eval_count*median_evaluation
                             if eval_count is not None and median_evaluation is not None else None)
    return dict(schema='compact_pino.rapid-runtime-projection.v1', status='MEASURED_PROJECTION_NOT_SPEEDUP',
        observed_committed_updates=len(committed), discarded_warmup=warmup, measured_samples=len(values),
        skipped_attempts=len(rows)-len(committed), current_step=current, target_step=total_steps,
        step_seconds=dict(p50=p50, p95=p95, p99=p99),
        measured_evaluation_passes=len(evaluation_times), evaluation_median_seconds=median_evaluation,
        remaining_scheduled_evaluations=eval_count,
        remaining_update_hours=dict(p50=remaining*p50/3600, p95=remaining*p95/3600),
        estimated_remaining_evaluation_hours=None if evaluation_projection is None else evaluation_projection/3600,
        peak_measured_cuda_allocated_bytes=max(peaks) if peaks else None,
        limitations=['descriptive timing quantiles, not confidence bounds',
                     'future data-dependent query costs and failures may differ',
                     'dataset loading and independent native solver validation not included',
                     'evaluation costs unavailable if no evaluation_seconds recorded'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--total-steps', type=int, default=10000)
    parser.add_argument('--warmup', type=int, default=3)
    parser.add_argument('--protocol')
    args = parser.parse_args()
    result = estimate(args.run, total_steps=args.total_steps, warmup=args.warmup, protocol=args.protocol)
    _json_write(args.output, result)
    print(json.dumps(result, indent=2))


if __name__ == '__main__': main()
