"""Whole-recording metrics measure removed/modified vocals and useful trimming."""

import time
from dataclasses import asdict, replace

import numpy as np

from .cuts import plan_cuts
from .data import intersection_length
from .engine import read_audio


def complement(intervals, duration):
    result, cursor = [], 0.0
    for start, end in intervals:
        if cursor < start:
            result.append((cursor, start))
        cursor = end
    if cursor < duration:
        result.append((cursor, duration))
    return result


def score_plan(plan, record):
    sr = plan.sample_rate
    duration = plan.source_samples / sr
    removed = [(c.start / sr, c.end / sr) for c in plan.cuts]
    modified = [((c.start - c.fade_samples) / sr, (c.end + c.fade_samples) / sr) for c in plan.cuts]
    protected = complement(record.remove, duration)
    explicit = [(e["start"], e["end"]) for e in record.protected]
    events = explicit or protected
    eligible_removed = sum(intersection_length(a, b) for a in removed for b in record.remove)
    protected_removed = sum(intersection_length(a, b) for a in removed for b in protected)
    protected_modified = sum(intersection_length(a, b) for a in modified for b in protected)
    damaged_events = sum(
        any(intersection_length(event, change) > 1e-9 for change in modified) for event in events
    )
    # Evaluate actual cut edges against the annotated safe region, separately
    # from the intentionally retained pause/padding inside that region.
    errors, matched = [], set()
    for cut in plan.cuts:
        approved = (
            (cut.detected_start if cut.detected_start is not None else cut.approved_start) / sr,
            (cut.detected_end if cut.detected_end is not None else cut.approved_end) / sr,
        )
        candidates = [
            (intersection_length(approved, gap), i)
            for i, gap in enumerate(record.remove)
            if i not in matched
        ]
        if candidates:
            overlap, index = max(candidates)
            if overlap > 0:
                matched.add(index)
                gap = record.remove[index]
                errors.extend([abs(approved[0] - gap[0]), abs(approved[1] - gap[1])])
    return {
        "id": record.id,
        "audio_seconds": duration,
        "eligible_seconds": sum(e - s for s, e in record.remove),
        "eligible_removed_seconds": eligible_removed,
        "protected_removed_seconds": protected_removed,
        "protected_modified_seconds": protected_modified,
        "protected_events": len(events),
        "damaged_protected_events": damaged_events,
        "cuts": len(plan.cuts),
        "annotated_gaps": len(record.remove),
        "matched_gaps": len(matched),
        "boundary_errors_seconds": errors,
    }


def aggregate(rows):
    numeric = (
        "audio_seconds",
        "eligible_seconds",
        "eligible_removed_seconds",
        "protected_removed_seconds",
        "protected_modified_seconds",
        "protected_events",
        "damaged_protected_events",
        "cuts",
        "annotated_gaps",
        "matched_gaps",
    )
    totals = {key: sum(row[key] for row in rows) for key in numeric}
    errors = [value for row in rows for value in row["boundary_errors_seconds"]]
    totals.update(
        {
            "eligible_removal_fraction": totals["eligible_removed_seconds"]
            / max(1e-12, totals["eligible_seconds"]),
            "damaged_event_fraction": totals["damaged_protected_events"]
            / max(1, totals["protected_events"]),
            "gap_recall": totals["matched_gaps"] / max(1, totals["annotated_gaps"]),
            "boundary_mean_ms": float(np.mean(errors) * 1000) if errors else None,
            "boundary_p95_ms": float(np.percentile(errors, 95) * 1000) if errors else None,
        }
    )
    return totals


def evaluate(predictor, records, policies=None):
    """Cache each recording's prediction across policies; never tune on test data."""
    if not records:
        raise ValueError("No recordings in the requested evaluation split")
    policies = policies or [predictor.config.cuts]
    rows = [[] for _ in policies]
    started = time.perf_counter()
    for record in records:
        waveform, sr = read_audio(record.audio)
        probabilities, boundaries = predictor.predict(waveform, sr)
        for results, policy in zip(rows, policies):
            plan = plan_cuts(
                probabilities,
                sr,
                waveform.shape[-1],
                waveform.shape[0],
                predictor.config.frontend.hop_seconds,
                policy,
                boundaries,
                waveform.numpy(),
            )
            results.append(score_plan(plan, record))
    elapsed = time.perf_counter() - started
    reports = []
    for policy, results in zip(policies, rows):
        totals = aggregate(results)
        reports.append(
            {
                "policy": asdict(policy),
                "totals": totals,
                "recordings": results,
                "evaluation_wall_seconds": elapsed,
                "evaluation_realtime_factor": elapsed / max(1e-12, totals["audio_seconds"]),
            }
        )
    return reports


def calibrate(predictor, records):
    if any(record.split != "validation" for record in records):
        raise ValueError("Threshold calibration is restricted to the validation split")
    base = predictor.config.cuts
    policies = [replace(base, enabled=False)]
    for threshold in (0.80, 0.90, 0.95, 0.98, 0.995):
        policies.append(
            replace(
                base,
                enabled=True,
                enter_probability=threshold,
                exit_probability=max(0.0, threshold - 0.08),
            )
        )
    reports = evaluate(predictor, records, policies)
    # Fail closed: no protected samples or fades damaged on validation. This is
    # a measured validation constraint, not a guarantee for unseen recordings.
    feasible = [
        i
        for i, report in enumerate(reports)
        if report["totals"]["protected_modified_seconds"] <= 1e-9
        and report["totals"]["damaged_protected_events"] == 0
    ]
    best = max(feasible, key=lambda i: reports[i]["totals"]["eligible_removed_seconds"])
    return policies[best], reports[best], reports
