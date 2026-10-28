"""Metrics from Section IV-A, implemented without a metrics dependency.

AUPRC and AUROC are computed from ranks, P@k follows the weekly review list,
and confidence intervals resample *drugs* rather than drug-weeks so that
within-drug correlation is respected.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import numpy as np


def _as_arrays(y_true, y_score) -> tuple[np.ndarray, np.ndarray]:
    y = np.asarray(y_true, dtype=float).ravel()
    s = np.asarray(y_score, dtype=float).ravel()
    if y.shape != s.shape:
        raise ValueError(f"shape mismatch: {y.shape} vs {s.shape}")
    return y, s


def average_precision(y_true, y_score) -> float:
    """Area under the precision-recall curve, step-wise (sklearn semantics)."""
    y, s = _as_arrays(y_true, y_score)
    positives = y.sum()
    if positives == 0:
        return float("nan")
    order = np.argsort(-s, kind="mergesort")
    y = y[order]
    tp = np.cumsum(y)
    fp = np.cumsum(1.0 - y)
    precision = tp / np.maximum(tp + fp, 1e-12)
    recall = tp / positives
    # step-wise AP: sum precision at each positive, times the recall step
    d_recall = np.diff(np.concatenate([[0.0], recall]))
    return float((precision * d_recall).sum())


def roc_auc(y_true, y_score) -> float:
    """Rank-based AUROC (Mann-Whitney U), ties averaged."""
    y, s = _as_arrays(y_true, y_score)
    n_pos = y.sum()
    n_neg = len(y) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s), dtype=float)
    sorted_scores = s[order]
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and sorted_scores[j + 1] == sorted_scores[i]:
            j += 1
        ranks[order[i : j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return float((ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def precision_at_k(y_true, y_score, k: int) -> float:
    """Fraction of the top-k ranked drugs that are positive."""
    y, s = _as_arrays(y_true, y_score)
    if len(y) == 0:
        return float("nan")
    k = min(k, len(y))
    top = np.argsort(-s, kind="mergesort")[:k]
    return float(y[top].sum() / k)


def brier_score(y_true, y_score) -> float:
    y, s = _as_arrays(y_true, y_score)
    return float(np.mean((s - y) ** 2))


@dataclass
class BootstrapCI:
    point: float
    low: float
    high: float
    n_resamples: int

    def __str__(self) -> str:
        return f"{self.point:.3f} [{self.low:.3f}-{self.high:.3f}]"


def bootstrap_ci_by_group(
    y_true,
    y_score,
    groups,
    metric=average_precision,
    *,
    n_resamples: int = 1000,
    alpha: float = 0.05,
    seed: int = 0,
) -> BootstrapCI:
    """Percentile bootstrap over the 95% interval, resampling groups (drugs)."""
    y, s = _as_arrays(y_true, y_score)
    g = np.asarray(groups)
    if len(g) != len(y):
        raise ValueError("groups must align with y_true")
    point = metric(y, s)
    by_group: dict[object, list[int]] = defaultdict(list)
    for idx, key in enumerate(g):
        by_group[key].append(idx)
    keys = list(by_group)
    rng = np.random.default_rng(seed)
    samples: list[float] = []
    for _ in range(n_resamples):
        draw = rng.choice(len(keys), size=len(keys), replace=True)
        indices = np.concatenate([by_group[keys[k]] for k in draw])
        value = metric(y[indices], s[indices])
        if not np.isnan(value):
            samples.append(value)
    if not samples:
        return BootstrapCI(point, float("nan"), float("nan"), n_resamples)
    low, high = np.quantile(samples, [alpha / 2, 1 - alpha / 2])
    return BootstrapCI(point, float(low), float(high), n_resamples)


def lead_time_weeks(alert_weeks: dict[str, list[int]], onset_weeks: dict[str, int],
                    threshold: int = 4) -> dict[str, float]:
    """Share of onsets flagged at least `threshold` weeks before posting.

    `alert_weeks` maps drug key -> sorted snapshot week numbers in which the
    drug appeared in the review list. `onset_weeks` maps drug key -> the week
    it was posted. Only drugs that actually entered shortage are counted, so
    the denominator is the number of onsets, not the number of alerts.
    """
    flagged_any = 0
    flagged_early = 0
    lead_times: list[int] = []
    for drug, onset in onset_weeks.items():
        weeks = alert_weeks.get(drug, [])
        prior = [w for w in weeks if w < onset]
        if not prior:
            continue
        flagged_any += 1
        lead = onset - min(prior)
        lead_times.append(lead)
        if lead >= threshold:
            flagged_early += 1
    total = len(onset_weeks)
    if total == 0:
        return {"onsets": 0, "any": float("nan"), "early": float("nan"),
                "median_lead": float("nan")}
    return {
        "onsets": total,
        "any": flagged_any / total,
        "early": flagged_early / total,
        "median_lead": float(np.median(lead_times)) if lead_times else float("nan"),
    }
