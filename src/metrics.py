"""
metrics.py — every number the paper reports.

Design note for the Methodology section: ECE uses QUANTILE (equal-mass)
bins, not equal-width. At ~9% prevalence the predicted probabilities pile
up below 0.3; equal-width bins leave the upper bins nearly empty and the
resulting ECE is dominated by a handful of points. Reviewers of
calibration papers check the binning scheme first.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import (average_precision_score, brier_score_loss,
                             log_loss, roc_auc_score)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as C


# ======================================================================
# Binning
# ======================================================================
def _bin_index(p: np.ndarray, n_bins: int, strategy: str):
    """Return (bin_index_per_sample, n_effective_bins)."""
    p = np.asarray(p, dtype=float)
    if strategy == "quantile":
        edges = np.quantile(p, np.linspace(0.0, 1.0, n_bins + 1))
        edges = np.unique(edges)
    else:  # 'uniform' — provided only for the robustness appendix
        edges = np.linspace(0.0, 1.0, n_bins + 1)

    if edges.size < 2:
        return np.zeros_like(p, dtype=int), 1

    idx = np.searchsorted(edges, p, side="left") - 1
    idx = np.clip(idx, 0, edges.size - 2)
    return idx, edges.size - 1


def ece(y: np.ndarray, p: np.ndarray, n_bins: int = C.ECE_BINS_PRIMARY,
        strategy: str = C.ECE_STRATEGY) -> float:
    """Expected Calibration Error, Guo et al. (2017), with quantile bins.

        ECE = sum_m (|B_m| / N) * | acc(B_m) - conf(B_m) |
    """
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    n = y.size
    if n == 0:
        return np.nan

    idx, nb = _bin_index(p, n_bins, strategy)
    cnt = np.bincount(idx, minlength=nb).astype(float)
    s_p = np.bincount(idx, weights=p, minlength=nb)
    s_y = np.bincount(idx, weights=y, minlength=nb)

    nz = cnt > 0
    gap = np.abs(s_y[nz] / cnt[nz] - s_p[nz] / cnt[nz])
    return float(np.sum(cnt[nz] / n * gap))


def mce(y, p, n_bins=C.ECE_BINS_PRIMARY, strategy=C.ECE_STRATEGY,
        min_bin=10) -> float:
    """Maximum Calibration Error, ignoring bins with < min_bin samples."""
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    idx, nb = _bin_index(p, n_bins, strategy)
    cnt = np.bincount(idx, minlength=nb).astype(float)
    s_p = np.bincount(idx, weights=p, minlength=nb)
    s_y = np.bincount(idx, weights=y, minlength=nb)
    nz = cnt >= min_bin
    if not nz.any():
        return np.nan
    return float(np.max(np.abs(s_y[nz] / cnt[nz] - s_p[nz] / cnt[nz])))


def signed_error(y, p) -> float:
    """Signed calibration error = mean predicted - observed prevalence.

    ECE is unsigned and therefore hides WHICH DIRECTION a group is wrong.
    A group that is systematically under-predicted is a group whose
    patients do not get flagged for follow-up. The sign is the clinical
    harm; the magnitude alone is not.
    """
    return float(np.mean(p) - np.mean(y))


def reliability_curve(y, p, n_bins=C.ECE_BINS_PRIMARY,
                      strategy=C.ECE_STRATEGY):
    """Return (conf, acc, count) per non-empty bin, for plotting."""
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    idx, nb = _bin_index(p, n_bins, strategy)
    cnt = np.bincount(idx, minlength=nb).astype(float)
    s_p = np.bincount(idx, weights=p, minlength=nb)
    s_y = np.bincount(idx, weights=y, minlength=nb)
    nz = cnt > 0
    return s_p[nz] / cnt[nz], s_y[nz] / cnt[nz], cnt[nz]


# ======================================================================
# Aggregate metric bundle
# ======================================================================
def all_metrics(y, p) -> dict:
    y = np.asarray(y)
    p = np.clip(np.asarray(p, dtype=float), 1e-7, 1 - 1e-7)
    out = {
        "n": int(y.size),
        "prevalence": float(y.mean()),
        "brier": float(brier_score_loss(y, p)),
        "logloss": float(log_loss(y, p, labels=[0, 1])),
        "auc_roc": float(roc_auc_score(y, p)) if len(np.unique(y)) > 1 else np.nan,
        "auc_pr": float(average_precision_score(y, p)) if len(np.unique(y)) > 1 else np.nan,
        f"ece{C.ECE_BINS_PRIMARY}": ece(y, p, C.ECE_BINS_PRIMARY),
        f"ece{C.ECE_BINS_ROBUST}": ece(y, p, C.ECE_BINS_ROBUST),
        "mce": mce(y, p),
        "signed_err": signed_error(y, p),
    }
    return out


# ======================================================================
# Bootstrap
# ======================================================================
def bootstrap_ci(y, p, fn, B=C.N_BOOTSTRAP, alpha=C.BOOTSTRAP_ALPHA,
                 rng=None):
    """Percentile bootstrap CI for any statistic fn(y, p)."""
    rng = np.random.default_rng(0) if rng is None else rng
    y = np.asarray(y)
    p = np.asarray(p)
    n = y.size
    if n == 0:
        return np.nan, np.nan, np.nan
    stats = np.empty(B)
    for b in range(B):
        i = rng.integers(0, n, n)
        stats[b] = fn(y[i], p[i])
    point = fn(y, p)
    lo, hi = np.nanpercentile(stats, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(point), float(lo), float(hi)


# ======================================================================
# C1 — the Calibration Gap
# ======================================================================
def calibration_gap(y, p, groups, eligible, n_bins=C.ECE_BINS_PRIMARY):
    """CG = max over eligible subgroup pairs of |ECE_g - ECE_g'|.

    Equivalently max(ECE_g) - min(ECE_g) over eligible groups.
    Also returns the signed-error spread, which carries the direction of
    the harm that unsigned ECE conceals.
    """
    groups = np.asarray(groups)
    per = {}
    for g in eligible:
        m = groups == g
        if m.sum() == 0:
            continue
        per[g] = (ece(y[m], p[m], n_bins), signed_error(y[m], p[m]))
    if len(per) < 2:
        return dict(gap=np.nan, signed_spread=np.nan, per_group=per,
                    argmax=None, argmin=None)

    eces = {g: v[0] for g, v in per.items()}
    sgn = {g: v[1] for g, v in per.items()}
    gmax = max(eces, key=eces.get)
    gmin = min(eces, key=eces.get)
    return dict(gap=eces[gmax] - eces[gmin],
                signed_spread=max(sgn.values()) - min(sgn.values()),
                per_group=per, argmax=gmax, argmin=gmin)


def bootstrap_gap(y, p, groups, eligible, B=C.N_BOOTSTRAP,
                  alpha=C.BOOTSTRAP_ALPHA, n_bins=C.ECE_BINS_PRIMARY,
                  seed=0):
    """Stratified bootstrap for the Calibration Gap.

    We resample WITHIN each subgroup, preserving the observed subgroup
    sizes. Resampling the pooled test set instead would let subgroup n
    fluctuate and would conflate sampling error in group composition with
    sampling error in calibration — a reviewer will spot that.
    """
    rng = np.random.default_rng(seed)
    groups = np.asarray(groups)
    y = np.asarray(y)
    p = np.asarray(p)

    idx_by_g = {g: np.flatnonzero(groups == g) for g in eligible}
    idx_by_g = {g: i for g, i in idx_by_g.items() if i.size > 0}
    if len(idx_by_g) < 2:
        return np.nan, np.nan, np.nan, {}

    gaps = np.empty(B)
    grp_ece_boot = {g: np.empty(B) for g in idx_by_g}
    for b in range(B):
        vals = {}
        for g, idx in idx_by_g.items():
            s = rng.choice(idx, size=idx.size, replace=True)
            vals[g] = ece(y[s], p[s], n_bins)
            grp_ece_boot[g][b] = vals[g]
        gaps[b] = max(vals.values()) - min(vals.values())

    point = calibration_gap(y, p, groups, list(idx_by_g), n_bins)["gap"]
    lo, hi = np.nanpercentile(gaps, [100 * alpha / 2, 100 * (1 - alpha / 2)])

    grp_ci = {}
    for g, arr in grp_ece_boot.items():
        glo, ghi = np.nanpercentile(arr, [100 * alpha / 2,
                                          100 * (1 - alpha / 2)])
        grp_ci[g] = (float(glo), float(ghi))

    return float(point), float(lo), float(hi), grp_ci
