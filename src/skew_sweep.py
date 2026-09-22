"""
skew_sweep.py — the mechanism experiment (contribution C2b).

The observational result is that global recalibration widens the
Calibration Gap for race (majority share 0.77) but not for age (0.25) or
gender (0.53). Three attributes is an anecdote, not a dose-response.

This module turns it into a controlled experiment. The base model and the
test set are held fixed; only the COMPOSITION of the calibration set is
varied. We resample the calibration split so that the majority subgroup
holds a target share s in {0.4 ... 0.9} at constant total size, refit the
global calibrator, and measure the Calibration Gap on the untouched test
fold.

If gap inflation is monotone in s, the harm is not a property of race,
of this dataset, or of Platt scaling. It is a property of fitting one
reliability map on a skewed pool — which is what every deployed
recalibration pipeline does. That is a mechanism, and mechanisms survive
review in a way that single point estimates do not.

Cost: the base model is fitted once per fold; each sweep point only
refits a two-parameter sigmoid, so the whole sweep is minutes.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from sklearn.model_selection import StratifiedKFold, train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as C
from src.calib import make_model
from src.data_prep import build_dataset, subgroup_support, eligible_groups
from src.metrics import ece

SHARES = [0.40, 0.50, 0.60, 0.70, 0.80, 0.90]


def _gap(y, p, gv, elig):
    e = {g: ece(y[gv == g], p[gv == g]) for g in elig
         if (gv == g).sum() >= 30}
    if len(e) < 2:
        return np.nan, e
    return max(e.values()) - min(e.values()), e


def _resample_to_share(rng, groups, y, majority, target_share, total):
    """Indices for a calibration pool where `majority` holds target_share."""
    maj_idx = np.flatnonzero(groups == majority)
    min_idx = np.flatnonzero(groups != majority)

    n_maj = int(round(total * target_share))
    n_min = total - n_maj
    if n_maj > maj_idx.size or n_min > min_idx.size:
        # fall back to the largest feasible total at this share
        scale = min(maj_idx.size / max(n_maj, 1), min_idx.size / max(n_min, 1))
        n_maj, n_min = int(n_maj * scale), int(n_min * scale)
    if n_maj < 50 or n_min < 50:
        return None

    take = np.concatenate([rng.choice(maj_idx, n_maj, replace=False),
                           rng.choice(min_idx, n_min, replace=False)])
    if y[take].sum() < 25 or (1 - y[take]).sum() < 25:
        return None
    return take


def run_sweep(seed=C.PRIMARY_SEED, attr="race", model_name="HGB",
              k_folds=5, shares=None, n_repeats=5, cal_total=None,
              verbose=True):
    shares = shares or SHARES
    t0 = time.time()

    d = build_dataset(seed=seed)
    X, y, pid = d["X"], d["y"], d["pid"]
    sup = subgroup_support(X.reset_index(drop=True), y, attr)
    elig = eligible_groups(sup)
    majority = sup.loc[sup["eligible_for_gap"], "group"].iloc[0]
    if verbose:
        print(f"[sweep] attr={attr} majority={majority} eligible={elig}")

    uniq_pid, first_idx = np.unique(pid, return_index=True)
    strat = y[first_idx]
    skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=seed)

    rows = []
    for k, (dev_i, te_i) in enumerate(skf.split(uniq_pid, strat)):
        pid_te = uniq_pid[te_i]
        pid_tr, pid_ca = train_test_split(
            uniq_pid[dev_i], train_size=0.75, stratify=strat[dev_i],
            random_state=seed * 100 + k)
        tr, ca, te = (np.isin(pid, pid_tr), np.isin(pid, pid_ca),
                      np.isin(pid, pid_te))

        base = make_model(model_name, d["num_cols"], d["cat_cols"], None, seed)
        base.fit(X[tr], y[tr])

        Xca = X[ca].reset_index(drop=True)
        yca, gca = y[ca], Xca[attr].to_numpy()
        Xte = X[te].reset_index(drop=True)
        yte, gte = y[te], Xte[attr].to_numpy()

        p_raw = base.predict_proba(Xte)[:, 1]
        g_raw, _ = _gap(yte, p_raw, gte, elig)

        total = cal_total or int(0.55 * len(Xca))
        rng = np.random.default_rng(seed * 1000 + k)

        for s in shares:
            for r in range(n_repeats):
                take = _resample_to_share(rng, gca, yca, majority, s, total)
                if take is None:
                    continue
                cal = CalibratedClassifierCV(
                    FrozenEstimator(base), method="sigmoid"
                ).fit(Xca.loc[take], yca[take])
                p_cal = cal.predict_proba(Xte)[:, 1]
                g_cal, per = _gap(yte, p_cal, gte, elig)
                rows.append(dict(
                    seed=seed, fold=k, attribute=attr, model=model_name,
                    target_share=s, repeat=r, n_cal=len(take),
                    actual_share=float((gca[take] == majority).mean()),
                    gap_raw=g_raw, gap_cal=g_cal, delta_gap=g_cal - g_raw,
                    pooled_ece_raw=ece(yte, p_raw),
                    pooled_ece_cal=ece(yte, p_cal)))
        if verbose:
            print(f"  [fold {k}] done  {time.time()-t0:.0f}s")

    df = pd.DataFrame(rows)
    df.to_csv(C.TABLES / f"skew_sweep_{attr}_{model_name}.csv", index=False)

    summ = (df.groupby("target_share")
            .agg(delta_gap_mean=("delta_gap", "mean"),
                 delta_gap_sd=("delta_gap", "std"),
                 gap_cal_mean=("gap_cal", "mean"),
                 pooled_ece_cal=("pooled_ece_cal", "mean"),
                 n=("delta_gap", "size"))
            .reset_index())
    summ.to_csv(C.TABLES / f"skew_sweep_summary_{attr}_{model_name}.csv",
                index=False)
    if verbose:
        print("\n[sweep] summary")
        print(summ.to_string(index=False))
        from scipy.stats import spearmanr
        rho, p = spearmanr(df["actual_share"], df["delta_gap"])
        print(f"\n[sweep] Spearman(share, delta_gap) = {rho:.3f}  p = {p:.2e}")
        print(f"[sweep] total {time.time()-t0:.0f}s")
    return df, summ


def fig7_sweep(attr="race", model_name="HGB"):
    """Figure 7 — gap inflation as a function of calibration-set skew."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from src.figures import COL, _save

    df = pd.read_csv(C.TABLES / f"skew_sweep_{attr}_{model_name}.csv")
    g = df.groupby("target_share")["delta_gap"]
    mu, sd, n = g.mean(), g.std(), g.size()
    se = sd / np.sqrt(n)

    fig, ax = plt.subplots(figsize=(COL, 2.4))
    ax.axhline(0, ls=":", c="k", lw=0.8)
    ax.errorbar(mu.index, mu.values, yerr=1.96 * se.values, fmt="o-",
                ms=3.5, lw=1.1, capsize=2, color="#d62728")
    ax.set_xlabel("Majority-subgroup share of the calibration set")
    ax.set_ylabel("$\\Delta$ Calibration Gap\n(after $-$ before recalibration)")
    ax.set_title(f"{model_name} — {attr}", fontsize=8)
    _save(fig, f"fig7_skew_sweep_{attr}")


if __name__ == "__main__":
    run_sweep()
    fig7_sweep()
