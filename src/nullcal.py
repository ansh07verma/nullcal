"""
nullcal.py — contribution C1: making subgroup ECE comparable across
subgroups of unequal size.

THE PROBLEM
-----------
ECE is a sum of absolute deviations. Absolute deviations are strictly
positive in expectation even when the model is perfectly calibrated,
because each bin's observed frequency is a noisy estimate of its mean
predicted probability. The noise shrinks like 1/sqrt(n), so a SMALL
subgroup gets a LARGE ECE for free.

Any comparison of ECE across subgroups of different sizes therefore
confounds miscalibration with sample size. On the Diabetes-130 benchmark
this is not a subtle effect: the Asian subgroup (n=488) has an
uncalibrated ECE of 0.0258, while a perfectly calibrated model at that
same n produces an expected ECE of 0.0296. The observed value is BELOW
the noise floor. A naive Calibration Gap reports this group as the worst
calibrated in the cohort.

THE CORRECTION
--------------
Reference every subgroup's ECE against its own null distribution:
simulate labels from the model's own predicted probabilities, so the
group is perfectly calibrated by construction, and recompute ECE at the
observed n. Report

    excess ECE   = ECE_obs - E[ECE_null]
    calibration z = (ECE_obs - E[ECE_null]) / sd[ECE_null]
    percentile    = P(ECE_null < ECE_obs)

The z-score is dimensionless and comparable across subgroup sizes, which
is exactly what the raw ECE is not.

CLOSED FORM
-----------
For M equal-mass bins, n samples and mean predicted probability p, each
bin holds n/M points and its deviation is approximately half-normal with
scale sqrt(p(1-p)M/n). Averaging over bins,

    E[ECE_null] ~= sqrt( 2 * p * (1-p) * M / (pi * n) )

Validated against simulation to within 6% for n in [300, 52305] and
M in {10, 15}. The closed form is what lets a reviewer check subgroup
sizes without rerunning anything, and it yields a sample-size rule.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as C
from src.metrics import ece

# Empirical correction factor: the asymptotic form slightly overstates
# the null at finite n (quantile bins are not exactly equal-mass and the
# half-normal approximation ignores discreteness). Measured across the
# validated grid.
_ANALYTIC_FUDGE = 0.935


# ======================================================================
def null_ece_analytic(n: int, pbar: float, M: int = C.ECE_BINS_PRIMARY,
                      calibrated: bool = True) -> float:
    """Closed-form expected ECE of a PERFECTLY calibrated model."""
    if n <= 0:
        return np.nan
    val = np.sqrt(2.0 * pbar * (1.0 - pbar) * M / (np.pi * n))
    return float(val * (_ANALYTIC_FUDGE if calibrated else 1.0))


def null_ece_simulated(p: np.ndarray, M: int = C.ECE_BINS_PRIMARY,
                       B: int = 1000, rng=None) -> np.ndarray:
    """Null distribution of ECE by simulating y* ~ Bernoulli(p).

    This is the version to report. The closed form is for reasoning about
    design; the simulation respects the actual shape of the predicted
    probability distribution, which the closed form does not.
    """
    rng = np.random.default_rng(0) if rng is None else rng
    p = np.asarray(p, dtype=float)
    n = p.size
    out = np.empty(B)
    for b in range(B):
        ys = (rng.random(n) < p).astype(float)
        out[b] = ece(ys, p, M)
    return out


def ece_null_referenced(y, p, M: int = C.ECE_BINS_PRIMARY, B: int = 1000,
                        rng=None) -> dict:
    """Observed ECE placed against its own perfect-calibration null."""
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    obs = ece(y, p, M)
    null = null_ece_simulated(p, M, B, rng)
    mu, sd = float(null.mean()), float(null.std(ddof=1))
    return dict(
        n=int(y.size),
        ece_obs=float(obs),
        null_mean=mu,
        null_sd=sd,
        null_p95=float(np.percentile(null, 95)),
        excess_ece=float(obs - mu),
        cal_z=float((obs - mu) / sd) if sd > 0 else np.nan,
        null_pctile=float((null < obs).mean()),
        # A group is only evidence of miscalibration if it clears its own
        # 95th null percentile. Below that, "high ECE" means "small n".
        exceeds_null=bool(obs > np.percentile(null, 95)),
        analytic_null=null_ece_analytic(y.size, float(p.mean()), M),
    )


# ======================================================================
# The power rule — how big must a subgroup be to be auditable at all?
# ======================================================================
def min_n_for_gap(delta: float, pbar: float = 0.09,
                  M: int = C.ECE_BINS_PRIMARY, ratio: float = 1.0) -> float:
    """Subgroup size needed before an ECE difference of `delta` is visible.

    Solves E[ECE_null] = delta / ratio for n:

        n = 2 * p * (1-p) * M * ratio^2 / (pi * delta^2)

    ratio=1 is the absolute floor, where the noise floor equals the effect
    you are trying to measure — you cannot detect anything below this.
    ratio=2 is a defensible minimum for actually reporting a difference.
    """
    if delta <= 0:
        return np.inf
    f = _ANALYTIC_FUDGE ** 2
    return float(2.0 * pbar * (1 - pbar) * M * (ratio ** 2) * f
                 / (np.pi * delta ** 2))


def auditability_table(deltas=(0.005, 0.01, 0.02, 0.05),
                       pbar: float = 0.09, M: int = C.ECE_BINS_PRIMARY
                       ) -> pd.DataFrame:
    """Table 1 material: minimum subgroup n by target effect size."""
    rows = []
    for d in deltas:
        rows.append(dict(
            target_delta_ece=d,
            n_floor=round(min_n_for_gap(d, pbar, M, ratio=1.0)),
            n_recommended=round(min_n_for_gap(d, pbar, M, ratio=2.0)),
        ))
    return pd.DataFrame(rows)


# ======================================================================
# Null-referenced Calibration Gap
# ======================================================================
def corrected_gap(y, p, groups, eligible, M: int = C.ECE_BINS_PRIMARY,
                  B: int = 500, seed: int = 0) -> dict:
    """Calibration Gap computed on bias-corrected quantities.

    Returns the naive gap alongside the excess-ECE gap and the z-gap, so
    the paper can show all three side by side. The naive-vs-corrected
    contrast is the empirical core of C1.
    """
    rng = np.random.default_rng(seed)
    groups = np.asarray(groups)
    per = {}
    for g in eligible:
        m = groups == g
        if m.sum() < 30:
            continue
        per[g] = ece_null_referenced(y[m], p[m], M, B, rng)

    if len(per) < 2:
        return dict(naive_gap=np.nan, excess_gap=np.nan, z_gap=np.nan,
                    per_group=per, n_auditable=0)

    naive = {g: v["ece_obs"] for g, v in per.items()}
    exc = {g: v["excess_ece"] for g, v in per.items()}
    zs = {g: v["cal_z"] for g, v in per.items()}

    return dict(
        naive_gap=max(naive.values()) - min(naive.values()),
        naive_worst=max(naive, key=naive.get),
        excess_gap=max(exc.values()) - min(exc.values()),
        excess_worst=max(exc, key=exc.get),
        z_gap=max(zs.values()) - min(zs.values()),
        z_worst=max(zs, key=zs.get),
        n_exceeding_null=sum(v["exceeds_null"] for v in per.values()),
        n_groups=len(per),
        per_group=per,
    )


def group_null_table(y, p, X: pd.DataFrame, attr: str, eligible,
                     M: int = C.ECE_BINS_PRIMARY, B: int = 1000,
                     seed: int = 0, min_n: int = 30) -> pd.DataFrame:
    """Per-subgroup null-referenced table — the paper's central table."""
    rng = np.random.default_rng(seed)
    gv = X[attr].to_numpy()
    rows = []
    for g in eligible:
        m = gv == g
        if m.sum() < min_n:
            continue
        r = ece_null_referenced(y[m], p[m], M, B, rng)
        r.update(attribute=attr, group=g,
                 positives=int(np.asarray(y)[m].sum()),
                 prevalence=float(np.asarray(y)[m].mean()),
                 mean_pred=float(np.asarray(p)[m].mean()))
        rows.append(r)
    cols = ["attribute", "group", "n", "positives", "prevalence",
            "mean_pred", "ece_obs", "null_mean", "null_sd", "null_p95",
            "analytic_null", "excess_ece", "cal_z", "null_pctile",
            "exceeds_null"]
    return pd.DataFrame(rows)[cols]


# ======================================================================
def fig8_null_bands(y, p_by_arm: dict, X: pd.DataFrame, attr: str,
                    eligible, model_name: str = "HGB", B: int = 600):
    """Figure 8 — observed subgroup ECE against its own null band.

    This is the figure that replaces the old Figure 4. Any group whose
    marker sits inside the shaded band is statistically indistinguishable
    from perfectly calibrated, however large its ECE looks.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from src.figures import FULL, _save, ARM_COLOR, ARM_LABEL

    arms = list(p_by_arm)
    fig, axes = plt.subplots(1, len(arms), figsize=(FULL, 2.9), sharey=True)
    axes = np.atleast_1d(axes)
    rng = np.random.default_rng(0)

    for ax, arm in zip(axes, arms):
        p = p_by_arm[arm]
        gv = X[attr].to_numpy()
        ns, obs, lo, hi, mid, labels = [], [], [], [], [], []
        for g in eligible:
            m = gv == g
            if m.sum() < 30:
                continue
            null = null_ece_simulated(p[m], C.ECE_BINS_PRIMARY, B, rng)
            ns.append(m.sum()); labels.append(g)
            obs.append(ece(y[m], p[m]))
            lo.append(np.percentile(null, 5))
            hi.append(np.percentile(null, 95))
            mid.append(null.mean())

        o = np.argsort(ns)
        ns = np.array(ns)[o]; obs = np.array(obs)[o]
        lo = np.array(lo)[o]; hi = np.array(hi)[o]; mid = np.array(mid)[o]
        labels = [labels[i] for i in o]

        ax.fill_between(ns, lo, hi, color="#cccccc", alpha=0.8,
                        label="Null band (perfectly calibrated)")
        ax.plot(ns, mid, ls="--", c="#777777", lw=0.9)
        ax.plot(ns, obs, "o", ms=5, color=ARM_COLOR.get(arm, "#d62728"),
                label="Observed", zorder=5)
        for x, yv, lab in zip(ns, obs, labels):
            ax.annotate(lab, (x, yv), textcoords="offset points",
                        xytext=(0, 6), ha="center", fontsize=5.5)
        ax.set_xscale("log")
        ax.set_xlabel("Subgroup size $n$ (log)")
        ax.set_title(f"{model_name} — {ARM_LABEL.get(arm, arm)}")
    axes[0].set_ylabel("ECE")
    axes[0].legend(frameon=False, loc="upper right", fontsize=6)
    _save(fig, f"fig8_null_bands_{attr}")


if __name__ == "__main__":
    print("Auditability rule (prevalence 0.09, M=10 quantile bins)\n")
    print(auditability_table().to_string(index=False))


# ======================================================================
# Stage runner: null-referenced audit over cached predictions
# ======================================================================
def run_null_audit(seed: int = C.PRIMARY_SEED, design: str = "crossfit",
                   models=("LR", "RF", "HGB"), class_weight: str = "none",
                   attrs=None, B: int = 500, verbose: bool = True):
    """Produce the corrected tables for every reported configuration.

    Reads cached out-of-fold probabilities, so it costs nothing to rerun
    with a different bin count or B once the models are fitted.
    """
    from src.figures import load_preds

    attrs = attrs or C.SENSITIVE_ATTRS
    preds, y, X = load_preds(seed, design)

    per_group_rows, gap_rows = [], []
    for attr in attrs:
        sup = None
        from src.data_prep import subgroup_support, eligible_groups
        sup = subgroup_support(X, y, attr)
        elig = eligible_groups(sup)
        gv = X[attr].to_numpy()

        for mname in models:
            for arm in C.CALIBRATION_ARMS:
                key = (mname, class_weight, attr, arm)
                if key not in preds:
                    continue
                p = preds[key]
                t = group_null_table(y, p, X, attr, elig, B=B, seed=seed)
                t.insert(0, "arm", arm)
                t.insert(0, "class_weight", class_weight)
                t.insert(0, "model", mname)
                t.insert(0, "seed", seed)
                per_group_rows.append(t)

                cg = corrected_gap(y, p, gv, elig, B=B, seed=seed)
                gap_rows.append(dict(
                    seed=seed, model=mname, class_weight=class_weight,
                    attribute=attr, arm=arm,
                    naive_gap=cg["naive_gap"], naive_worst=cg["naive_worst"],
                    excess_gap=cg["excess_gap"], excess_worst=cg["excess_worst"],
                    z_gap=cg["z_gap"], z_worst=cg["z_worst"],
                    n_exceeding_null=cg["n_exceeding_null"],
                    n_groups=cg["n_groups"]))
                if verbose:
                    print(f"  [audit] {mname:3s} {attr:6s} {arm:13s} "
                          f"naive={cg['naive_gap']:.5f} "
                          f"excess={cg['excess_gap']:.5f} "
                          f"real={cg['n_exceeding_null']}/{cg['n_groups']}")

    pg = pd.concat(per_group_rows, ignore_index=True)
    gp = pd.DataFrame(gap_rows)
    pg.to_csv(C.TABLES / "nullreferenced_subgroups.csv", index=False)
    gp.to_csv(C.TABLES / "nullreferenced_gap.csv", index=False)
    auditability_table().to_csv(C.TABLES / "auditability_rule.csv", index=False)

    for attr in attrs:
        elig = eligible_groups(subgroup_support(X, y, attr))
        pba = {a: preds[(C.SHAP_HEADLINE_MODEL, class_weight, attr, a)]
               for a in ("raw", "platt_global")
               if (C.SHAP_HEADLINE_MODEL, class_weight, attr, a) in preds}
        if len(pba) == 2:
            fig8_null_bands(y, pba, X, attr, elig,
                            model_name=C.SHAP_HEADLINE_MODEL, B=min(B, 600))

    print(f"[audit] wrote nullreferenced_*.csv to {C.TABLES}")
    return pg, gp
