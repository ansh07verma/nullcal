"""
shap_analysis.py — contribution C3 (supporting analysis, not the headline).

Two questions:
  (a) Where does residual miscalibration live in feature space?
  (b) Does recalibration reorder the explanation?

Methodological choice that matters. Attribution is computed in RAW
feature space (about 42 clinical variables), not in the ~200-column
one-hot space. Two reasons:

  1. A calibrated pipeline is a composition (one-hot -> model -> link).
     Only a model-agnostic explainer can pass through the calibration
     map, and its cost scales with the number of features. In raw space
     the Permutation explainer needs ~2p+1 evaluations per row, which is
     minutes on a standard multi-core CPU; in one-hot space it is hours.
  2. "Number of prior inpatient visits" is a clinical variable a reviewer
     can reason about. "number_inpatient_bin_7" is not.

TreeExplainer is still run on the uncalibrated tree models as a fast
cross-check, with one-hot attributions summed back to parent features.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as C
from src.calib import fit_calibrators, make_model
from src.data_prep import build_dataset, make_splits
from src.figures import COL, FULL, _save


# ======================================================================
def _mean_abs_shap(values) -> np.ndarray:
    v = np.asarray(values)
    if v.ndim == 3:            # (n, p, n_classes) -> positive class
        v = v[:, :, -1]
    return np.abs(v).mean(axis=0)


def _perm_shap(f, X_explain, X_background, feature_names, max_evals=None):
    """Model-agnostic Permutation SHAP through an arbitrary function."""
    import shap
    p = X_explain.shape[1]
    max_evals = max_evals or (2 * p + 1)
    masker = shap.maskers.Independent(X_background, max_samples=len(X_background))
    ex = shap.explainers.Permutation(f, masker, feature_names=feature_names)
    return ex(X_explain, max_evals=max_evals, silent=True)


# ======================================================================
def run_shap(seed=C.PRIMARY_SEED, model_name=C.SHAP_HEADLINE_MODEL,
             attr="age", n_explain=C.SHAP_PERM_ROWS,
             n_background=C.SHAP_BACKGROUND, verbose=True):
    """Fit one model on a single split and explain raw vs calibrated."""
    import shap

    t0 = time.time()
    d = build_dataset(seed=seed)
    X, y = d["X"], d["y"]
    tr, ca, te = make_splits(y, d["pid"], seed)

    Xtr, ytr = X[tr], y[tr]
    Xca, yca = X[ca].reset_index(drop=True), y[ca]
    Xte = X[te].reset_index(drop=True)
    yte = y[te]

    base = make_model(model_name, d["num_cols"], d["cat_cols"], None, seed)
    base.fit(Xtr, ytr)
    cals = fit_calibrators(base, Xca, yca, Xca[attr].to_numpy())
    print(f"[shap] base + calibrators fitted ({time.time()-t0:.1f}s)")

    rng = np.random.default_rng(seed)
    idx_e = rng.choice(len(Xte), min(n_explain, len(Xte)), replace=False)
    idx_b = rng.choice(len(Xte), min(n_background, len(Xte)), replace=False)
    feats = list(X.columns)
    cat_cols = d["cat_cols"]

    # SHAP's tabular masker is numeric-only: it calls np.isclose on the
    # feature matrix, which raises on string categoricals. So the
    # explainer works over an integer-coded matrix, and the wrapper
    # decodes codes back to category labels before the pipeline sees
    # them. Masking a coded column therefore swaps in a whole category
    # drawn from the background, which is exactly the intended
    # counterfactual — and the attribution stays at the level of the
    # clinical variable rather than a one-hot indicator.
    codebook = {c: list(pd.Index(X[c].unique())) for c in cat_cols}
    code_index = {c: {v: i for i, v in enumerate(vals)}
                  for c, vals in codebook.items()}
    num_dtypes = {c: X[c].dtype for c in feats if c not in cat_cols}

    def encode(df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        for c in cat_cols:
            out[c] = out[c].map(code_index[c]).fillna(0).astype(int)
        return out.astype(float)

    def _as_frame(A) -> pd.DataFrame:
        df = pd.DataFrame(np.asarray(A, dtype=float), columns=feats)
        for c in cat_cols:
            codes = np.clip(np.rint(df[c].to_numpy()).astype(int),
                            0, len(codebook[c]) - 1)
            df[c] = [codebook[c][i] for i in codes]
        for c, dt in num_dtypes.items():
            df[c] = df[c].astype(dt)
        return df

    Xe = encode(Xte.iloc[idx_e])
    Xb = encode(Xte.iloc[idx_b])

    # ---- the three functions we explain -------------------------------
    def f_raw(A):
        return base.predict_proba(_as_frame(A))[:, 1]

    def f_platt(A):
        return cals["platt_global"].predict_proba(_as_frame(A))[:, 1]

    def f_group(A):
        df = _as_frame(A)
        return cals["platt_group"].predict_proba(df, df[attr].to_numpy())

    results, rankings = {}, {}
    for tag, f in [("raw", f_raw), ("platt_global", f_platt),
                   ("platt_group", f_group)]:
        t1 = time.time()
        sv = _perm_shap(f, Xe, Xb, feats)
        results[tag] = sv
        rankings[tag] = pd.Series(_mean_abs_shap(sv.values), index=feats)
        if verbose:
            print(f"[shap] {tag:13s} {time.time()-t1:6.1f}s")

    # ---- drift: Kendall tau between mean|SHAP| rankings ---------------
    rows = []
    for a, b in [("raw", "platt_global"), ("raw", "platt_group"),
                 ("platt_global", "platt_group")]:
        ra = rankings[a].rank(ascending=False)
        rb = rankings[b].rank(ascending=False)
        tau, ptau = kendalltau(ra, rb)
        rho, prho = spearmanr(ra, rb)
        top10 = set(rankings[a].nlargest(10).index) & set(rankings[b].nlargest(10).index)
        rows.append(dict(comparison=f"{a} vs {b}", kendall_tau=tau,
                         tau_p=ptau, spearman_rho=rho, rho_p=prho,
                         top10_overlap=len(top10),
                         max_rank_shift=int(np.abs(ra - rb).max())))
    drift = pd.DataFrame(rows)
    drift.to_csv(C.TABLES / f"shap_drift_{model_name}_{attr}.csv", index=False)

    rank_tbl = pd.DataFrame(rankings)
    rank_tbl["rank_raw"] = rank_tbl["raw"].rank(ascending=False)
    rank_tbl = rank_tbl.sort_values("raw", ascending=False)
    rank_tbl.to_csv(C.TABLES / f"shap_rankings_{model_name}_{attr}.csv")

    print("\n[shap] attribution drift")
    print(drift.to_string(index=False))

    # ---- residual miscalibration localisation -------------------------
    p_raw = base.predict_proba(Xte)[:, 1]
    p_cal = cals["platt_global"].predict_proba(Xte)[:, 1]
    resid = pd.DataFrame({"resid": p_cal - yte})
    loc_rows = []
    for a in C.SENSITIVE_ATTRS + [C.GENDER_ATTR]:
        for g, sub in Xte.groupby(a, observed=True):
            m = Xte[a].to_numpy() == g
            if m.sum() < C.MIN_GROUP_N:
                continue
            loc_rows.append(dict(attribute=a, group=g, n=int(m.sum()),
                                 mean_residual_raw=float((p_raw - yte)[m].mean()),
                                 mean_residual_cal=float((p_cal - yte)[m].mean())))
    pd.DataFrame(loc_rows).to_csv(
        C.TABLES / f"residual_localisation_{model_name}.csv", index=False)

    _fig5_beeswarm(results["raw"], model_name)
    _fig6_drift(rankings, model_name, attr, drift)

    print(f"[shap] total {time.time()-t0:.1f}s")
    return results, rankings, drift


# ======================================================================
def _fig5_beeswarm(sv, model_name, top=15):
    import shap
    fig = plt.figure(figsize=(COL, 3.2))
    shap.plots.beeswarm(sv, max_display=top, show=False,
                        color_bar_label="Feature value")
    plt.title(f"{model_name} — uncalibrated", fontsize=8)
    _save(plt.gcf(), "fig5a_shap_beeswarm")

    fig = plt.figure(figsize=(COL, 3.2))
    shap.plots.bar(sv, max_display=top, show=False)
    plt.title(f"{model_name} — mean |SHAP|", fontsize=8)
    _save(plt.gcf(), "fig5b_shap_bar")


def _fig6_drift(rankings, model_name, attr, drift, top=15):
    """Slope plot: feature rank before vs after recalibration."""
    raw = rankings["raw"].rank(ascending=False)
    order = rankings["raw"].nlargest(top).index.tolist()
    cols = ["raw", "platt_global", "platt_group"]
    ranks = {c: rankings[c].rank(ascending=False) for c in cols}

    fig, ax = plt.subplots(figsize=(COL, 3.4))
    xs = np.arange(len(cols))
    cmap = plt.get_cmap("tab20")
    for i, feat in enumerate(order):
        ys = [ranks[c][feat] for c in cols]
        ax.plot(xs, ys, "o-", ms=3, lw=0.9, color=cmap(i % 20))
        ax.annotate(feat, (xs[0] - 0.06, ys[0]), ha="right", va="center",
                    fontsize=5.5)
    ax.set_xticks(xs)
    ax.set_xticklabels(["Uncalib.", "Global\nPlatt", "Group\nPlatt"])
    ax.invert_yaxis()
    ax.set_ylabel("Rank by mean |SHAP|")
    tau = drift.loc[drift.comparison == "raw vs platt_global",
                    "kendall_tau"].iloc[0]
    ax.set_title(f"Attribution stability ($\\tau$={tau:.3f})", fontsize=8)
    ax.set_xlim(-1.4, len(cols) - 0.7)
    _save(fig, "fig6_attribution_drift")


if __name__ == "__main__":
    run_shap()
