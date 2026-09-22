"""
figures.py — the six figures, sized for IEEE two-column.

Column width is 3.5 in, full width 7.16 in. Everything is exported as
vector PDF (what IEEE actually wants) plus a 300 dpi PNG for pasting into
slides and drafts.

Figures 2 and 4 carry the paper. Everything else is support.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as C
from src.metrics import reliability_curve

COL, FULL = 3.5, 7.16

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
    "axes.linewidth": 0.6, "grid.linewidth": 0.4, "lines.linewidth": 1.1,
    "axes.grid": True, "grid.alpha": 0.30,
    "figure.constrained_layout.use": True,
    "savefig.bbox": "tight", "pdf.fonttype": 42,
})

ARM_LABEL = {"raw": "Uncalibrated", "platt_global": "Global Platt",
             "iso_global": "Global Isotonic", "platt_group": "Group Platt",
             "iso_group": "Group Isotonic"}
ARM_COLOR = {"raw": "#444444", "platt_global": "#1f77b4",
             "iso_global": "#2ca02c", "platt_group": "#ff7f0e",
             "iso_group": "#d62728"}


def _save(fig, name):
    for ext in C.FIG_FORMATS:
        fig.savefig(C.FIGS / f"{name}.{ext}", dpi=C.DPI)
    plt.close(fig)
    print(f"  [fig] {name} -> {'/'.join(C.FIG_FORMATS)}")


def load_preds(seed=C.PRIMARY_SEED, design="crossfit"):
    """Load out-of-fold probabilities plus the evaluation frame."""
    z = np.load(C.CACHE / f"preds_seed{seed}_{design}.npz")
    y = z["y"]
    preds = {tuple(k.split("|")): z[k] for k in z.files if k != "y"}
    X = pd.read_csv(C.CACHE / f"Xeval_seed{seed}_{design}.csv.gz",
                    low_memory=False)
    return preds, y, X


# ======================================================================
# FIGURE 1 — reliability diagrams, 3 models x 3 global arms
# ======================================================================
def fig1_reliability(preds, y, attr="race"):
    models = ["LR", "RF", "HGB"]
    arms = ["raw", "platt_global", "iso_global"]
    fig = plt.figure(figsize=(FULL, 3.6))
    gs = GridSpec(2, 3, height_ratios=[3, 1], hspace=0.05, figure=fig)

    for j, m in enumerate(models):
        ax = fig.add_subplot(gs[0, j])
        axh = fig.add_subplot(gs[1, j], sharex=ax)
        ax.plot([0, 1], [0, 1], ls=":", c="k", lw=0.8, zorder=1)

        for arm in arms:
            p = preds[(m, "none", attr, arm)]
            ok = ~np.isnan(p)
            conf, acc, cnt = reliability_curve(y[ok], p[ok])
            ax.plot(conf, acc, "o-", ms=2.5, color=ARM_COLOR[arm],
                    label=ARM_LABEL[arm], zorder=3)

        p0 = preds[(m, "none", attr, "raw")]
        axh.hist(p0[~np.isnan(p0)], bins=40, range=(0, 0.6),
                 color="#888888", edgecolor="none")
        axh.set_yscale("log")
        axh.set_xlabel("Predicted probability")
        if j == 0:
            ax.set_ylabel("Observed frequency")
            axh.set_ylabel("Count")
            ax.legend(loc="upper left", frameon=False)
        ax.set_title(m)
        ax.set_xlim(0, 0.6); ax.set_ylim(0, 0.6)
        plt.setp(ax.get_xticklabels(), visible=False)

    _save(fig, "fig1_reliability_global")


# ======================================================================
# FIGURE 2 — per-subgroup reliability. THE MONEY FIGURE.
# ======================================================================
def fig2_subgroup_reliability(preds, y, X, groups_df, attr="race",
                              model="HGB", arms=("raw", "platt_global")):
    elig = (groups_df.query("attribute == @attr and eligible")
            ["group"].unique().tolist())
    if not elig:
        print(f"  [fig] skipping fig2 for {attr}: no eligible groups")
        return

    cmap = plt.get_cmap("tab10")
    fig, axes = plt.subplots(1, len(arms), figsize=(FULL, 2.9), sharey=True)
    axes = np.atleast_1d(axes)

    for ax, arm in zip(axes, arms):
        p = preds[(model, "none", attr, arm)]
        ax.plot([0, 1], [0, 1], ls=":", c="k", lw=0.8)
        for i, g in enumerate(elig):
            m = (X[attr].to_numpy() == g) & ~np.isnan(p)
            if m.sum() < 50:
                continue
            conf, acc, cnt = reliability_curve(y[m], p[m],
                                              n_bins=min(10, max(3, m.sum() // 60)))
            ax.plot(conf, acc, "o-", ms=3, color=cmap(i % 10),
                    label=f"{g} (n={m.sum():,})")
        ax.set_title(f"{model} — {ARM_LABEL[arm]}")
        ax.set_xlabel("Predicted probability")
        ax.set_xlim(0, 0.45); ax.set_ylim(0, 0.45)
    axes[0].set_ylabel("Observed frequency")
    axes[0].legend(loc="upper left", frameon=False)
    _save(fig, f"fig2_subgroup_reliability_{attr}")


# ======================================================================
# FIGURE 3 — ROC and PR, to show discrimination is untouched
# ======================================================================
def fig3_roc_pr(preds, y, attr="race", model="HGB"):
    from sklearn.metrics import (average_precision_score, precision_recall_curve,
                                 roc_auc_score, roc_curve)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(FULL, 2.7))
    for arm in C.CALIBRATION_ARMS:
        p = preds[(model, "none", attr, arm)]
        ok = ~np.isnan(p)
        fpr, tpr, _ = roc_curve(y[ok], p[ok])
        a1.plot(fpr, tpr, color=ARM_COLOR[arm], lw=1.0,
                label=f"{ARM_LABEL[arm]} ({roc_auc_score(y[ok], p[ok]):.4f})")
        pr, rc, _ = precision_recall_curve(y[ok], p[ok])
        a2.plot(rc, pr, color=ARM_COLOR[arm], lw=1.0,
                label=f"{ARM_LABEL[arm]} ({average_precision_score(y[ok], p[ok]):.4f})")
    a1.plot([0, 1], [0, 1], ls=":", c="k", lw=0.8)
    a1.set_xlabel("False positive rate"); a1.set_ylabel("True positive rate")
    a1.set_title(f"ROC — {model}"); a1.legend(loc="lower right", frameon=False)
    a2.axhline(y.mean(), ls=":", c="k", lw=0.8)
    a2.set_xlabel("Recall"); a2.set_ylabel("Precision")
    a2.set_title(f"Precision–Recall — {model}")
    a2.legend(loc="upper right", frameon=False)
    _save(fig, "fig3_roc_pr")


# ======================================================================
# FIGURE 4 — Calibration Gap with bootstrap CIs. CARRIES THE PAPER.
# ======================================================================
def fig4_gap_bars(gaps_df, attrs=("race", "age"), cw="none"):
    d = gaps_df.query("class_weight == @cw")
    models = ["LR", "RF", "HGB"]
    arms = C.CALIBRATION_ARMS
    fig, axes = plt.subplots(1, len(attrs), figsize=(FULL, 2.8), sharey=False)
    axes = np.atleast_1d(axes)

    for ax, attr in zip(axes, attrs):
        sub = d.query("attribute == @attr")
        if sub.empty:
            continue
        w = 0.15
        xs = np.arange(len(models))
        for k, arm in enumerate(arms):
            vals, los, his = [], [], []
            for m in models:
                r = sub.query("model == @m and arm == @arm")
                if r.empty:
                    vals.append(np.nan); los.append(0); his.append(0); continue
                r = r.iloc[0]
                vals.append(r["gap"])
                los.append(max(0, r["gap"] - r["gap_lo"]))
                his.append(max(0, r["gap_hi"] - r["gap"]))
            ax.bar(xs + (k - 2) * w, vals, w, color=ARM_COLOR[arm],
                   label=ARM_LABEL[arm], edgecolor="none")
            ax.errorbar(xs + (k - 2) * w, vals, yerr=[los, his], fmt="none",
                        ecolor="k", elinewidth=0.6, capsize=1.5)
        ax.set_xticks(xs); ax.set_xticklabels(models)
        ax.set_title(f"Calibration Gap — {attr}")
        ax.set_ylabel("CG (max pairwise $\\Delta$ECE)")
    axes[0].legend(frameon=False, ncol=2, fontsize=6)
    _save(fig, "fig4_calibration_gap")


def fig4b_signed_forest(groups_df, attr="age", model="HGB", cw="none",
                        arms=("raw", "platt_global")):
    """Signed subgroup error with CIs — the direction of the harm."""
    d = groups_df.query("attribute == @attr and model == @model and "
                        "class_weight == @cw and eligible")
    if d.empty:
        return
    order = sorted(d["group"].unique())
    fig, ax = plt.subplots(figsize=(COL, 0.32 * len(order) + 1.1))
    off = np.linspace(-0.18, 0.18, len(arms))
    for k, arm in enumerate(arms):
        s = d.query("arm == @arm").set_index("group").reindex(order)
        ypos = np.arange(len(order)) + off[k]
        ax.errorbar(s["signed_err"], ypos,
                    xerr=[s["signed_err"] - s["signed_lo"],
                          s["signed_hi"] - s["signed_err"]],
                    fmt="o", ms=3, lw=0.9, capsize=1.8,
                    color=ARM_COLOR[arm], label=ARM_LABEL[arm])
    ax.axvline(0, ls=":", c="k", lw=0.8)
    ax.set_yticks(np.arange(len(order))); ax.set_yticklabels(order)
    ax.set_xlabel("Signed calibration error\n(mean predicted $-$ observed)")
    ax.set_title(f"{model} — {attr}")
    ax.legend(frameon=False, loc="best")
    _save(fig, f"fig4b_signed_error_{attr}")


# ======================================================================
# FIGURES 5 & 6 live in shap_analysis.py (they need the explainer objects)
# ======================================================================
def make_all(seed=C.PRIMARY_SEED, design="crossfit"):
    preds, y, X = load_preds(seed, design)
    groups_df = pd.read_csv(C.TABLES / "subgroup_metrics.csv")
    gaps_df = pd.read_csv(C.TABLES / "calibration_gap.csv")
    groups_df = groups_df.query("seed == @seed")
    gaps_df = gaps_df.query("seed == @seed")

    print("[figures] building ...")
    fig1_reliability(preds, y)
    for attr in C.SENSITIVE_ATTRS:
        fig2_subgroup_reliability(preds, y, X, groups_df, attr=attr)
        fig4b_signed_forest(groups_df, attr=attr)
    fig3_roc_pr(preds, y)
    fig4_gap_bars(gaps_df)
    print(f"[figures] done -> {C.FIGS}")


if __name__ == "__main__":
    make_all()
