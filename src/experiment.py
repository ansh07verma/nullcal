"""
experiment.py — train, calibrate, and produce test-set probabilities.

Two experimental designs are supported.

  design='single'   60/20/20 train/calibration/test, one split.
                    Fast (~1 min). Use it to smoke-test the pipeline.

  design='crossfit' 5-fold cross-fitting. Each fold is held out as test;
                    the remaining 80% is split 75/25 into train and
                    calibration, so the global proportions are still
                    60/20/20 and the calibration set is still disjoint
                    from both train and test. Every patient receives
                    exactly one out-of-fold probability.

Why cross-fitting matters here, and this belongs in the paper: under a
single 20% test split the minority race subgroups fall below the
minimum-support rule (Hispanic, Asian and Other carry fewer than 25
positive events) and the Calibration Gap collapses to a single pairwise
comparison. Cross-fitting raises every subgroup's effective test size
fivefold at no cost to independence, which is what makes a five-group
Gap reportable at all. It also averages over split randomness.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as C
from src.calib import fit_calibrators, make_model, predict_arm
from src.data_prep import build_dataset, subgroup_support, eligible_groups

MODELS = ["LR", "RF", "HGB"]


# ======================================================================
def _fold_iter(y, pid, seed, design, k_folds=5):
    """Yield (fold_id, train_mask, cal_mask, test_mask)."""
    if design == "single":
        from src.data_prep import make_splits
        tr, ca, te = make_splits(y, pid, seed)
        yield 0, tr, ca, te
        return

    uniq_pid, first_idx = np.unique(pid, return_index=True)
    strat = y[first_idx]
    skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=seed)

    for k, (dev_i, te_i) in enumerate(skf.split(uniq_pid, strat)):
        pid_te = uniq_pid[te_i]
        pid_dev, y_dev = uniq_pid[dev_i], strat[dev_i]
        # 75/25 of the 80% development pool -> 60/20 of the whole
        pid_tr, pid_ca = train_test_split(
            pid_dev, train_size=0.75, stratify=y_dev,
            random_state=seed * 100 + k)
        tr, ca, te = (np.isin(pid, pid_tr), np.isin(pid, pid_ca),
                      np.isin(pid, pid_te))
        assert not (tr & ca).any() and not (tr & te).any() and not (ca & te).any()
        yield k, tr, ca, te


# ======================================================================
def run(seed=C.PRIMARY_SEED, design="crossfit", k_folds=5,
        models=None, verbose=True):
    """Fit everything and return a dict of out-of-fold probabilities."""
    models = models or MODELS
    t0 = time.time()

    d = build_dataset(seed=seed)
    X, y, pid = d["X"], d["y"], d["pid"]
    num_cols, cat_cols = d["num_cols"], d["cat_cols"]
    n = len(y)

    attrs = C.SENSITIVE_ATTRS + [C.GENDER_ATTR]

    # preds[(model, cw, attr, arm)] = array of length n (NaN where unfilled)
    preds = {}
    covered = np.zeros(n, dtype=bool)
    fallback_log = []

    for fold, tr, ca, te in _fold_iter(y, pid, seed, design, k_folds):
        covered |= te
        Xtr, ytr = X[tr], y[tr]
        Xca, yca = X[ca].reset_index(drop=True), y[ca]
        Xte = X[te].reset_index(drop=True)

        for mname in models:
            for cw in C.IMBALANCE_VARIANTS:
                cw_tag = "balanced" if cw == "balanced" else "none"
                t1 = time.time()
                base = make_model(mname, num_cols, cat_cols, cw, seed)
                base.fit(Xtr, ytr)

                p_raw = base.predict_proba(Xte)[:, 1]

                for attr in attrs:
                    g_ca = Xca[attr].to_numpy()
                    g_te = Xte[attr].to_numpy()
                    cals = fit_calibrators(base, Xca, yca, g_ca)

                    for gc_arm in ("platt_group", "iso_group"):
                        for g, gn, gp in cals[gc_arm].fallback_groups_:
                            fallback_log.append(dict(
                                seed=seed, fold=fold, model=mname,
                                class_weight=cw_tag, attribute=attr,
                                arm=gc_arm, group=g, n_cal=gn, pos_cal=gp))

                    for arm in C.CALIBRATION_ARMS:
                        key = (mname, cw_tag, attr, arm)
                        if key not in preds:
                            preds[key] = np.full(n, np.nan)
                        pv = (p_raw if arm == "raw"
                              else predict_arm(arm, base, cals, Xte, g_te))
                        preds[key][te] = pv

                if verbose:
                    print(f"  [fold {fold}] {mname:3s} cw={cw_tag:8s} "
                          f"{time.time()-t1:6.1f}s")

    assert covered.all() or design == "single", "some patients uncovered"

    # ---- support tables computed on the evaluated population -----------
    eval_mask = covered
    Xe = X[eval_mask].reset_index(drop=True)
    ye = y[eval_mask]
    support, elig = {}, {}
    for attr in attrs:
        s = subgroup_support(Xe, ye, attr)
        s.insert(0, "seed", seed)
        support[attr] = s
        elig[attr] = eligible_groups(s)

    if verbose:
        print(f"[run] seed={seed} design={design} "
              f"total {time.time()-t0:.1f}s  evaluated n={eval_mask.sum()}")

    return dict(preds=preds, y=ye, X=Xe, eval_mask=eval_mask,
                support=support, eligible=elig,
                fallback=pd.DataFrame(fallback_log), audit=d["audit"],
                seed=seed, design=design)


# ======================================================================
def score(res, n_boot=C.N_BOOTSTRAP, verbose=True):
    """Turn probabilities into every table the paper needs."""
    from src.metrics import (all_metrics, bootstrap_ci, bootstrap_gap,
                             calibration_gap, ece, signed_error)

    y, X = res["y"], res["X"]
    seed = res["seed"]

    overall_rows, group_rows, gap_rows = [], [], []

    for (mname, cw, attr, arm), p in res["preds"].items():
        p = p[res["eval_mask"]] if p.size != y.size else p
        ok = ~np.isnan(p)
        pv, yv = p[ok], y[ok]
        gv = X[attr].to_numpy()[ok]

        m = all_metrics(yv, pv)
        m.update(seed=seed, model=mname, class_weight=cw,
                 attribute=attr, arm=arm)
        overall_rows.append(m)

        elig = res["eligible"][attr]

        for g in X[attr].unique():
            mask = gv == g
            if mask.sum() < 20:
                continue
            e_pt, e_lo, e_hi = bootstrap_ci(
                yv[mask], pv[mask], lambda a, b: ece(a, b),
                B=n_boot, rng=np.random.default_rng(abs(hash((g, arm))) % 2**31))
            s_pt, s_lo, s_hi = bootstrap_ci(
                yv[mask], pv[mask], lambda a, b: signed_error(a, b),
                B=n_boot, rng=np.random.default_rng(abs(hash((g, arm, 1))) % 2**31))
            prev = float(yv[mask].mean())
            group_rows.append(dict(
                seed=seed, model=mname, class_weight=cw, attribute=attr,
                arm=arm, group=g, n=int(mask.sum()),
                positives=int(yv[mask].sum()),
                prevalence=prev,
                mean_pred=float(pv[mask].mean()),
                ece=e_pt, ece_lo=e_lo, ece_hi=e_hi,
                signed_err=s_pt, signed_lo=s_lo, signed_hi=s_hi,
                # Relative signed error is what makes the harm legible.
                # An absolute bias of -0.01 sounds negligible; at a 9%
                # base rate it is an 11% relative under-estimation of
                # risk for every patient in that group.
                rel_signed_err=(s_pt / prev) if prev > 0 else np.nan,
                eligible=bool(g in elig)))

        if len(elig) >= 2:
            cg = calibration_gap(yv, pv, gv, elig)
            g_pt, g_lo, g_hi, _ = bootstrap_gap(
                yv, pv, gv, elig, B=n_boot, seed=seed)
            gap_rows.append(dict(
                seed=seed, model=mname, class_weight=cw, attribute=attr,
                arm=arm, n_groups=len(elig),
                gap=g_pt, gap_lo=g_lo, gap_hi=g_hi,
                signed_spread=cg["signed_spread"],
                worst_group=cg["argmax"], best_group=cg["argmin"],
                pooled_ece=m[f"ece{C.ECE_BINS_PRIMARY}"],
                auc_roc=m["auc_roc"]))

    overall = pd.DataFrame(overall_rows)
    groups = pd.DataFrame(group_rows)
    gaps = pd.DataFrame(gap_rows)

    if verbose:
        print(f"[score] overall={overall.shape} groups={groups.shape} "
              f"gaps={gaps.shape}")
    return overall, groups, gaps


# ======================================================================
def main(seeds=None, design="crossfit", k_folds=5, n_boot=C.N_BOOTSTRAP):
    seeds = seeds or C.SEEDS
    OV, GR, GP, SUP, FB = [], [], [], [], []
    audit = None

    for s in seeds:
        print(f"\n===== SEED {s} ({design}) =====")
        res = run(seed=s, design=design, k_folds=k_folds)
        ov, gr, gp = score(res, n_boot=n_boot)
        OV.append(ov); GR.append(gr); GP.append(gp)
        FB.append(res["fallback"])
        for a, s_df in res["support"].items():
            SUP.append(s_df)
        audit = res["audit"]
        np.savez_compressed(
            C.CACHE / f"preds_seed{s}_{design}.npz",
            y=res["y"],
            **{"|".join(k): v for k, v in res["preds"].items()})
        res["X"].to_csv(C.CACHE / f"Xeval_seed{s}_{design}.csv.gz", index=False)

    overall = pd.concat(OV, ignore_index=True)
    groups = pd.concat(GR, ignore_index=True)
    gaps = pd.concat(GP, ignore_index=True)
    support = pd.concat(SUP, ignore_index=True)
    fallback = pd.concat(FB, ignore_index=True) if len(FB) else pd.DataFrame()

    overall.to_csv(C.TABLES / "overall_metrics.csv", index=False)
    groups.to_csv(C.TABLES / "subgroup_metrics.csv", index=False)
    gaps.to_csv(C.TABLES / "calibration_gap.csv", index=False)
    support.to_csv(C.TABLES / "subgroup_support.csv", index=False)
    fallback.to_csv(C.TABLES / "groupwise_fallbacks.csv", index=False)
    with open(C.TABLES / "data_audit.json", "w") as f:
        json.dump(audit, f, indent=2)

    print(f"\n[main] wrote 5 CSVs + data_audit.json to {C.TABLES}")
    return overall, groups, gaps, support, fallback


if __name__ == "__main__":
    main()
