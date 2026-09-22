#!/usr/bin/env python3
"""
run_all.py — one entry point for the whole study.

Typical use on a multi-core CPU workstation:

    # 1. smoke test, ~3 minutes, confirms the pipeline is sane
    python run_all.py --design single --quick

    # 2. the real run that produces the paper's numbers, ~25-35 minutes
    python run_all.py --design crossfit --seeds 0

    # 3. seed-stability appendix, run last if time remains
    python run_all.py --design single --seeds 0 1 2 3 4 --quick --stages experiment

Stages can be run independently once predictions are cached:

    python run_all.py --stages figures
    python run_all.py --stages shap
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import config as C


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--design", choices=["single", "crossfit"],
                    default="crossfit")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--boot", type=int, default=C.N_BOOTSTRAP)
    ap.add_argument("--quick", action="store_true",
                    help="200 bootstrap iterations instead of 1000")
    ap.add_argument("--stages", nargs="+",
                    default=["experiment", "audit", "figures", "shap"],
                    choices=["experiment", "audit", "figures", "shap",
                             "sweep"])
    ap.add_argument("--shap-model", default=C.SHAP_HEADLINE_MODEL)
    ap.add_argument("--shap-attr", default="age")
    a = ap.parse_args()

    n_boot = 200 if a.quick else a.boot
    t0 = time.time()

    print("=" * 66)
    print(f"design={a.design}  seeds={a.seeds}  bootstrap={n_boot}")
    print(f"outputs -> {C.OUT}")
    print("=" * 66)

    if "experiment" in a.stages:
        from src.experiment import main as run_experiment
        run_experiment(seeds=a.seeds, design=a.design,
                       k_folds=a.folds, n_boot=n_boot)

    if "audit" in a.stages:
        from src.nullcal import run_null_audit
        run_null_audit(seed=a.seeds[0], design=a.design,
                       B=200 if a.quick else 500)

    if "sweep" in a.stages:
        from src.skew_sweep import fig7_sweep, run_sweep
        run_sweep(seed=a.seeds[0], attr="race", k_folds=a.folds)
        fig7_sweep(attr="race")

    if "figures" in a.stages:
        from src.figures import make_all
        make_all(seed=a.seeds[0], design=a.design)

    if "shap" in a.stages:
        from src.shap_analysis import run_shap
        run_shap(seed=a.seeds[0], model_name=a.shap_model, attr=a.shap_attr)

    print(f"\nALL DONE in {(time.time()-t0)/60:.1f} min")
    print(f"tables  -> {C.TABLES}")
    print(f"figures -> {C.FIGS}")


if __name__ == "__main__":
    main()
