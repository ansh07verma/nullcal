"""
calib.py — model pipelines and the five calibration arms.

Trap 2 (pre-solved): on scikit-learn >= 1.6, cv="prefit" is deprecated and
it is REMOVED in 1.8. Every tutorial online still uses the old form. The
correct modern usage wraps the already-fitted estimator in FrozenEstimator:

    base.fit(X_train, y_train)
    cal = CalibratedClassifierCV(FrozenEstimator(base), method='sigmoid')
    cal.fit(X_cal, y_cal)      # X_cal MUST be disjoint from X_train
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.frozen import FrozenEstimator
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as C


# ======================================================================
# Preprocessing transformer (shared by all three models)
# ======================================================================
def make_preprocessor(num_cols, cat_cols, scale: bool):
    """One-hot for categoricals; standardise numerics only for LR.

    min_frequency collapses rare categories into an 'infrequent' level,
    which keeps the one-hot matrix around 250 columns instead of
    thousands and stops rare levels from producing unstable splits.
    """
    num_steps = [("impute", SimpleImputer(strategy="median"))]
    if scale:
        num_steps.append(("scale", StandardScaler()))

    try:
        ohe = OneHotEncoder(handle_unknown="infrequent_if_exist",
                            min_frequency=20, sparse_output=False)
    except TypeError:                     # very old sklearn
        ohe = OneHotEncoder(handle_unknown="ignore", sparse=False)

    return ColumnTransformer(
        [("num", Pipeline(num_steps), num_cols),
         ("cat", ohe, cat_cols)],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def make_model(name: str, num_cols, cat_cols, class_weight, seed: int):
    """Return a fitted-ready Pipeline for LR / RF / HGB."""
    if name == "LR":
        clf = LogisticRegression(class_weight=class_weight,
                                 random_state=seed, **C.LR_PARAMS)
        pre = make_preprocessor(num_cols, cat_cols, scale=True)
    elif name == "RF":
        p = dict(C.RF_PARAMS)
        p["random_state"] = seed
        clf = RandomForestClassifier(class_weight=class_weight, **p)
        pre = make_preprocessor(num_cols, cat_cols, scale=False)
    elif name == "HGB":
        p = dict(C.HGB_PARAMS)
        p["random_state"] = seed
        # HistGradientBoosting has no class_weight before sklearn 1.4;
        # from 1.4 onward it does. Fall back to sample_weight if needed.
        try:
            clf = HistGradientBoostingClassifier(class_weight=class_weight, **p)
        except TypeError:
            clf = HistGradientBoostingClassifier(**p)
        pre = make_preprocessor(num_cols, cat_cols, scale=False)
    else:
        raise ValueError(f"unknown model {name}")

    return Pipeline([("pre", pre), ("clf", clf)])


# ======================================================================
# Group-wise calibration (C2)
# ======================================================================
class GroupwiseCalibrator:
    """One post-hoc calibrator per subgroup, with a global fallback.

    The minimum-support rule is applied at FIT time on the calibration
    split: a subgroup gets its own calibrator only if it carries at least
    MIN_CAL_N members and MIN_CAL_POS positive events there. Subgroups
    below threshold fall back to the global calibrator, and the fallback
    list is recorded so it can be disclosed in the paper.

    Note the expected failure mode, which is itself a finding: isotonic
    regression fitted on ~200 subgroup points is high-variance and can be
    WORSE than the global map. Sigmoid/Platt has two parameters and
    degrades gracefully. Do not hide this; it is the honest version of C2.
    """

    def __init__(self, base_fitted, method="sigmoid",
                 min_n=C.MIN_CAL_N, min_pos=C.MIN_CAL_POS):
        self.base = base_fitted
        self.method = method
        self.min_n = min_n
        self.min_pos = min_pos

    def fit(self, X_cal, y_cal, groups_cal):
        y_cal = np.asarray(y_cal)
        groups_cal = np.asarray(groups_cal)

        self.global_ = CalibratedClassifierCV(
            FrozenEstimator(self.base), method=self.method
        ).fit(X_cal, y_cal)

        self.per_group_, self.fallback_groups_, self.fitted_groups_ = {}, [], []
        for g in np.unique(groups_cal):
            m = groups_cal == g
            n, pos = int(m.sum()), int(y_cal[m].sum())
            neg = n - pos
            if n >= self.min_n and pos >= self.min_pos and neg >= self.min_pos:
                Xg = X_cal[m] if isinstance(X_cal, np.ndarray) else X_cal.loc[m]
                self.per_group_[g] = CalibratedClassifierCV(
                    FrozenEstimator(self.base), method=self.method
                ).fit(Xg, y_cal[m])
                self.fitted_groups_.append(g)
            else:
                self.fallback_groups_.append((g, n, pos))
        return self

    def predict_proba(self, X, groups):
        groups = np.asarray(groups)
        out = self.global_.predict_proba(X)[:, 1].astype(float)
        for g, cal in self.per_group_.items():
            m = groups == g
            if m.any():
                Xg = X[m] if isinstance(X, np.ndarray) else X.loc[m]
                out[m] = cal.predict_proba(Xg)[:, 1]
        return out


def fit_calibrators(base_fitted, X_cal, y_cal, groups_cal):
    """Fit all four post-hoc arms. Arm 1 (raw) needs no fitting."""
    plat_g = CalibratedClassifierCV(FrozenEstimator(base_fitted),
                                    method="sigmoid").fit(X_cal, y_cal)
    iso_g = CalibratedClassifierCV(FrozenEstimator(base_fitted),
                                   method="isotonic").fit(X_cal, y_cal)
    plat_grp = GroupwiseCalibrator(base_fitted, "sigmoid").fit(
        X_cal, y_cal, groups_cal)
    iso_grp = GroupwiseCalibrator(base_fitted, "isotonic").fit(
        X_cal, y_cal, groups_cal)
    return {"platt_global": plat_g, "iso_global": iso_g,
            "platt_group": plat_grp, "iso_group": iso_grp}


def predict_arm(arm: str, base_fitted, cals: dict, X, groups):
    """Uniform prediction interface across all five arms."""
    if arm == "raw":
        return base_fitted.predict_proba(X)[:, 1].astype(float)
    obj = cals[arm]
    if isinstance(obj, GroupwiseCalibrator):
        return obj.predict_proba(X, groups)
    return obj.predict_proba(X)[:, 1].astype(float)
