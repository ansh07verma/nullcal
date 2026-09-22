"""
data_prep.py — load, clean, and split the UCI Diabetes 130-US Hospitals data.

Every step here is a reviewer defence. Read the docstrings before changing
anything: several of these look like optional hygiene and are not.
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as C

warnings.filterwarnings("ignore", category=FutureWarning)


# ======================================================================
# 1. LOADING
# ======================================================================
def load_raw(force_download: bool = False) -> pd.DataFrame:
    """Fetch the dataset, preferring ucimlrepo, caching to parquet.

    Falls back to the static CSV endpoint if ucimlrepo is unavailable or
    the machine has an SSL-intercepting proxy.
    """
    if C.RAW_CACHE.exists() and not force_download:
        print(f"[load] cache hit -> {C.RAW_CACHE}")
        return pd.read_csv(C.RAW_CACHE, low_memory=False)

    REQUIRED = {"encounter_id", "patient_nbr", "readmitted",
                "discharge_disposition_id", "race", "gender", "age"}

    df = None
    try:
        from ucimlrepo import fetch_ucirepo
        print(f"[load] fetching UCI id={C.UCI_ID} via ucimlrepo ...")
        ds = fetch_ucirepo(id=C.UCI_ID)
        # ucimlrepo splits ID columns out of `features`: encounter_id and
        # patient_nbr live in `ds.data.ids`. Without them there is no
        # patient-level deduplication and no patient-level split, so we
        # validate and fall back to the raw CSV rather than proceed.
        parts = [q for q in (getattr(ds.data, "ids", None),
                             ds.data.features, ds.data.targets)
                 if q is not None]
        df = pd.concat(parts, axis=1)
        df = df.loc[:, ~df.columns.duplicated()]
        gap = REQUIRED - set(df.columns)
        if gap:
            raise KeyError(f"ucimlrepo response missing {sorted(gap)}")
    except Exception as e:                                   # noqa: BLE001
        print(f"[load] ucimlrepo path unusable ({type(e).__name__}: {e})")
        print(f"[load] falling back to {C.UCI_CSV_FALLBACK}")
        df = pd.read_csv(C.UCI_CSV_FALLBACK, low_memory=False)

    gap = REQUIRED - set(df.columns)
    if gap:
        raise RuntimeError(
            f"Loaded data missing required columns {sorted(gap)}; "
            f"got {len(df.columns)} columns.")

    df.to_csv(C.RAW_CACHE, index=False)
    print(f"[load] raw shape = {df.shape}  cached -> {C.RAW_CACHE}")
    return df


# ======================================================================
# 2. ICD-9 GROUPING  (Strack et al. 2014, Table 2 — 9 clinical buckets)
# ======================================================================
def _icd9_bucket(code) -> str:
    """Collapse a raw ICD-9 code into one of nine clinical categories.

    ~700 distinct codes per diagnosis column would explode one-hot
    encoding into thousands of columns and degrade computational performance.
    """
    if code is None or (isinstance(code, float) and np.isnan(code)):
        return "Missing"
    s = str(code).strip()
    if s in ("?", "", "nan", "None"):
        return "Missing"

    # E and V codes are supplementary classifications -> "Other"
    if s[0].upper() in ("E", "V"):
        return "Other"

    try:
        v = float(s)
    except ValueError:
        return "Other"

    # Diabetes: the 250.xx family
    if 250 <= v < 251:
        return "Diabetes"

    iv = int(v)
    if (390 <= iv <= 459) or iv == 785:
        return "Circulatory"
    if (460 <= iv <= 519) or iv == 786:
        return "Respiratory"
    if (520 <= iv <= 579) or iv == 787:
        return "Digestive"
    if 800 <= iv <= 999:
        return "Injury"
    if 710 <= iv <= 739:
        return "Musculoskeletal"
    if (580 <= iv <= 629) or iv == 788:
        return "Genitourinary"
    if 140 <= iv <= 239:
        return "Neoplasms"
    return "Other"


# ======================================================================
# 3. CLEANING
# ======================================================================
_MISSING_TOKENS = {"?", "", "nan", "NaN", "None", "none", "NULL", "null",
                   "Not Available", "Not Mapped", "Unknown/Invalid", "<NA>"}


def _clean_cat(s: pd.Series, fill: str) -> pd.Series:
    """Normalise a categorical column to plain Python strings.

    Works identically on pandas 2.x (object dtype) and 3.x (str dtype),
    and collapses every flavour of missing marker onto one level.
    """
    out = s.astype("object").where(s.notna(), fill)
    out = out.map(lambda v: str(v).strip())
    return out.map(lambda v: fill if v in _MISSING_TOKENS else v)


def preprocess(df: pd.DataFrame, verbose: bool = True) -> tuple[pd.DataFrame, dict]:
    """Apply the six non-negotiable preprocessing steps.

    Returns the clean frame plus an audit dict — the audit is what you
    paste into the paper's Data section, so nothing is unverifiable.
    """
    audit = {}
    n0 = len(df)
    audit["n_encounters_raw"] = n0

    df = df.copy()
    df.columns = [c.strip() for c in df.columns]

    # -- Target -------------------------------------------------------
    # Binary: 1 if readmitted within 30 days, else 0.
    df["y"] = (df["readmitted"].astype(str).str.strip() == "<30").astype(int)
    audit["prevalence_raw"] = float(df["y"].mean())

    # -- Step 3: drop expired / hospice discharges ---------------------
    # These patients CANNOT be readmitted. Leaving them in creates
    # structural negatives that inflate every performance number.
    dd = pd.to_numeric(df["discharge_disposition_id"], errors="coerce")
    mask_alive = ~dd.isin(C.EXPIRED_HOSPICE_IDS)
    audit["n_dropped_expired_hospice"] = int((~mask_alive).sum())
    df = df[mask_alive].copy()

    # -- Drop invalid gender (3 rows of 'Unknown/Invalid') -------------
    g = df["gender"].astype(str).str.strip()
    mask_g = g.isin(["Male", "Female"])
    audit["n_dropped_invalid_gender"] = int((~mask_g).sum())
    df = df[mask_g].copy()

    # -- Step 1: deduplicate to FIRST encounter per patient ------------
    # patient_nbr repeats. Without this the same patient appears in both
    # train and test -> leakage. encounter_id is monotonically increasing
    # in time, so the minimum encounter_id is the earliest admission.
    df = df.sort_values("encounter_id", kind="mergesort")
    before = len(df)
    df = df.drop_duplicates(subset="patient_nbr", keep="first").copy()
    audit["n_dropped_repeat_encounters"] = int(before - len(df))
    audit["n_patients"] = len(df)
    audit["prevalence_final"] = float(df["y"].mean())

    # -- Step 4: high-missingness columns ------------------------------
    df = df.drop(columns=[c for c in C.DROP_HIGH_MISSING if c in df.columns])
    # medical_specialty (~49% missing) is KEPT: missingness is informative
    # (it encodes whether the admitting service was recorded at all).
    df["medical_specialty"] = _clean_cat(df["medical_specialty"], "Missing")

    # -- Race: missing becomes an explicit level -----------------------
    # The UCI distribution encodes missing race as '?' in the zip release
    # and as an empty cell in the static CSV endpoint. Handle both.
    df["race"] = _clean_cat(df["race"], "Unknown")

    # -- Step 5: ICD-9 grouping ----------------------------------------
    for c in ["diag_1", "diag_2", "diag_3"]:
        df[c + "_grp"] = df[c].map(_icd9_bucket)
    df = df.drop(columns=["diag_1", "diag_2", "diag_3"])

    # -- Admission / discharge / source IDs are categorical, not numeric
    for c in ["admission_type_id", "discharge_disposition_id",
              "admission_source_id"]:
        df[c] = df[c].astype(str)

    # -- Remaining missing markers -> 'Missing' in every non-numeric col
    # NOTE: do NOT test `dtype == object` here. pandas 3.0 gives string
    # columns a dedicated `str` dtype, so that test silently matches
    # nothing and every missing value survives into the model. This bug
    # is invisible until a group column hits np.unique with mixed types.
    for c in df.columns:
        if c == "y" or pd.api.types.is_numeric_dtype(df[c]):
            continue
        df[c] = _clean_cat(df[c], "Missing")

    # -- Drop constant columns (examide, citoglipton are single-valued) -
    const = [c for c in df.columns
             if c not in ("y",) and df[c].nunique(dropna=False) <= 1]
    audit["constant_columns_dropped"] = const
    df = df.drop(columns=const)

    df = df.drop(columns=["readmitted"])

    if verbose:
        print("\n[preprocess] audit")
        for k, v in audit.items():
            print(f"    {k:34s} : {v}")

    return df.reset_index(drop=True), audit


# ======================================================================
# 4. FEATURE / TARGET SEPARATION
# ======================================================================
def split_features(df: pd.DataFrame):
    """Return (X, y, patient_ids, numeric_cols, categorical_cols)."""
    pid = df["patient_nbr"].to_numpy()
    y = df["y"].to_numpy()

    drop = C.IDENTIFIER_COLS + ["y"]
    X = df.drop(columns=[c for c in drop if c in df.columns]).copy()

    if not C.INCLUDE_SENSITIVE_AS_FEATURES:
        X = X.drop(columns=[c for c in ["race", "gender", "age"]
                            if c in X.columns])

    num_cols = [c for c in X.columns
                if pd.api.types.is_numeric_dtype(X[c])]
    cat_cols = [c for c in X.columns if c not in num_cols]
    return X, y, pid, num_cols, cat_cols


# ======================================================================
# 5. PATIENT-LEVEL STRATIFIED SPLIT  (60 / 20 / 20)
# ======================================================================
def make_splits(y: np.ndarray, pid: np.ndarray, seed: int):
    """Split by *patient*, stratified on the target.

    After deduplication one row == one patient, so an ordinary stratified
    split is already patient-level. We nonetheless partition the unique
    patient identifiers explicitly, so the guarantee survives if
    deduplication is ever disabled — and so we can state it in the paper
    without hand-waving.

    Returns three boolean index arrays: train / calibration / test.
    The calibration set is disjoint from BOTH train and test. This is
    the single most common error in the calibration literature.
    """
    from sklearn.model_selection import train_test_split

    uniq_pid, first_idx = np.unique(pid, return_index=True)
    strat = y[first_idx]

    pid_tr, pid_hold, y_tr, y_hold = train_test_split(
        uniq_pid, strat, train_size=C.TRAIN_FRAC,
        stratify=strat, random_state=seed)

    rel = C.CAL_FRAC / (C.CAL_FRAC + C.TEST_FRAC)   # 0.5
    pid_cal, pid_te = train_test_split(
        pid_hold, train_size=rel, stratify=y_hold, random_state=seed)

    tr = np.isin(pid, pid_tr)
    ca = np.isin(pid, pid_cal)
    te = np.isin(pid, pid_te)

    assert not (tr & ca).any() and not (tr & te).any() and not (ca & te).any()
    assert (tr | ca | te).all()
    return tr, ca, te


# ======================================================================
# 6. SUBGROUP SUPPORT TABLE  (the reviewer kill-shot defence)
# ======================================================================
def subgroup_support(X: pd.DataFrame, y: np.ndarray, attr: str,
                     min_n: int = C.MIN_GROUP_N,
                     min_pos: int = C.MIN_GROUP_POS) -> pd.DataFrame:
    """Count members and positive events per level; flag eligibility.

    Groups below threshold are REPORTED as 'insufficient support', never
    silently dropped. At ~11% prevalence a small subgroup can carry ~12
    events, where ECE is pure noise.
    """
    rows = []
    for lvl, idx in X.groupby(attr, observed=True).groups.items():
        pos = int(y[X.index.get_indexer(idx)].sum())
        n = len(idx)
        excluded_level = lvl in C.EXCLUDE_LEVELS.get(attr, [])
        rows.append({
            "attribute": attr,
            "group": lvl,
            "n": n,
            "positives": pos,
            "prevalence": pos / n if n else np.nan,
            "meets_support": bool(n >= min_n and pos >= min_pos),
            "eligible_for_gap": bool(n >= min_n and pos >= min_pos
                                     and not excluded_level),
            "reason": ("non-demographic level" if excluded_level
                       else "" if (n >= min_n and pos >= min_pos)
                       else f"insufficient support (n={n}, pos={pos})"),
        })
    return (pd.DataFrame(rows)
            .sort_values("n", ascending=False)
            .reset_index(drop=True))


def eligible_groups(support_df: pd.DataFrame) -> list:
    return support_df.loc[support_df["eligible_for_gap"], "group"].tolist()


# ======================================================================
def build_dataset(seed: int = C.PRIMARY_SEED, force_download: bool = False):
    """One call that returns everything downstream code needs."""
    raw = load_raw(force_download=force_download)
    clean, audit = preprocess(raw)
    X, y, pid, num_cols, cat_cols = split_features(clean)
    tr, ca, te = make_splits(y, pid, seed)

    audit["n_train"] = int(tr.sum())
    audit["n_cal"] = int(ca.sum())
    audit["n_test"] = int(te.sum())
    audit["n_features_raw"] = X.shape[1]
    audit["n_numeric"] = len(num_cols)
    audit["n_categorical"] = len(cat_cols)

    return dict(X=X, y=y, pid=pid, tr=tr, ca=ca, te=te,
                num_cols=num_cols, cat_cols=cat_cols, audit=audit)


if __name__ == "__main__":
    d = build_dataset()
    X, y = d["X"], d["y"]
    print(f"\nX = {X.shape}   prevalence = {y.mean():.4f}")
    print(f"train/cal/test = {d['tr'].sum()}/{d['ca'].sum()}/{d['te'].sum()}")
    for attr in C.SENSITIVE_ATTRS + [C.GENDER_ATTR]:
        s = subgroup_support(X[d["te"]].reset_index(drop=True),
                             y[d["te"]], attr)
        print(f"\n--- support: {attr} (TEST split) ---")
        print(s.to_string(index=False))
        s.to_csv(C.TABLES / f"support_{attr}.csv", index=False)
