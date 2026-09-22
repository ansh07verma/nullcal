"""
config.py — Single source of truth for every constant in the study.

Every number a reviewer might ask about lives here, so the paper's
Methodology section can be written straight from this file.
"""
from pathlib import Path

# ----------------------------------------------------------------------
# Paths
# ----------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs"
CACHE = OUT / "cache"
FIGS = OUT / "figures"
TABLES = OUT / "tables"
for _d in (OUT, CACHE, FIGS, TABLES):
    _d.mkdir(parents=True, exist_ok=True)

UCI_ID = 296
UCI_CSV_FALLBACK = "https://archive.ics.uci.edu/static/public/296/data.csv"
RAW_CACHE = CACHE / "raw.csv.gz"   # csv.gz avoids a pyarrow dependency

# ----------------------------------------------------------------------
# Reproducibility
# ----------------------------------------------------------------------
SEEDS = [0]                 # override from CLI: --seeds 0 1 2 3 4
PRIMARY_SEED = 0

# ----------------------------------------------------------------------
# Splits  (Sec. 5 of the brief: 60 / 20 / 20, train / calibration / test)
# ----------------------------------------------------------------------
TRAIN_FRAC = 0.60
CAL_FRAC = 0.20
TEST_FRAC = 0.20

# ----------------------------------------------------------------------
# Preprocessing
# ----------------------------------------------------------------------
# discharge_disposition_id values meaning "died" or "discharged to hospice".
# These patients cannot be readmitted -> structural negatives.
EXPIRED_HOSPICE_IDS = {11, 13, 14, 19, 20, 21}

DROP_HIGH_MISSING = ["weight", "payer_code"]     # ~97% and ~40% missing
IDENTIFIER_COLS = ["encounter_id", "patient_nbr"]

# Sensitive attributes are KEPT as model features. Rationale for the paper:
# the calibration gap we report arises *despite* the model having access to
# group membership, which is strictly stronger than the blinded case.
INCLUDE_SENSITIVE_AS_FEATURES = True

SENSITIVE_ATTRS = ["race", "age"]        # attributes the Gap is computed over
GENDER_ATTR = "gender"                    # reported, usually only 2 groups

# 'Unknown' race is not a demographic group; it is reported in the support
# table but excluded from the Calibration Gap by default.
EXCLUDE_LEVELS = {"race": ["Unknown"]}

# ----------------------------------------------------------------------
# Subgroup minimum-support rule  (Sec. 3 of the brief)
# ----------------------------------------------------------------------
MIN_GROUP_N = 200          # minimum test-set members
MIN_GROUP_POS = 25         # minimum positive events

# Support rule for *fitting* a group-wise calibrator (calibration split)
MIN_CAL_N = 200
MIN_CAL_POS = 25

# ----------------------------------------------------------------------
# Calibration metric settings  (Trap 3: quantile bins, not equal-width)
# ----------------------------------------------------------------------
ECE_BINS_PRIMARY = 10
ECE_BINS_ROBUST = 15
ECE_STRATEGY = "quantile"   # equal-mass bins

N_BOOTSTRAP = 1000
BOOTSTRAP_ALPHA = 0.05      # -> 95% percentile CI

# ----------------------------------------------------------------------
# Models
# ----------------------------------------------------------------------
RF_PARAMS = dict(n_estimators=300, max_depth=12, min_samples_leaf=20,
                 n_jobs=-1, random_state=PRIMARY_SEED)
HGB_PARAMS = dict(max_iter=300, learning_rate=0.06, max_leaf_nodes=31,
                  min_samples_leaf=20, early_stopping=False,
                  random_state=PRIMARY_SEED)
LR_PARAMS = dict(max_iter=2000, C=1.0, solver="lbfgs")

IMBALANCE_VARIANTS = [None, "balanced"]   # class_weight settings

CALIBRATION_ARMS = ["raw", "platt_global", "iso_global",
                    "platt_group", "iso_group"]

# ----------------------------------------------------------------------
# SHAP  (Sec. 5 of the brief)
# ----------------------------------------------------------------------
SHAP_TREE_ROWS = 2000       # rows for TreeExplainer on the base model
SHAP_PERM_ROWS = 500        # rows for model-agnostic explainer
SHAP_BACKGROUND = 100       # background/masker rows
SHAP_HEADLINE_MODEL = "HGB" # model used for C3 figures

# ----------------------------------------------------------------------
# Figures
# ----------------------------------------------------------------------
DPI = 300
FIG_FORMATS = ["pdf", "png"]   # PDF is the vector format IEEE wants
