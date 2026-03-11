"""
config.py - All configuration constants and hyperparameters.

Every tunable parameter is set here. Modules receive configuration
via the Config dataclass or by importing individual constants.
"""

from dataclasses import dataclass, field
from typing import List

# ──────────────────────────────────────────
# PATHS
# ──────────────────────────────────────────
DATA_BASE_PATH = "../../Data/"
DATA_TESTING_PATH = f"{DATA_BASE_PATH}PHM2025_test_data/"
DATA_TRAINING_PATH = f"{DATA_BASE_PATH}PHM2025_training_data/"
DATA_VALIDATION_PATH = f"{DATA_BASE_PATH}PHM2025_validation_data/"
DATA_TRAINING_DATA = f"{DATA_TRAINING_PATH}training_data.csv"
PLOT_PATH = "./img/"
MODELS_DIR = "./saved_models"
SUBMISSION_OUTPUT = "../../submission.csv"


# ──────────────────────────────────────────
# SENSORS
# ──────────────────────────────────────────
SENSORS: List[str] = [
    "Sensed_Altitude",
    "Sensed_Mach",
    "Sensed_Pamb",
    "Sensed_Pt2",
    "Sensed_TAT",
    "Sensed_WFuel",
    "Sensed_VAFN",
    "Sensed_VBV",
    "Sensed_Fan_Speed",
    "Sensed_Core_Speed",
    "Sensed_T25",
    "Sensed_T3",
    "Sensed_Ps3",
    "Sensed_T45",
    "Sensed_P25",
    "Sensed_T5",
]

OPERATING_VARS: List[str] = [
    "Sensed_Altitude",
    "Sensed_Mach",
    "Sensed_Pamb",
    "Sensed_TAT",
    "Sensed_VAFN",
    "Sensed_VBV",
    "Sensed_Fan_Speed",
    "Sensed_Pt2",
]

# Degradation vars = SENSORS minus OPERATING_VARS minus excluded sensors
_EXCLUDED_SENSORS = {"Sensed_P25", "Sensed_T5"}
DEGRAD_VARS: List[str] = [
    s for s in SENSORS
    if s not in OPERATING_VARS and s not in _EXCLUDED_SENSORS
]

ALL_VARS: List[str] = OPERATING_VARS + DEGRAD_VARS


# ──────────────────────────────────────────
# LINEAR REGRESSOR (nominal behaviour model)
# ──────────────────────────────────────────
TESTING_ESN: int = 103
INCLUDE_TEST: bool = False
CYCLES_HEALTHY: int = 5
AUGMENTED_DATA: bool = False
AUGMENTED_COUNT: int = 100
SMOTE: bool = False
ENSAMBLE: bool = True
SEPARATE_MODELS: bool = True


# ──────────────────────────────────────────
# HI COEFFICIENT SEARCH (differential_evolution)
# ──────────────────────────────────────────
DO_NOT_TRAIN_COEFS: bool = False
USE_ALL_VARS: bool = False
THIS_ALL_VARS: List[str] = ["Sensed_T3", "Sensed_T45", "Sensed_Core_Speed", "Sensed_T25"]

DE_MAXITER: int = 800
DE_POPSIZE: int = 200
DE_TOL: float = 0.001

USE_ONLY_TRAIN: bool = False
USE_CLEAN_DATA: bool = True
COEF_OUTLIERS_THRESHOLD: float = 3
SEPARATE_COEFS: bool = False

# Default coefficients (when DO_NOT_TRAIN_COEFS=True)
DEFAULT_CHPT: dict = {
    "101": -2.23025081,
    "102": -2.21770178,
    "103": -1.86356251,
    "104": -2.40613967,
}
DEFAULT_CHPC: dict = {
    "101": 4.14196053,
    "102": 3.40982944,
    "103": 4.72156249,
    "104": 3.80263469,
}


# ──────────────────────────────────────────
# LGBM CLASSIFIER (cycle classification)
# ──────────────────────────────────────────
CLF_N_ESTIMATORS: int = 600
CLF_LEARNING_RATE: float = 0.002
CLF_MAX_DEPTH: int = 10
CLF_NUM_LEAVES: int = 63
CLF_WINDOW: int = 20


# ──────────────────────────────────────────
# LGBM GAP CORRECTION
# ──────────────────────────────────────────
GAP_FEATURE_WINDOW: int = 20
GAP_LGBM_PARAMS: dict = {
    "objective": "regression",
    "metric": "rmse",
    "n_estimators": 5000,
    "learning_rate": 0.002,
    "max_depth": 12,
    "num_leaves": 63,
    "min_child_samples": 5,
    "reg_alpha": 0.1,
    "reg_lambda": 0.1,
    "n_jobs": -1,
    "verbose": -1,
    "random_state": 42,
}
SMOOTHING_WINDOW: int = 10


# ──────────────────────────────────────────
# WW PREDICTION
# ──────────────────────────────────────────
WW_DETECTION_WINDOW: int = 11
WW_FACTOR_MULT: int = 80


# ──────────────────────────────────────────
# PLOTTING
# ──────────────────────────────────────────
PLOTS_DIR: str = "./plots/"
PLOT_GROUP_CYCLES: bool = True
PLOT_REMOVE_OUTLIERS: bool = True
PLOT_OUTLIERS_THRESHOLD: float = 3
