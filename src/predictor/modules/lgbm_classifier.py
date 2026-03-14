"""
lgbm_classifier.py - LGBMCycleClassifier class.

Classifies the current maintenance cycle (how many service visits have
already occurred) for HPT and HPC using LightGBM classifiers.

Supports: train, save/load, predict, and plotting.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import lightgbm as lgbm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score

from modules import config as cfg, save_fig
from modules.data import Data
from modules.hi_trainer import HITrainer


class LGBMCycleClassifier:
    """LightGBM-based classifier for maintenance-cycle prediction.

    Determines how many HPT/HPC service visits have already been
    performed, so that downstream scale_to_target uses the correct
    reference range.

    Parameters
    ----------
    hi_trainer : HITrainer
        Trained HITrainer with coefficients and linear models.
    """

    def __init__(self, hi_trainer: HITrainer) -> None:
        self.hi = hi_trainer

        # Trained LightGBM classifiers
        self.clf_hpt: lgbm.LGBMClassifier | None = None
        self.clf_hpc: lgbm.LGBMClassifier | None = None

        # Feature column names (order matters for prediction)
        self.feature_cols: list[str] = []

        # Scale coefficients for classifier-input HI
        self.scale_coefs_hpt: dict[str, float] = {}
        self.scale_coefs_hpc: dict[str, float] = {}

    @staticmethod
    def _split_train_valid_by_esn(
        clf_data: pd.DataFrame,
    ) -> tuple[pd.DataFrame, pd.DataFrame | None]:
        """Deterministic ESN-level train/validation split for early-stopping."""
        if not cfg.LGBM_USE_EARLY_STOPPING:
            return clf_data, None

        esn_values = sorted(clf_data["ESN"].dropna().unique())
        if len(esn_values) < 3:
            return clf_data, None

        n_valid = int(np.ceil(len(esn_values) * cfg.LGBM_VALID_FRACTION_BY_ESN))
        n_valid = max(1, min(n_valid, len(esn_values) - 1))

        valid_esn = set(esn_values[-n_valid:])
        valid_df = clf_data[clf_data["ESN"].isin(valid_esn)].copy()
        train_df = clf_data[~clf_data["ESN"].isin(valid_esn)].copy()

        if train_df.empty or valid_df.empty:
            return clf_data, None
        return train_df, valid_df

    # ════════════════════════════════════════════════════════════════
    #  FEATURE BUILDING
    # ════════════════════════════════════════════════════════════════

    def build_features(
        self,
        data: pd.DataFrame,
        window: int = cfg.CLF_WINDOW,
    ) -> pd.DataFrame:
        """Build classification features for every ESN in *data*.

        Returns DataFrame with feature columns, ESN, and label columns.
        """
        feat_list: list[pd.DataFrame] = []

        for esn in data["ESN"].unique():
            sd = data[data["ESN"] == esn].copy()
            ahpt, ahpc = self.hi.get_coefs_for_esn(esn)
            hi_hpt, hi_hpc = self.hi.calc_hi(sd, ahpt, ahpc)

            # Scale HI with classifier-level coefficients
            hi_hpt_s = HITrainer.scale_to_target_test(hi_hpt, self.scale_coefs_hpt)
            hi_hpc_s = HITrainer.scale_to_target_test(hi_hpc, self.scale_coefs_hpc)

            feat = sd[cfg.DEGRAD_VARS].copy()
            feat["HI_HPT"] = hi_hpt_s.values
            feat["HI_HPC"] = hi_hpc_s.values
            feat["HI_HPT_slope"] = (
                hi_hpt_s.rolling(window=window, min_periods=1)
                .apply(Data.get_slope, raw=True)
                .values
            )
            feat["HI_HPC_slope"] = (
                hi_hpc_s.rolling(window=window, min_periods=1)
                .apply(Data.get_slope, raw=True)
                .values
            )
            feat["HI_HPT_rolling_mean"] = (
                hi_hpt_s.rolling(window=window, min_periods=1).mean().values
            )
            feat["HI_HPC_rolling_mean"] = (
                hi_hpc_s.rolling(window=window, min_periods=1).mean().values
            )
            feat["ESN"] = esn
            feat["label_hpt"] = sd["Cumulative_HPT_SVs"].values
            feat["label_hpc"] = sd["Cumulative_HPC_SVs"].values
            feat_list.append(feat)

        return pd.concat(feat_list, ignore_index=True)

    # ════════════════════════════════════════════════════════════════
    #  TRAIN
    # ════════════════════════════════════════════════════════════════

    def train(self, coef_data: pd.DataFrame | None = None) -> None:
        """Train HPT and HPC cycle classifiers on *coef_data*.

        If *coef_data* is ``None``, uses ``hi_trainer.coef_data``.
        """
        if coef_data is None:
            coef_data = self.hi.coef_data
        if coef_data is None:
            raise ValueError(
                "No coef_data available. Run HITrainer.train_coefficients() first."
            )

        # Pre-compute classifier-level scale coefficients
        self.scale_coefs_hpt = {
            "min": float(coef_data["Cycles_to_HPT_SV"].min()),
            "max": float(coef_data["Cycles_to_HPT_SV"].max()),
        }
        self.scale_coefs_hpc = {
            "min": float(coef_data["Cycles_to_HPC_SV"].min()),
            "max": float(coef_data["Cycles_to_HPC_SV"].max()),
        }

        clf_data = self.build_features(coef_data)
        self.feature_cols = [
            c for c in clf_data.columns if c not in ("ESN", "label_hpt", "label_hpc")
        ]

        train_df, valid_df = self._split_train_valid_by_esn(clf_data)

        X_train = train_df[self.feature_cols].values
        y_hpt_train = train_df["label_hpt"].values.astype(int)
        y_hpc_train = train_df["label_hpc"].values.astype(int)

        X_valid = None
        y_hpt_valid = None
        y_hpc_valid = None
        if valid_df is not None:
            X_valid = valid_df[self.feature_cols].values
            y_hpt_valid = valid_df["label_hpt"].values.astype(int)
            y_hpc_valid = valid_df["label_hpc"].values.astype(int)

        self.clf_hpt = lgbm.LGBMClassifier(
            objective="multiclass",
            n_estimators=cfg.CLF_N_ESTIMATORS,
            learning_rate=cfg.CLF_LEARNING_RATE,
            max_depth=cfg.CLF_MAX_DEPTH,
            num_leaves=cfg.CLF_NUM_LEAVES,
            n_jobs=-1,
            verbose=-1,
            random_state=42,
        )
        self.clf_hpc = lgbm.LGBMClassifier(
            objective="multiclass",
            n_estimators=cfg.CLF_N_ESTIMATORS,
            learning_rate=cfg.CLF_LEARNING_RATE,
            max_depth=cfg.CLF_MAX_DEPTH,
            num_leaves=cfg.CLF_NUM_LEAVES,
            n_jobs=-1,
            verbose=-1,
            random_state=42,
        )

        fit_kwargs_hpt: dict[str, Any] = {}
        fit_kwargs_hpc: dict[str, Any] = {}
        if X_valid is not None:
            callbacks = [
                lgbm.early_stopping(
                    cfg.LGBM_EARLY_STOPPING_ROUNDS,
                    verbose=False,
                )
            ]
            fit_kwargs_hpt = {
                "eval_set": [(X_valid, y_hpt_valid)],
                "eval_metric": "multi_logloss",
                "callbacks": callbacks,
            }
            fit_kwargs_hpc = {
                "eval_set": [(X_valid, y_hpc_valid)],
                "eval_metric": "multi_logloss",
                "callbacks": callbacks,
            }

        self.clf_hpt.fit(X_train, y_hpt_train, **fit_kwargs_hpt)
        self.clf_hpc.fit(X_train, y_hpc_train, **fit_kwargs_hpc)

        pred_hpt_train = self.clf_hpt.predict(X_train)
        pred_hpc_train = self.clf_hpc.predict(X_train)
        acc_hpt = accuracy_score(y_hpt_train, pred_hpt_train)
        acc_hpc = accuracy_score(y_hpc_train, pred_hpc_train)

        valid_msg = ""
        if X_valid is not None:
            pred_hpt_valid = self.clf_hpt.predict(X_valid)
            pred_hpc_valid = self.clf_hpc.predict(X_valid)
            val_acc_hpt = accuracy_score(y_hpt_valid, pred_hpt_valid)
            val_acc_hpc = accuracy_score(y_hpc_valid, pred_hpc_valid)
            valid_msg = (
                f" | Validation accuracy: HPT={val_acc_hpt:.4f}, "
                f"HPC={val_acc_hpc:.4f}"
            )

        print(
            f"Classifiers trained. "
            f"In-sample accuracy: HPT={acc_hpt:.4f}, HPC={acc_hpc:.4f}"
            f"{valid_msg}"
        )
        print(f"Features: {self.feature_cols}")
        print(f"HPT classes: {sorted(set(y_hpt_train))}")
        print(f"HPC classes: {sorted(set(y_hpc_train))}")

    # ════════════════════════════════════════════════════════════════
    #  PREDICT
    # ════════════════════════════════════════════════════════════════

    def predict(
        self,
        engine_residuals: pd.DataFrame,
        esn: Any,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Predict cycle series for a single engine.

        Returns
        -------
        (cycle_hpt_series, cycle_hpc_series) : tuple[np.ndarray, np.ndarray]
        """
        if self.clf_hpt is None or self.clf_hpc is None:
            raise RuntimeError("Classifiers not trained. Call train() first.")

        sd = engine_residuals.copy()
        ahpt, ahpc = self.hi.get_coefs_for_esn(esn)
        hi_hpt, hi_hpc = self.hi.calc_hi(sd, ahpt, ahpc)

        hi_hpt_clf = HITrainer.scale_to_target_test(hi_hpt, self.scale_coefs_hpt)
        hi_hpc_clf = HITrainer.scale_to_target_test(hi_hpc, self.scale_coefs_hpc)

        window = cfg.CLF_WINDOW
        clf_feat = sd[cfg.DEGRAD_VARS].copy()
        clf_feat["HI_HPT"] = hi_hpt_clf.values
        clf_feat["HI_HPC"] = hi_hpc_clf.values
        clf_feat["HI_HPT_slope"] = (
            hi_hpt_clf.rolling(window=window, min_periods=1)
            .apply(Data.get_slope, raw=True)
            .values
        )
        clf_feat["HI_HPC_slope"] = (
            hi_hpc_clf.rolling(window=window, min_periods=1)
            .apply(Data.get_slope, raw=True)
            .values
        )
        clf_feat["HI_HPT_rolling_mean"] = (
            hi_hpt_clf.rolling(window=window, min_periods=1).mean().values
        )
        clf_feat["HI_HPC_rolling_mean"] = (
            hi_hpc_clf.rolling(window=window, min_periods=1).mean().values
        )

        try:
            X = clf_feat[self.feature_cols].values
            return self.clf_hpt.predict(X), self.clf_hpc.predict(X)
        except Exception as exc:
            raise RuntimeError(
                f"Cycle classifier prediction failed for ESN {esn}: {exc}"
            ) from exc

    # ════════════════════════════════════════════════════════════════
    #  SAVE / LOAD
    # ════════════════════════════════════════════════════════════════

    def save(self, directory: str = cfg.MODELS_DIR) -> None:
        """Persist classifiers and metadata to disk."""
        Path(directory).mkdir(parents=True, exist_ok=True)
        joblib.dump(self.clf_hpt, f"{directory}/clf_hpt.pkl")
        joblib.dump(self.clf_hpc, f"{directory}/clf_hpc.pkl")
        joblib.dump(self.feature_cols, f"{directory}/clf_feature_cols.pkl")
        joblib.dump(self.scale_coefs_hpt, f"{directory}/clf_scale_coefs_hpt.pkl")
        joblib.dump(self.scale_coefs_hpc, f"{directory}/clf_scale_coefs_hpc.pkl")
        print(f"Classifiers saved to {directory}/")

    def load(self, directory: str = cfg.MODELS_DIR) -> None:
        """Load classifiers and metadata from disk."""
        self.clf_hpt = joblib.load(f"{directory}/clf_hpt.pkl")
        self.clf_hpc = joblib.load(f"{directory}/clf_hpc.pkl")
        self.feature_cols = joblib.load(f"{directory}/clf_feature_cols.pkl")
        self.scale_coefs_hpt = joblib.load(f"{directory}/clf_scale_coefs_hpt.pkl")
        self.scale_coefs_hpc = joblib.load(f"{directory}/clf_scale_coefs_hpc.pkl")
        print(f"Classifiers loaded from {directory}/")

    # ════════════════════════════════════════════════════════════════
    #  PLOTTING
    # ════════════════════════════════════════════════════════════════

    def plot_cycle_distribution(self, results_df: pd.DataFrame) -> None:
        """Bar chart of predicted cycle counts for HPT and HPC."""
        if "HPT_cycle" not in results_df.columns:
            print("No cycle columns in results.")
            return

        fig, ax = plt.subplots(figsize=(8, 5))
        fig.suptitle("Predicted Maintenance Cycles", fontsize=14)

        cycles_hpt = results_df["HPT_cycle"].value_counts().sort_index()
        cycles_hpc = results_df["HPC_cycle"].value_counts().sort_index()
        width = 0.35
        x_hpt = np.arange(len(cycles_hpt))
        x_hpc = np.arange(len(cycles_hpc))

        ax.bar(
            x_hpt - width / 2,
            cycles_hpt.values,
            width,
            label="HPT cycle",
            color="tab:blue",
            alpha=0.7,
        )
        ax.bar(
            x_hpc + width / 2,
            cycles_hpc.values,
            width,
            label="HPC cycle",
            color="tab:green",
            alpha=0.7,
        )
        ax.set_xticks(np.arange(max(len(cycles_hpt), len(cycles_hpc))))
        ax.set_xlabel("Cycle")
        ax.set_ylabel("Count")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        save_fig(fig, "cycle_distribution")
