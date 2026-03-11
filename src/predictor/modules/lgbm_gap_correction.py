"""
lgbm_gap_correction.py - LGBMGapCorrection class.

Learns the residual gap between scale_to_target(HI) predictions and
ground-truth Cycles_to_SV, then corrects future predictions.

Supports: feature building, train, save/load, predict, and the full
``predict_cycles_to_sv_v2`` pipeline used for inference on val/test.
Also contains all associated plotting methods.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import lightgbm as lgbm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import LeaveOneGroupOut

from modules import config as cfg, save_fig
from modules.data import Data
from modules.hi_trainer import HITrainer
from modules.lgbm_classifier import LGBMCycleClassifier


class LGBMGapCorrection:
    """LightGBM-based gap correction for Cycles_to_SV prediction.

    Pipeline:
        1. ``scale_to_target(HI)`` → base prediction
        2. LightGBM learns ``gap = ground_truth - base``
        3. Final prediction = ``base + predicted_gap``

    Parameters
    ----------
    hi_trainer : HITrainer
        Trained HITrainer.
    classifier : LGBMCycleClassifier
        Trained cycle classifier.
    """

    def __init__(
        self,
        hi_trainer: HITrainer,
        classifier: LGBMCycleClassifier,
    ) -> None:
        self.hi = hi_trainer
        self.clf = classifier

        # Trained LGBM gap regressors
        self.lgbm_gap_hpt: lgbm.LGBMRegressor | None = None
        self.lgbm_gap_hpc: lgbm.LGBMRegressor | None = None

        # Feature column order
        self.feature_cols: list[str] = []

        # scale_to_target coefficients per maintenance cycle
        self.scale_coefs_hpt: dict[int, dict[str, float]] = {}
        self.scale_coefs_hpc: dict[int, dict[str, float]] = {}

        # Training-time base predictions (for plotting)
        self._base_pred_hpt: np.ndarray | None = None
        self._base_pred_hpc: np.ndarray | None = None
        self._regs_data: pd.DataFrame | None = None

    # ════════════════════════════════════════════════════════════════
    #  REGRESSION FEATURE BUILDING
    # ════════════════════════════════════════════════════════════════

    def _build_features_for_esn(
        self,
        sd: pd.DataFrame,
        esn: Any,
        window: int = cfg.GAP_FEATURE_WINDOW,
    ) -> pd.DataFrame:
        """Build rich regression features for a single ESN block."""
        sd = sd.copy().sort_values("Cycles_Since_New")
        ahpt, ahpc = self.hi.get_coefs_for_esn(esn)
        hi_hpt, hi_hpc = self.hi.calc_hi(sd, ahpt, ahpc)

        feat = sd[cfg.DEGRAD_VARS].copy()

        # ── Base HI features ────────────────────────────────────
        feat["HI_HPT"] = hi_hpt.values
        feat["HI_HPC"] = hi_hpc.values

        # ── Multi-scale trend features ──────────────────────────
        for w in [10, 20, 50]:
            feat[f"HI_HPT_slope_{w}"] = (
                hi_hpt.rolling(window=w, min_periods=1)
                .apply(Data.get_slope, raw=True)
                .values
            )
            feat[f"HI_HPC_slope_{w}"] = (
                hi_hpc.rolling(window=w, min_periods=1)
                .apply(Data.get_slope, raw=True)
                .values
            )
            feat[f"HI_HPT_mean_{w}"] = (
                hi_hpt.rolling(window=w, min_periods=1).mean().values
            )
            feat[f"HI_HPC_mean_{w}"] = (
                hi_hpc.rolling(window=w, min_periods=1).mean().values
            )
            feat[f"HI_HPT_std_{w}"] = (
                hi_hpt.rolling(window=w, min_periods=2).std().fillna(0).values
            )
            feat[f"HI_HPC_std_{w}"] = (
                hi_hpc.rolling(window=w, min_periods=2).std().fillna(0).values
            )

        # ── Position / degradation features ─────────────────────
        feat["HI_HPT_delta_from_start"] = hi_hpt.values - hi_hpt.values[0]
        feat["HI_HPC_delta_from_start"] = hi_hpc.values - hi_hpc.values[0]
        feat["HI_HPT_cumulative_change"] = hi_hpt.diff().fillna(0).cumsum().values
        feat["HI_HPC_cumulative_change"] = hi_hpc.diff().fillna(0).cumsum().values

        exp_mean_hpt = hi_hpt.expanding(min_periods=1).mean()
        exp_mean_hpc = hi_hpc.expanding(min_periods=1).mean()
        feat["HI_HPT_ratio_to_hist"] = (hi_hpt.values / exp_mean_hpt.values).clip(
            -10, 10
        )
        feat["HI_HPC_ratio_to_hist"] = (hi_hpc.values / exp_mean_hpc.values).clip(
            -10, 10
        )

        for w in [20, 50]:
            feat[f"HI_HPT_range_{w}"] = (
                hi_hpt.rolling(w, min_periods=1).max().values
                - hi_hpt.rolling(w, min_periods=1).min().values
            )
            feat[f"HI_HPC_range_{w}"] = (
                hi_hpc.rolling(w, min_periods=1).max().values
                - hi_hpc.rolling(w, min_periods=1).min().values
            )

        # ── Acceleration ────────────────────────────────────────
        slope_hpt = hi_hpt.rolling(window=20, min_periods=1).apply(
            Data.get_slope, raw=True
        )
        slope_hpc = hi_hpc.rolling(window=20, min_periods=1).apply(
            Data.get_slope, raw=True
        )
        feat["HI_HPT_acceleration"] = slope_hpt.diff().fillna(0).values
        feat["HI_HPC_acceleration"] = slope_hpc.diff().fillna(0).values

        # ── Sensor-level slopes ─────────────────────────────────
        for var in cfg.DEGRAD_VARS:
            feat[f"{var}_slope_20"] = (
                sd[var]
                .rolling(window=20, min_periods=1)
                .apply(Data.get_slope, raw=True)
                .values
            )

        return feat

    def build_regression_features(
        self,
        coef_data: pd.DataFrame,
        window: int = cfg.GAP_FEATURE_WINDOW,
    ) -> pd.DataFrame:
        """Build regression features for all ESNs in *coef_data*."""
        feat_list: list[pd.DataFrame] = []
        for esn in coef_data["ESN"].unique():
            sd = coef_data[coef_data["ESN"] == esn].copy()
            feat = self._build_features_for_esn(sd, esn, window)
            feat["cycle_hpt"] = sd["Cumulative_HPT_SVs"].values
            feat["cycle_hpc"] = sd["Cumulative_HPC_SVs"].values
            feat["ESN"] = esn
            feat["target_hpt"] = sd["Cycles_to_HPT_SV"].values
            feat["target_hpc"] = sd["Cycles_to_HPC_SV"].values
            feat_list.append(feat)
        return pd.concat(feat_list, ignore_index=True)

    # ════════════════════════════════════════════════════════════════
    #  TRAIN
    # ════════════════════════════════════════════════════════════════

    def train(self, coef_data: pd.DataFrame | None = None) -> None:
        """Train LightGBM gap-correction regressors.

        Includes leave-one-engine-out validation and a final fit on all data.
        """
        if coef_data is None:
            coef_data = self.hi.coef_data
        if coef_data is None:
            raise ValueError("No coef_data. Run HITrainer.train_coefficients().")

        regs_data = self.build_regression_features(coef_data)
        self.feature_cols = [
            c for c in regs_data.columns if c not in ("ESN", "target_hpt", "target_hpc")
        ]
        self._regs_data = regs_data
        print(f"Gap correction features: {len(self.feature_cols)}")

        y_true_hpt = regs_data["target_hpt"].values
        y_true_hpc = regs_data["target_hpc"].values

        # ── Scale to target per maintenance cycle ────────────
        base_pred_hpt = np.full(len(regs_data), np.nan)
        base_pred_hpc = np.full(len(regs_data), np.nan)

        for cycle in sorted(regs_data["cycle_hpt"].unique()):
            mask = regs_data["cycle_hpt"] == cycle
            coefs: dict = {}
            scaled = HITrainer.scale_to_target(
                regs_data.loc[mask, "HI_HPT"],
                regs_data.loc[mask, "target_hpt"],
                coefs,
            )
            base_pred_hpt[mask.values] = scaled.values
            self.scale_coefs_hpt[int(cycle)] = {
                "min": coefs["min"][0],
                "max": coefs["max"][0],
            }

        for cycle in sorted(regs_data["cycle_hpc"].unique()):
            mask = regs_data["cycle_hpc"] == cycle
            coefs = {}
            scaled = HITrainer.scale_to_target(
                regs_data.loc[mask, "HI_HPC"],
                regs_data.loc[mask, "target_hpc"],
                coefs,
            )
            base_pred_hpc[mask.values] = scaled.values
            self.scale_coefs_hpc[int(cycle)] = {
                "min": coefs["min"][0],
                "max": coefs["max"][0],
            }

        self._base_pred_hpt = base_pred_hpt
        self._base_pred_hpc = base_pred_hpc

        rmse_hpt = np.sqrt(np.nanmean((y_true_hpt - base_pred_hpt) ** 2))
        rmse_hpc = np.sqrt(np.nanmean((y_true_hpc - base_pred_hpc) ** 2))
        print(f"Base HPT (scale_to_target) RMSE: {rmse_hpt:.2f}")
        print(f"Base HPC (scale_to_target) RMSE: {rmse_hpc:.2f}")

        # ── Gap = ground_truth - base ────────────────────────
        gap_hpt = y_true_hpt - base_pred_hpt
        gap_hpc = y_true_hpc - base_pred_hpc

        X_reg = regs_data[self.feature_cols].values
        groups = regs_data["ESN"].values

        self.lgbm_gap_hpt = lgbm.LGBMRegressor(**cfg.GAP_LGBM_PARAMS)
        self.lgbm_gap_hpc = lgbm.LGBMRegressor(**cfg.GAP_LGBM_PARAMS)

        # ── Leave-one-engine-out validation ──────────────────
        logo = LeaveOneGroupOut()
        print("\n=== Gap Correction — Leave-One-Engine-Out ===")
        for train_idx, test_idx in logo.split(X_reg, gap_hpt, groups):
            test_esn = groups[test_idx[0]]

            self.lgbm_gap_hpt.fit(X_reg[train_idx], gap_hpt[train_idx])
            gap_pred_hpt = self.lgbm_gap_hpt.predict(X_reg[test_idx])
            final_hpt = base_pred_hpt[test_idx] + gap_pred_hpt
            rmse_hpt_ = np.sqrt(np.mean((y_true_hpt[test_idx] - final_hpt) ** 2))

            self.lgbm_gap_hpc.fit(X_reg[train_idx], gap_hpc[train_idx])
            gap_pred_hpc = self.lgbm_gap_hpc.predict(X_reg[test_idx])
            final_hpc = base_pred_hpc[test_idx] + gap_pred_hpc
            rmse_hpc_ = np.sqrt(np.mean((y_true_hpc[test_idx] - final_hpc) ** 2))

            base_rmse_hpt = np.sqrt(
                np.mean((y_true_hpt[test_idx] - base_pred_hpt[test_idx]) ** 2)
            )
            base_rmse_hpc = np.sqrt(
                np.mean((y_true_hpc[test_idx] - base_pred_hpc[test_idx]) ** 2)
            )
            print(f"ESN {test_esn}:")
            print(
                f"  HPT: base RMSE={base_rmse_hpt:.2f} → corrected RMSE={rmse_hpt_:.2f}"
            )
            print(
                f"  HPC: base RMSE={base_rmse_hpc:.2f} → corrected RMSE={rmse_hpc_:.2f}"
            )

        # ── Final fit on all data ────────────────────────────
        self.lgbm_gap_hpt.fit(X_reg, gap_hpt)
        self.lgbm_gap_hpc.fit(X_reg, gap_hpc)
        print("\nLGBM gap regressors trained on all engines.")

    # ════════════════════════════════════════════════════════════════
    #  PREDICT — single engine (v2 pipeline)
    # ════════════════════════════════════════════════════════════════

    def predict_engine(
        self,
        engine_df: pd.DataFrame,
        engine_residuals: pd.DataFrame,
        esn: Any,
    ) -> dict[str, Any]:
        """Full v2 prediction pipeline for one engine.

        Returns dict with ESN, Cycles_to_HPT_SV, Cycles_to_HPC_SV,
        HPT_cycle, HPC_cycle, and full prediction series.
        """
        sd = engine_df.copy()
        sd[cfg.DEGRAD_VARS] = engine_residuals[cfg.DEGRAD_VARS].values

        if "Cycles" in sd.columns and "Cycles_Since_New" not in sd.columns:
            sd = sd.rename(columns={"Cycles": "Cycles_Since_New"})
        sd = sd.sort_values("Cycles_Since_New")

        # Build gap-regression features
        feat = self._build_features_for_esn(sd, esn)

        # Classify maintenance cycle
        cycle_hpt_s, cycle_hpc_s = self.clf.predict(sd, esn)
        cycle_hpt = int(cycle_hpt_s[-1])
        cycle_hpc = int(cycle_hpc_s[-1])

        feat["cycle_hpt"] = cycle_hpt_s
        feat["cycle_hpc"] = cycle_hpc_s

        X_feat = feat[self.feature_cols].values

        # scale_to_target base prediction
        ahpt, ahpc = self.hi.get_coefs_for_esn(esn)
        hi_hpt, hi_hpc = self.hi.calc_hi(sd, ahpt, ahpc)

        hpt_key = (
            cycle_hpt
            if cycle_hpt in self.scale_coefs_hpt
            else min(self.scale_coefs_hpt.keys())
        )
        hpc_key = (
            cycle_hpc
            if cycle_hpc in self.scale_coefs_hpc
            else min(self.scale_coefs_hpc.keys())
        )

        base_hpt = HITrainer.scale_to_target_test(hi_hpt, self.scale_coefs_hpt[hpt_key])
        base_hpc = HITrainer.scale_to_target_test(hi_hpc, self.scale_coefs_hpc[hpc_key])
        # Gap correction
        gap_hpt = self.lgbm_gap_hpt.predict(X_feat)
        gap_hpc = self.lgbm_gap_hpc.predict(X_feat)

        pred_hpt = np.clip(base_hpt.values + gap_hpt, 0, None)
        pred_hpc = np.clip(base_hpc.values + gap_hpc, 0, None)

        # Smoothing — ONLY here, right before extracting the final value
        if cfg.SMOOTH_PREDICTIONS:
            pred_hpt = (
                pd.Series(pred_hpt)
                .rolling(window=cfg.SMOOTHING_WINDOW, min_periods=1)
                .mean()
                .values
            )
            pred_hpc = (
                pd.Series(pred_hpc)
                .rolling(window=cfg.SMOOTHING_WINDOW, min_periods=1)
                .mean()
                .values
            )

        return {
            "ESN": esn,
            "Cycles_to_HPT_SV": pred_hpt[-1],
            "Cycles_to_HPC_SV": pred_hpc[-1],
            "HPT_cycle": cycle_hpt,
            "HPC_cycle": cycle_hpc,
            "pred_series_hpt": pred_hpt,
            "pred_series_hpc": pred_hpc,
        }

    # ════════════════════════════════════════════════════════════════
    #  PREDICT — all engines
    # ════════════════════════════════════════════════════════════════

    def predict_all_engines(
        self,
        engine_list: list[pd.DataFrame],
    ) -> pd.DataFrame:
        """Predict Cycles_to_SV for every engine in a list of DataFrames.

        Returns a summary DataFrame with one row per ESN.
        """
        if self._regs_data is None:
            raise RuntimeError("Not trained yet. Call train() first.")

        train_max_hpt = self._regs_data["target_hpt"].max()
        train_max_hpc = self._regs_data["target_hpc"].max()
        train_mean_hpt = self._regs_data["target_hpt"].mean()
        train_mean_hpc = self._regs_data["target_hpc"].mean()

        clip_max_hpt = train_max_hpt * 1.3
        clip_max_hpc = train_max_hpc * 1.3

        print(
            f"Training range HPT: 0 — {train_max_hpt:.0f} (mean={train_mean_hpt:.0f})"
        )
        print(
            f"Training range HPC: 0 — {train_max_hpc:.0f} (mean={train_mean_hpc:.0f})"
        )

        results: list[dict] = []

        for file_idx, engine_df in enumerate(engine_list):
            for esn in engine_df["ESN"].unique():
                edf = engine_df[engine_df["ESN"] == esn].copy()
                engine_res = self.hi.residuals_single(edf)
                if engine_res is None:
                    print(f"  ESN {esn}: SKIP (residuals None)")
                    fb = self._fallback_result(
                        esn, train_mean_hpt, train_mean_hpc, "fallback"
                    )
                    fb["file_idx"] = file_idx
                    results.append(fb)
                    continue
                try:
                    pred = self.predict_engine(edf, engine_res, esn)
                    pred_hpt = np.clip(pred["Cycles_to_HPT_SV"], 0, clip_max_hpt)
                    pred_hpc = np.clip(pred["Cycles_to_HPC_SV"], 0, clip_max_hpc)
                    results.append(
                        {
                            "file_idx": file_idx,
                            "ESN": esn,
                            "Cycles_to_HPT_SV": pred_hpt,
                            "Cycles_to_HPC_SV": pred_hpc,
                            "HPT_cycle": pred["HPT_cycle"],
                            "HPC_cycle": pred["HPC_cycle"],
                            "pred_series_hpt": pred["pred_series_hpt"],
                            "pred_series_hpc": pred["pred_series_hpc"],
                            "confidence": "ok",
                        }
                    )
                except Exception as ex:
                    print(f"  ESN {esn}: ERROR ({ex}) — fallback to mean")
                    fb = self._fallback_result(
                        esn,
                        train_mean_hpt,
                        train_mean_hpc,
                        "error_fallback",
                    )
                    fb["file_idx"] = file_idx
                    results.append(fb)

        return pd.DataFrame(results)

    @staticmethod
    def _fallback_result(
        esn, mean_hpt: float, mean_hpc: float, confidence: str
    ) -> dict:
        return {
            "ESN": esn,
            "Cycles_to_HPT_SV": mean_hpt,
            "Cycles_to_HPC_SV": mean_hpc,
            "HPT_cycle": -1,
            "HPC_cycle": -1,
            "confidence": confidence,
        }

    # ════════════════════════════════════════════════════════════════
    #  SAVE / LOAD
    # ════════════════════════════════════════════════════════════════

    def save(self, directory: str = cfg.MODELS_DIR) -> None:
        """Persist gap regressors and metadata."""
        Path(directory).mkdir(parents=True, exist_ok=True)
        joblib.dump(self.lgbm_gap_hpt, f"{directory}/lgbm_gap_hpt.pkl")
        joblib.dump(self.lgbm_gap_hpc, f"{directory}/lgbm_gap_hpc.pkl")
        joblib.dump(self.feature_cols, f"{directory}/regs_feature_cols.pkl")
        joblib.dump(self.scale_coefs_hpt, f"{directory}/scale_coefs_hpt.pkl")
        joblib.dump(self.scale_coefs_hpc, f"{directory}/scale_coefs_hpc.pkl")
        if self._regs_data is not None:
            joblib.dump(self._regs_data, f"{directory}/gap_regs_data.pkl")
        if self._base_pred_hpt is not None:
            joblib.dump(self._base_pred_hpt, f"{directory}/gap_base_pred_hpt.pkl")
        if self._base_pred_hpc is not None:
            joblib.dump(self._base_pred_hpc, f"{directory}/gap_base_pred_hpc.pkl")
        print(f"Gap correction models saved to {directory}/")

    def load(self, directory: str = cfg.MODELS_DIR) -> None:
        """Load gap regressors and metadata."""
        self.lgbm_gap_hpt = joblib.load(f"{directory}/lgbm_gap_hpt.pkl")
        self.lgbm_gap_hpc = joblib.load(f"{directory}/lgbm_gap_hpc.pkl")
        self.feature_cols = joblib.load(f"{directory}/regs_feature_cols.pkl")
        self.scale_coefs_hpt = joblib.load(f"{directory}/scale_coefs_hpt.pkl")
        self.scale_coefs_hpc = joblib.load(f"{directory}/scale_coefs_hpc.pkl")
        p = Path(directory)
        if (p / "gap_regs_data.pkl").exists():
            self._regs_data = joblib.load(p / "gap_regs_data.pkl")
        if (p / "gap_base_pred_hpt.pkl").exists():
            self._base_pred_hpt = joblib.load(p / "gap_base_pred_hpt.pkl")
        if (p / "gap_base_pred_hpc.pkl").exists():
            self._base_pred_hpc = joblib.load(p / "gap_base_pred_hpc.pkl")
        print(f"Gap correction models loaded from {directory}/")

    # ════════════════════════════════════════════════════════════════
    #  PLOTTING
    # ════════════════════════════════════════════════════════════════

    def plot_results(
        self,
        results_df: pd.DataFrame,
        val_list: list[pd.DataFrame],
        test_list: list[pd.DataFrame],
    ) -> None:
        """Plot bar charts, distributions, and per-engine detail for
        the prediction results.
        """
        n_val = len(val_list)
        val_results = results_df[results_df["file_idx"] < n_val].copy()
        test_results = results_df[results_df["file_idx"] >= n_val].copy()

        # ── Bar plots ───────────────────────────────────────────
        fig, axs = plt.subplots(2, 2, figsize=(20, 10))
        fig.suptitle("Predictions — Cycles to Service Visit", fontsize=18)
        self._plot_bar(
            axs[0, 0], val_results, "Cycles_to_HPT_SV", "Validation — HPT", "tab:blue"
        )
        self._plot_bar(
            axs[0, 1], val_results, "Cycles_to_HPC_SV", "Validation — HPC", "tab:green"
        )
        self._plot_bar(
            axs[1, 0], test_results, "Cycles_to_HPT_SV", "Test — HPT", "tab:orange"
        )
        self._plot_bar(
            axs[1, 1], test_results, "Cycles_to_HPC_SV", "Test — HPC", "tab:red"
        )
        plt.tight_layout()
        save_fig(fig, "results_bar")

        # ── Distributions ───────────────────────────────────────
        fig, axs = plt.subplots(1, 2, figsize=(16, 5))
        fig.suptitle("Prediction Distributions", fontsize=16)
        self._plot_histogram(
            axs[0],
            val_results,
            test_results,
            "Cycles_to_HPT_SV",
            "Cycles_to_HPC_SV",
        )
        self._plot_cycle_dist(
            axs[1],
            pd.concat([val_results, test_results], ignore_index=True),
        )
        plt.tight_layout()
        save_fig(fig, "results_distributions")

        # ── Per-engine detail ───────────────────────────────────
        print("\n=== VALIDATION DETAIL ===")
        self._plot_engine_detail(val_list, val_results, "Validation")
        print("\n=== TEST DETAIL ===")
        self._plot_engine_detail(test_list, test_results, "Test")

        # ── Summary table ───────────────────────────────────────
        self._print_summary(val_results, test_results)

    @staticmethod
    def _plot_bar(
        ax: plt.Axes,
        df: pd.DataFrame,
        col: str,
        title: str,
        color: str,
    ) -> None:
        """Single bar-chart subplot."""
        if len(df) == 0:
            return
        x = np.arange(len(df))
        ax.bar(x, df[col], color=color, alpha=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(df["ESN"].astype(str), rotation=90, fontsize=6)
        ax.set_title(title)
        ax.set_ylabel("Cycles")
        ax.grid(True, alpha=0.3)

    @staticmethod
    def _plot_histogram(
        ax: plt.Axes,
        val_results: pd.DataFrame,
        test_results: pd.DataFrame,
        col_hpt: str,
        col_hpc: str,
    ) -> None:
        """Histogram of prediction distributions."""
        if len(val_results) > 0:
            ax.hist(
                val_results[col_hpt],
                bins=20,
                alpha=0.6,
                color="tab:blue",
                label="Val HPT",
            )
            ax.hist(
                val_results[col_hpc],
                bins=20,
                alpha=0.6,
                color="tab:green",
                label="Val HPC",
            )
        if len(test_results) > 0:
            ax.hist(
                test_results[col_hpt],
                bins=20,
                alpha=0.4,
                color="tab:orange",
                label="Test HPT",
            )
            ax.hist(
                test_results[col_hpc],
                bins=20,
                alpha=0.4,
                color="tab:red",
                label="Test HPC",
            )
        ax.set_title("Cycles to SV Distribution")
        ax.set_xlabel("Cycles")
        ax.set_ylabel("Count")
        ax.legend()
        ax.grid(True, alpha=0.3)

    @staticmethod
    def _plot_cycle_dist(
        ax: plt.Axes,
        all_results: pd.DataFrame,
    ) -> None:
        """Predicted maintenance-cycle distribution."""
        if "HPT_cycle" not in all_results.columns:
            return
        cycles_hpt = all_results["HPT_cycle"].value_counts().sort_index()
        cycles_hpc = all_results["HPC_cycle"].value_counts().sort_index()
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
        ax.set_title("Predicted Maintenance Cycles")
        ax.set_xlabel("Cycle")
        ax.set_ylabel("Count")
        ax.legend()
        ax.grid(True, alpha=0.3)

    def _plot_engine_detail(
        self,
        engine_list: list[pd.DataFrame],
        results_sub: pd.DataFrame,
        label_prefix: str,
    ) -> None:
        """Plot the actual predicted RUL signal for every file.

        Uses the prediction series stored in results_sub (the same
        signal whose last value goes into submission.csv).
        Each file gets its own plot, even when multiple files share an ESN.
        """
        for i, engine_df in enumerate(engine_list):
            file_idx_val = results_sub["file_idx"].min() + i
            row_match = results_sub[results_sub["file_idx"] == file_idx_val]
            if len(row_match) == 0:
                continue

            row = row_match.iloc[0]
            esn = row["ESN"]
            series_hpt = row.get("pred_series_hpt")
            series_hpc = row.get("pred_series_hpc")

            # If series not available, recompute via predict_engine
            if series_hpt is None or series_hpc is None:
                edf = engine_df[engine_df["ESN"] == esn].copy()
                engine_res = self.hi.residuals_single(edf)
                if engine_res is None:
                    continue
                pred = self.predict_engine(edf, engine_res, esn)
                series_hpt = pred["pred_series_hpt"]
                series_hpc = pred["pred_series_hpc"]

            fig, axs = plt.subplots(1, 2, figsize=(18, 5))
            fig.suptitle(f"{label_prefix} file {i} (ESN {esn})", fontsize=14)
            self._plot_engine_pred_subplot(
                axs[0],
                series_hpt,
                row["Cycles_to_HPT_SV"],
                row["HPT_cycle"],
                "HPT",
                "tab:blue",
            )
            self._plot_engine_pred_subplot(
                axs[1],
                series_hpc,
                row["Cycles_to_HPC_SV"],
                row["HPC_cycle"],
                "HPC",
                "tab:green",
            )
            plt.tight_layout()
            save_fig(fig, f"engine_detail_{label_prefix.lower()}_file{i}_esn{esn}")

    @staticmethod
    def _plot_engine_pred_subplot(
        ax: plt.Axes,
        pred_series: np.ndarray,
        final_value: float,
        cycle: int,
        component: str,
        color: str,
    ) -> None:
        """Single subplot: actual predicted RUL series + final value."""
        ax.plot(
            pred_series,
            color=color,
            linewidth=0.8,
            label=f"Predicted RUL {component}",
        )
        ax.axhline(
            y=final_value,
            color="red",
            linestyle="--",
            alpha=0.6,
            label=f"Final = {final_value:.0f}",
        )
        ax.set_title(
            f"{component} — Pred: {final_value:.0f} cycles (cycle {cycle})"
        )
        ax.set_xlabel("Observation")
        ax.set_ylabel("Predicted Cycles to SV")
        ax.legend(fontsize="small")
        ax.grid(True, alpha=0.3)

    @staticmethod
    def _print_summary(
        val_results: pd.DataFrame,
        test_results: pd.DataFrame,
    ) -> None:
        """Print summary statistics."""
        print("\n=== SUMMARY STATISTICS ===")
        rows = []
        for label, rdf, comp, col in [
            ("Validation", val_results, "HPT", "Cycles_to_HPT_SV"),
            ("Validation", val_results, "HPC", "Cycles_to_HPC_SV"),
            ("Test", test_results, "HPT", "Cycles_to_HPT_SV"),
            ("Test", test_results, "HPC", "Cycles_to_HPC_SV"),
        ]:
            rows.append(
                {
                    "Dataset": label,
                    "Component": comp,
                    "Mean": rdf[col].mean() if len(rdf) else np.nan,
                    "Std": rdf[col].std() if len(rdf) else np.nan,
                    "Min": rdf[col].min() if len(rdf) else np.nan,
                    "Max": rdf[col].max() if len(rdf) else np.nan,
                }
            )
        print(pd.DataFrame(rows).to_string(index=False))

    # ── Training before/after plotting ──────────────────────────

    def plot_training_before_after(self) -> None:
        """Compare base vs gap-corrected predictions on training data."""
        if self._regs_data is None or self._base_pred_hpt is None:
            print("No training data cached. Run train() first.")
            return

        reg_data = self._regs_data
        base_pred_hpt_all = self._base_pred_hpt
        base_pred_hpc_all = self._base_pred_hpc

        coef_data = self.hi.coef_data
        if coef_data is None:
            return

        for esn in coef_data["ESN"].unique():
            esn_mask = reg_data["ESN"] == esn
            esn_reg = reg_data[esn_mask].copy()

            y_true_hpt = esn_reg["target_hpt"].values
            y_true_hpc = esn_reg["target_hpc"].values

            base_hpt = base_pred_hpt_all[esn_mask.values].copy()
            base_hpc = base_pred_hpc_all[esn_mask.values].copy()

            X_feat = esn_reg[self.feature_cols].values
            gap_hpt = self.lgbm_gap_hpt.predict(X_feat)
            gap_hpc = self.lgbm_gap_hpc.predict(X_feat)
            final_hpt = np.clip(base_hpt + gap_hpt, 0, None)
            final_hpc = np.clip(base_hpc + gap_hpc, 0, None)

            # Build subplot grid based on enabled toggles
            subplot_specs = []
            if cfg.PLOT_BA_TIME_SERIES:
                subplot_specs.append(("time_series", "_plot_time_series"))
            if cfg.PLOT_BA_ERROR:
                subplot_specs.append(("error", "_plot_error"))
            if cfg.PLOT_BA_SCATTER:
                subplot_specs.append(("scatter", "_plot_scatter"))

            if not subplot_specs:
                continue

            n_cols = len(subplot_specs)
            fig, axs = plt.subplots(2, n_cols, figsize=(8 * n_cols, 10))
            fig.suptitle(
                f"Training ESN {esn} — Before vs After Correction",
                fontsize=16,
            )
            if n_cols == 1:
                axs = axs.reshape(-1, 1)
            x_axis = np.arange(len(y_true_hpt))

            for col_idx, (kind, method_name) in enumerate(subplot_specs):
                method = getattr(self, method_name)
                if kind == "scatter":
                    method(axs[0, col_idx], y_true_hpt, base_hpt, final_hpt, "HPT — Scatter", "tab:blue")
                    method(axs[1, col_idx], y_true_hpc, base_hpc, final_hpc, "HPC — Scatter", "tab:green")
                else:
                    label_hpt = "HPT — Cycles to SV" if kind == "time_series" else "HPT — Error"
                    label_hpc = "HPC — Cycles to SV" if kind == "time_series" else "HPC — Error"
                    method(axs[0, col_idx], x_axis, y_true_hpt, base_hpt, final_hpt, label_hpt, "tab:blue")
                    method(axs[1, col_idx], x_axis, y_true_hpc, base_hpc, final_hpc, label_hpc, "tab:green")

            plt.tight_layout()
            save_fig(fig, f"training_before_after_esn_{esn}")

            # Metrics
            rmse_b_hpt = np.sqrt(np.mean((y_true_hpt - base_hpt) ** 2))
            rmse_f_hpt = np.sqrt(np.mean((y_true_hpt - final_hpt) ** 2))
            rmse_b_hpc = np.sqrt(np.mean((y_true_hpc - base_hpc) ** 2))
            rmse_f_hpc = np.sqrt(np.mean((y_true_hpc - final_hpc) ** 2))
            print(f"ESN {esn}:")
            print(
                f"  HPT  Base RMSE={rmse_b_hpt:.2f} → "
                f"Corrected RMSE={rmse_f_hpt:.2f}  "
                f"(Δ={rmse_b_hpt - rmse_f_hpt:+.2f})"
            )
            print(
                f"  HPC  Base RMSE={rmse_b_hpc:.2f} → "
                f"Corrected RMSE={rmse_f_hpc:.2f}  "
                f"(Δ={rmse_b_hpc - rmse_f_hpc:+.2f})"
            )
            print()

        # Global summary
        if cfg.PLOT_BA_GLOBAL_ERROR:
            self._plot_global_error_distribution()

    @staticmethod
    def _plot_time_series(
        ax,
        x,
        y_true,
        base,
        corrected,
        title,
        color,
    ) -> None:
        """Ground truth vs base vs corrected time series."""
        rmse_b = np.sqrt(np.mean((y_true - base) ** 2))
        rmse_c = np.sqrt(np.mean((y_true - corrected) ** 2))
        ax.plot(
            x, y_true, color="black", linewidth=1.0, alpha=0.8, label="Ground Truth"
        )
        ax.plot(
            x,
            base,
            color="tab:orange",
            linewidth=0.8,
            alpha=0.7,
            linestyle="--",
            label=f"Base (RMSE={rmse_b:.1f})",
        )
        ax.plot(
            x,
            corrected,
            color=color,
            linewidth=0.8,
            alpha=0.9,
            label=f"Corrected (RMSE={rmse_c:.1f})",
        )
        ax.set_title(title)
        ax.set_xlabel("Observation")
        ax.set_ylabel("Cycles to SV")
        ax.legend(fontsize="small")
        ax.grid(True, alpha=0.3)

    @staticmethod
    def _plot_error(
        ax,
        x,
        y_true,
        base,
        corrected,
        title,
        color,
    ) -> None:
        """Residual error before and after correction."""
        err_b = y_true - base
        err_c = y_true - corrected
        mae_b = np.mean(np.abs(err_b))
        mae_c = np.mean(np.abs(err_c))
        ax.plot(
            x,
            err_b,
            color="tab:orange",
            linewidth=0.6,
            alpha=0.6,
            label=f"Base error (MAE={mae_b:.1f})",
        )
        ax.plot(
            x,
            err_c,
            color=color,
            linewidth=0.6,
            alpha=0.8,
            label=f"Corrected error (MAE={mae_c:.1f})",
        )
        ax.axhline(y=0, color="black", linestyle="-", linewidth=0.5)
        ax.fill_between(x, err_c, 0, alpha=0.15, color=color)
        ax.set_title(title)
        ax.set_xlabel("Observation")
        ax.set_ylabel("Error")
        ax.legend(fontsize="small")
        ax.grid(True, alpha=0.3)

    @staticmethod
    def _plot_scatter(
        ax,
        y_true,
        base,
        corrected,
        title,
        color,
    ) -> None:
        """Scatter: predicted vs ground truth."""
        lim = [
            min(y_true.min(), corrected.min(), base.min()) - 10,
            max(y_true.max(), corrected.max(), base.max()) + 10,
        ]
        ax.scatter(y_true, base, color="tab:orange", alpha=0.3, s=8, label="Base")
        ax.scatter(y_true, corrected, color=color, alpha=0.3, s=8, label="Corrected")
        ax.plot(lim, lim, color="red", linestyle="--", linewidth=1, label="Perfect")
        ax.set_xlim(lim)
        ax.set_ylim(lim)
        ax.set_title(title)
        ax.set_xlabel("Ground Truth")
        ax.set_ylabel("Prediction")
        ax.legend(fontsize="small")
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3)

    def _plot_global_error_distribution(self) -> None:
        """Histogram of errors across all training engines."""
        if self._regs_data is None:
            return
        reg_data = self._regs_data

        y_hpt = reg_data["target_hpt"].values
        y_hpc = reg_data["target_hpc"].values
        b_hpt = self._base_pred_hpt
        b_hpc = self._base_pred_hpc

        g_hpt = self.lgbm_gap_hpt.predict(reg_data[self.feature_cols].values)
        g_hpc = self.lgbm_gap_hpc.predict(reg_data[self.feature_cols].values)
        f_hpt = b_hpt + g_hpt
        f_hpc = b_hpc + g_hpc

        fig, axs = plt.subplots(1, 2, figsize=(16, 6))
        fig.suptitle("Training Set — Error Distribution", fontsize=16)

        ax = axs[0]
        ax.hist(
            y_hpt - b_hpt, bins=50, alpha=0.5, color="tab:orange", label="Base error"
        )
        ax.hist(
            y_hpt - f_hpt, bins=50, alpha=0.5, color="tab:blue", label="Corrected error"
        )
        ax.axvline(x=0, color="black", linestyle="-", linewidth=0.8)
        ax.set_title("HPT — Error Distribution")
        ax.set_xlabel("Error (Truth - Pred)")
        ax.set_ylabel("Count")
        ax.legend()
        ax.grid(True, alpha=0.3)

        ax = axs[1]
        ax.hist(
            y_hpc - b_hpc, bins=50, alpha=0.5, color="tab:orange", label="Base error"
        )
        ax.hist(
            y_hpc - f_hpc,
            bins=50,
            alpha=0.5,
            color="tab:green",
            label="Corrected error",
        )
        ax.axvline(x=0, color="black", linestyle="-", linewidth=0.8)
        ax.set_title("HPC — Error Distribution")
        ax.set_xlabel("Error (Truth - Pred)")
        ax.set_ylabel("Count")
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        save_fig(fig, "global_error_distribution")

        print("=== GLOBAL METRICS ===")
        for lbl, yt, bp, fp in [
            ("HPT", y_hpt, b_hpt, f_hpt),
            ("HPC", y_hpc, b_hpc, f_hpc),
        ]:
            print(
                f"{lbl}  Base    RMSE="
                f"{np.sqrt(np.mean((yt - bp) ** 2)):.2f}  "
                f"MAE={np.mean(np.abs(yt - bp)):.2f}"
            )
            print(
                f"{lbl}  Corrected RMSE="
                f"{np.sqrt(np.mean((yt - fp) ** 2)):.2f}  "
                f"MAE={np.mean(np.abs(yt - fp)):.2f}"
            )
