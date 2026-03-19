"""
hi_trainer.py - HITrainer class.

Handles:
- Training the linear regressor ensemble for nominal-behaviour modelling.
- Computing residuals (predicted - actual sensor readings).
- Finding optimised alpha coefficients via differential_evolution.
- Computing Health Indices (single-alpha HI and multi-variable HIE).
- scale_to_target normalisation.
- Full ``predict_cycles_to_sv_v2`` inference pipeline.
- All associated plotting methods (each subplot has its own function).
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from scipy import stats
from scipy.optimize import differential_evolution
from sklearn.linear_model import LinearRegression

from modules import config as cfg, save_fig
from modules.data import Data


class HITrainer:
    """Train nominal-behaviour models, compute residuals and Health Indices,
    and optimise HI coefficients.

    Parameters
    ----------
    data : Data
        Populated ``Data`` instance.
    """

    def __init__(self, data: Data) -> None:
        self.data = data

        # Linear regressor ensemble  {esn_str: LinearRegression}
        self.models: dict[str, LinearRegression] = {}

        # Alpha coefficients (scalar or dict)
        self.chpt: Any = None
        self.chpc: Any = None

        # Computed residuals (cached)
        self._res_train_healthy: pd.DataFrame | None = None
        self._res_train: pd.DataFrame | None = None
        self._res_test_loo: pd.DataFrame | None = None
        self._res_validation: pd.DataFrame | None = None
        self._res_test: pd.DataFrame | None = None

        # Coefficient-search intermediates
        self.coef_data: pd.DataFrame | None = None
        self.target_vars: list[str] = []

    # ════════════════════════════════════════════════════════════════
    #  1.  LINEAR REGRESSOR TRAINING
    # ════════════════════════════════════════════════════════════════

    def train_linear_models(self) -> None:
        """Train the ensemble of per-ESN linear regressors on
        ``data.train_data``.
        """
        train_data = self.data.train_data

        if cfg.ENSAMBLE:
            for esn in train_data["ESN"].unique():
                start = time.time()
                print(f"Training linear model for ESN {esn}...", end=" ")
                mask = train_data["ESN"] == esn
                X = train_data.loc[mask, cfg.OPERATING_VARS]
                Y = train_data.loc[mask, cfg.DEGRAD_VARS]
                model = LinearRegression().fit(X, Y)
                self.models[str(esn)] = model
                print(f"{time.time() - start:.3f}s")
        else:
            if not cfg.INCLUDE_TEST:
                mask = train_data["ESN"] != cfg.TESTING_ESN
            else:
                mask = pd.Series(True, index=train_data.index)
            X = train_data.loc[mask, cfg.OPERATING_VARS]
            Y = train_data.loc[mask, cfg.DEGRAD_VARS]
            model = LinearRegression().fit(X, Y)
            self.models["all"] = model

    # ════════════════════════════════════════════════════════════════
    #  2.  RESIDUAL COMPUTATION
    # ════════════════════════════════════════════════════════════════

    def _ensamble_predict(self, data: pd.DataFrame) -> np.ndarray:
        """Mean prediction across all ensemble models."""
        preds = [m.predict(data) for m in self.models.values()]
        return np.mean(preds, axis=0)

    def _predict(self, data: pd.DataFrame, esn: Any = None) -> np.ndarray:
        """Predict using per-ESN model (if SEPARATE_MODELS) or ensemble."""
        if not cfg.SEPARATE_MODELS:
            return self._ensamble_predict(data)
        if esn is not None:
            key = str(esn)
            if key in self.models:
                return self.models[key].predict(data)
        return self._ensamble_predict(data)

    def residuals_single(self, df: pd.DataFrame) -> pd.DataFrame | None:
        """Compute residuals for a single DataFrame (one or more ESNs)."""
        res_list: list[pd.DataFrame] = []
        for esn in df["ESN"].unique():
            mask = df["ESN"] == esn
            X = df.loc[mask, cfg.OPERATING_VARS]
            Y = df.loc[mask, cfg.DEGRAD_VARS]
            Y_pred = self._predict(X, esn)
            if Y_pred is None:
                return None
            res_temp = Y - Y_pred
            res_temp = Data.remove_outliers(res_temp, threshold=3, method="iqr")
            res_temp = res_temp.ffill().bfill()
            res_temp["ESN"] = esn
            try:
                res_temp["Cycles"] = df.loc[mask, "Cycles_Since_New"]
            except KeyError:
                res_temp["Cycles"] = df.loc[mask, "Cycles"]
            res_list.append(res_temp)
        return pd.concat(res_list)

    def residuals(self, df: pd.DataFrame | list[pd.DataFrame]) -> pd.DataFrame:
        """Compute residuals for a DataFrame or list of DataFrames."""
        if isinstance(df, list):
            parts = [self.residuals_single(d) for d in df]
            parts = [p for p in parts if p is not None]
        else:
            parts = [self.residuals_single(df)]
            parts = [p for p in parts if p is not None]
        return pd.concat(parts)

    def compute_all_residuals(self) -> None:
        """Compute and cache residuals for all datasets."""
        print("Computing residuals: training (healthy)...")
        self._res_train_healthy = self.residuals(self.data.train_data)

        train_full = self.data.train
        if not cfg.INCLUDE_TEST:
            train_full = train_full[train_full["ESN"] != cfg.TESTING_ESN].copy()
        print("Computing residuals: training (full)...")
        self._res_train = self.residuals(train_full)

        print("Computing residuals: leave-one-out test ESN...")
        self._res_test_loo = self.residuals(self.data.test_loo)

        print("Computing residuals: validation set...")
        self._res_validation = self.residuals(self.data.validation)

        print("Computing residuals: test set...")
        self._res_test = self.residuals(self.data.test)

    # ════════════════════════════════════════════════════════════════
    #  3.  HEALTH INDEX FUNCTIONS
    # ════════════════════════════════════════════════════════════════

    @staticmethod
    def HI(T3: pd.Series, T45: pd.Series, alpha: float) -> pd.Series:
        """Single-alpha Health Index: ``-alpha * T3 - T45``."""
        return -alpha * T3 - T45

    @staticmethod
    def HIE(params, variables: pd.DataFrame) -> pd.Series:
        """Multi-variable Health Index: ``vars . (-params)``."""
        return variables.dot(-np.array(params))

    def calc_hi(
        self,
        sensor_data: pd.DataFrame,
        ahpt: Any = None,
        ahpc: Any = None,
    ) -> tuple[pd.Series, pd.Series]:
        """Compute HI_HPT and HI_HPC for sensor data.

        Parameters
        ----------
        sensor_data : pd.DataFrame
            Must contain either target_vars (USE_ALL_VARS) or
            ``Sensed_T3`` + ``Sensed_T45``.
        ahpt, ahpc : coefficients (scalar or array).
            If ``None``, uses ``self.chpt`` / ``self.chpc``.
            When the stored coefficients are dicts, they are resolved
            via ``get_coefs_for_esn`` using the ESN in *sensor_data*.
        """
        if ahpt is None or ahpc is None:
            # Resolve dict coefficients when no explicit values given
            if isinstance(self.chpt, dict) or isinstance(self.chpc, dict):
                esn = (
                    sensor_data["ESN"].iloc[0] if "ESN" in sensor_data.columns else None
                )
                _ahpt, _ahpc = self.get_coefs_for_esn(esn)
                if ahpt is None:
                    ahpt = _ahpt
                if ahpc is None:
                    ahpc = _ahpc
            else:
                if ahpt is None:
                    ahpt = self.chpt
                if ahpc is None:
                    ahpc = self.chpc

        if cfg.USE_ALL_VARS:
            hi_hpt = self.HIE(ahpt, sensor_data[self.target_vars])
            hi_pc = self.HIE(ahpc, sensor_data[self.target_vars])
        else:
            hi_hpt = self.HI(
                sensor_data["Sensed_T3"],
                sensor_data["Sensed_T45"],
                ahpt,
            )
            hi_pc = self.HI(
                sensor_data["Sensed_T3"],
                sensor_data["Sensed_T45"],
                ahpc,
            )
        return hi_hpt, hi_pc

    # ════════════════════════════════════════════════════════════════
    #  4.  COEFFICIENT OPTIMISATION (differential_evolution)
    # ════════════════════════════════════════════════════════════════

    def train_coefficients(self) -> None:
        """Find optimal HI alpha coefficients via differential_evolution,
        or load defaults from config.
        """
        # Prepare data (always needed — downstream classifiers use coef_data)
        if cfg.USE_ONLY_TRAIN:
            esn_data = self.data.train[self.data.train["ESN"] != cfg.TESTING_ESN].copy()
        else:
            esn_data = self.data.train.copy()

        res = self.residuals(esn_data)
        esn_data[cfg.DEGRAD_VARS] = res[cfg.DEGRAD_VARS]
        X_train = esn_data.copy()

        if cfg.USE_CLEAN_DATA:
            X_train = X_train.groupby(
                ["ESN", "Cycles_Since_New"], as_index=False
            ).median(numeric_only=True)
            X_train = Data.remove_outliers(
                X_train, threshold=cfg.COEF_OUTLIERS_THRESHOLD
            )
            X_train = X_train.ffill().bfill().dropna()

        self.coef_data = X_train.copy()

        if cfg.DO_NOT_TRAIN_COEFS:
            self.chpt = cfg.DEFAULT_CHPT
            self.chpc = cfg.DEFAULT_CHPC
            print("Using default coefficients from config.")
            return

        # Select target function and bounds
        if not cfg.USE_ALL_VARS:
            self.target_vars = ["Sensed_T45", "Sensed_T3"]
            bounds = [(-1000, 1000)]
            target_fn = self._objective_single
        else:
            self.target_vars = list(cfg.THIS_ALL_VARS)
            bounds = [(-1000, 1000)] * len(self.target_vars)
            target_fn = self._objective_multi

        chpt_dict: dict[str, Any] = {}
        chpc_dict: dict[str, Any] = {}

        for esn in X_train["ESN"].unique():
            print(f"Optimizing ESN {esn}...", end=" ")
            esn_mask = X_train["ESN"] == esn
            tv = X_train.loc[esn_mask, self.target_vars]

            # HPT
            rul_hpt = X_train.loc[esn_mask, "Cycles_to_HPT_SV"]
            result_hpt = differential_evolution(
                target_fn,
                bounds=bounds,
                args=(tv, rul_hpt),
                strategy="best1bin",
                maxiter=cfg.DE_MAXITER,
                popsize=cfg.DE_POPSIZE,
                workers=-1,
                tol=cfg.DE_TOL,
            )
            chpt_dict[str(esn)] = result_hpt.x
            print(f"HPT α={result_hpt.x}", end=" | ")

            # HPC
            rul_hpc = X_train.loc[esn_mask, "Cycles_to_HPC_SV"]
            result_hpc = differential_evolution(
                target_fn,
                bounds=bounds,
                args=(tv, rul_hpc),
                strategy="best1bin",
                maxiter=cfg.DE_MAXITER,
                popsize=cfg.DE_POPSIZE,
                workers=-1,
                tol=cfg.DE_TOL,
            )
            chpc_dict[str(esn)] = result_hpc.x
            print(f"HPC α={result_hpc.x}")

        if not cfg.SEPARATE_COEFS:
            chpt_vals = np.array(list(chpt_dict.values()))
            chpc_vals = np.array(list(chpc_dict.values()))
            self.chpt = float(np.median(chpt_vals))
            self.chpc = float(np.median(chpc_vals))
        else:
            self.chpt = chpt_dict
            self.chpc = chpc_dict

        print("\nFINAL COEFFICIENTS:")
        if isinstance(self.chpt, dict):
            for k in self.chpt:
                print(f"  ESN {k}: HPT α={self.chpt[k]}, HPC α={self.chpc[k]}")
        else:
            print(f"  HPT (median): {self.chpt}")
            print(f"  HPC (median): {self.chpc}")

    # ── objective functions ──────────────────────────────────────

    def _objective_single(self, a, sensor_data: pd.DataFrame, RUL: pd.Series) -> float:
        """Minimises ``-pearsonr`` between HI and RUL (single alpha)."""
        hi = self.HI(sensor_data["Sensed_T3"], sensor_data["Sensed_T45"], a)
        # valid = hi.dropna().index.intersection(RUL.dropna().index)
        # if len(valid) < 3:
        #     return 1.0
        # hi_valid = hi.loc[valid]
        # rul_valid = RUL.loc[valid]
        # if hi_valid.max() == hi_valid.min():
        #     return 1.0
        # return -stats.pearsonr(rul_valid, hi_valid)[0]
        return -stats.pearsonr(RUL, hi)[0]

    def _objective_multi(
        self, params, sensor_data: pd.DataFrame, RUL: pd.Series
    ) -> float:
        """Minimises MSE between normalised HIE and RUL."""
        hi = self.HIE(params, sensor_data)
        hi_min, hi_max = hi.min(), hi.max()
        if hi_max == hi_min:
            return 1.0
        hi_norm = (hi - hi_min) / (hi_max - hi_min)
        valid = hi_norm.dropna().index.intersection(RUL.dropna().index)
        if len(valid) < 3:
            return 1.0
        return float(np.mean((hi_norm.loc[valid] - RUL.loc[valid]) ** 2))

    # ════════════════════════════════════════════════════════════════
    #  5.  SCALE-TO-TARGET
    # ════════════════════════════════════════════════════════════════

    @staticmethod
    def scale_to_target(
        source: pd.Series,
        target: pd.Series,
        coefs: dict,
    ) -> pd.Series:
        """Scale ``source`` to the range of ``target``, saving min/max
        coefficients into *coefs*.
        """
        if not cfg.SCALE_TARGET:
            if isinstance(coefs, dict):
                coefs.setdefault("min", []).append(0)
                coefs.setdefault("max", []).append(1)
            return source.copy()

        s_min, s_max = source.min(), source.max()
        t_min, t_max = target.min(), target.max()
        if isinstance(coefs, dict):
            coefs.setdefault("min", []).append(t_min)
            coefs.setdefault("max", []).append(t_max)
        denom = s_max - s_min
        if denom == 0:
            return pd.Series(
                np.full(len(source), (t_max + t_min) / 2),
                index=source.index,
            )
        return (source - s_min) / denom * (t_max - t_min) + t_min

    @staticmethod
    def scale_to_target_test(
        source: pd.Series,
        coefs: dict,
    ) -> pd.Series:
        """Scale ``source`` using pre-saved coefficients."""
        if not cfg.SCALE_TARGET:
            return source.copy()

        s_min, s_max = source.min(), source.max()
        denom = s_max - s_min
        if denom == 0:
            return pd.Series(
                np.full(len(source), (coefs["max"] + coefs["min"]) / 2),
                index=source.index,
            )
        return (source - s_min) / denom * (coefs["max"] - coefs["min"]) + coefs["min"]

    # ════════════════════════════════════════════════════════════════
    #  6.  HELPER: get per-ESN coefficients
    # ════════════════════════════════════════════════════════════════

    def get_coefs_for_esn(self, esn) -> tuple[Any, Any]:
        """Return (ahpt, ahpc) for the given ESN, handling dict vs scalar."""
        if cfg.SEPARATE_COEFS and isinstance(self.chpt, dict):
            ahpt = self.chpt.get(
                str(esn),
                np.median(
                    [
                        v[0] if isinstance(v, np.ndarray) else v
                        for v in self.chpt.values()
                    ]
                ),
            )
            ahpc = self.chpc.get(
                str(esn),
                np.median(
                    [
                        v[0] if isinstance(v, np.ndarray) else v
                        for v in self.chpc.values()
                    ]
                ),
            )
        else:
            ahpt = (
                self.chpt
                if not isinstance(self.chpt, dict)
                else np.median(list(self.chpt.values()))
            )
            ahpc = (
                self.chpc
                if not isinstance(self.chpc, dict)
                else np.median(list(self.chpc.values()))
            )
        return ahpt, ahpc

    # ════════════════════════════════════════════════════════════════
    #  SAVE / LOAD
    # ════════════════════════════════════════════════════════════════

    def save(self, directory: str = cfg.MODELS_DIR) -> None:
        """Persist HITrainer state (models, coefficients, coef_data) to disk."""
        Path(directory).mkdir(parents=True, exist_ok=True)
        joblib.dump(self.models, f"{directory}/hi_models.pkl")
        joblib.dump(self.chpt, f"{directory}/hi_chpt.pkl")
        joblib.dump(self.chpc, f"{directory}/hi_chpc.pkl")
        joblib.dump(self.target_vars, f"{directory}/hi_target_vars.pkl")
        if self.coef_data is not None:
            joblib.dump(self.coef_data, f"{directory}/hi_coef_data.pkl")
        print(f"HITrainer saved to {directory}/")

    def load(self, directory: str = cfg.MODELS_DIR) -> None:
        """Load HITrainer state from disk."""
        self.models = joblib.load(f"{directory}/hi_models.pkl")
        self.chpt = joblib.load(f"{directory}/hi_chpt.pkl")
        self.chpc = joblib.load(f"{directory}/hi_chpc.pkl")
        self.target_vars = joblib.load(f"{directory}/hi_target_vars.pkl")
        coef_path = Path(f"{directory}/hi_coef_data.pkl")
        if coef_path.exists():
            self.coef_data = joblib.load(coef_path)
        print(f"HITrainer loaded from {directory}/")

    def get_median_coefs(self) -> tuple[float, float]:
        """Return median (ahpt, ahpc) for unknown engines."""
        if isinstance(self.chpt, dict):
            hpt_vals = [
                v[0] if isinstance(v, np.ndarray) else v for v in self.chpt.values()
            ]
            hpc_vals = [
                v[0] if isinstance(v, np.ndarray) else v for v in self.chpc.values()
            ]
            return float(np.median(hpt_vals)), float(np.median(hpc_vals))
        ahpt = self.chpt[0] if isinstance(self.chpt, np.ndarray) else float(self.chpt)
        ahpc = self.chpc[0] if isinstance(self.chpc, np.ndarray) else float(self.chpc)
        return ahpt, ahpc

    # ════════════════════════════════════════════════════════════════
    #  7.  PLOTTING — residuals
    # ════════════════════════════════════════════════════════════════

    def plot_residuals(
        self,
        data: pd.DataFrame,
        window: int = 1,
        min_periods: int = 1,
        title_suffix: str = "",
    ) -> None:
        """Plot residual curves for all ESNs in *data*.

        Each degradation variable gets its own subplot.
        """
        fig, axs = plt.subplots(2, 3, figsize=(15, 8))
        fig.suptitle(
            f"Residuals Comparison (Window: {window}) {title_suffix}",
            fontsize=16,
        )
        axs_flat = axs.flatten()

        for esn in data["ESN"].unique():
            if "aug" in str(esn):
                continue
            res_temp = data[data["ESN"] == esn]
            if cfg.PLOT_GROUP_CYCLES:
                res_temp = res_temp.groupby("Cycles").mean()
            if cfg.PLOT_REMOVE_OUTLIERS:
                res_temp = Data.remove_outliers(
                    res_temp,
                    threshold=cfg.PLOT_OUTLIERS_THRESHOLD,
                )
                res_temp = res_temp.ffill().bfill()
            for i, ax in enumerate(axs_flat):
                if i >= len(cfg.DEGRAD_VARS):
                    break
                d_var = cfg.DEGRAD_VARS[i]
                t = res_temp[d_var]
                degrad = (
                    t.rolling(window=window, min_periods=min_periods)
                    .mean()
                    .reset_index(drop=True)
                )
                self._plot_residual_axis(ax, degrad, d_var, str(esn))
        axs_flat[0].legend(fontsize="small", loc="upper right")
        plt.tight_layout()
        suffix = title_suffix.replace(" ", "_").lower() if title_suffix else "residuals"
        save_fig(fig, f"residuals_{suffix}")

    @staticmethod
    def _plot_residual_axis(
        ax: Axes,
        degrad: pd.Series,
        var_name: str,
        label: str,
    ) -> None:
        """Draw a single residual subplot (line + scatter)."""
        ax.plot(degrad, linewidth=0.6, alpha=0.6, label=label)
        ax.scatter(
            range(len(degrad)),
            degrad,
            linewidth=0.6,
            alpha=0.7,
            label=label,
            s=2,
        )
        ax.set_title(var_name)
        ax.grid(True, alpha=0.3)

    # ════════════════════════════════════════════════════════════════
    #  8.  PLOTTING — training HI
    # ════════════════════════════════════════════════════════════════

    def plot_training_hi(self) -> None:
        """Plot HI_HPT and HI_HPC for each training ESN."""
        if self.coef_data is None:
            print("No coef_data. Run train_coefficients() first.")
            return

        for esn in self.coef_data["ESN"].unique():
            sd = self.coef_data.loc[self.coef_data["ESN"] == esn].copy()
            if "Cycles_Since_New" in sd.columns:
                sd = sd.sort_values("Cycles_Since_New")
                x_axis = sd["Cycles_Since_New"]
            elif "Cycles" in sd.columns:
                sd = sd.sort_values("Cycles")
                x_axis = sd["Cycles"]
            else:
                x_axis = pd.Series(np.arange(len(sd)), index=sd.index)

            ahpt, ahpc = self.get_coefs_for_esn(esn)
            hi_hpt, hi_hpc = self.calc_hi(sd, ahpt, ahpc)
            rul_hpt = (
                sd["Cycles_to_HPT_SV"] if "Cycles_to_HPT_SV" in sd.columns else None
            )
            rul_hpc = (
                sd["Cycles_to_HPC_SV"] if "Cycles_to_HPC_SV" in sd.columns else None
            )

            subplots = []
            if cfg.PLOT_HI_HPT:
                subplots.append(
                    (
                        hi_hpt,
                        rul_hpt,
                        "Health Index (HPT)",
                        "True RUL (HPT)",
                        "tab:blue",
                    )
                )
            if cfg.PLOT_HI_HPC:
                subplots.append(
                    (
                        hi_hpc,
                        rul_hpc,
                        "Health Index (HPC)",
                        "True RUL (HPC)",
                        "tab:green",
                    )
                )
            if not subplots:
                continue

            fig, axs = plt.subplots(1, len(subplots), figsize=(15 * len(subplots), 6))
            fig.suptitle(f"Training: ESN - {esn}", fontsize=16)
            if len(subplots) == 1:
                axs = [axs]
            for ax, (hi_data, rul_data, label, rul_label, color) in zip(axs, subplots):
                self._plot_hi_subplot(
                    ax,
                    x_axis,
                    hi_data,
                    rul_data,
                    label,
                    rul_label,
                    color,
                )
            fig.tight_layout()
            save_fig(fig, f"training_hi_esn_{esn}")

    @staticmethod
    def _plot_hi_subplot(
        ax: Axes,
        x_axis: pd.Series,
        hi: pd.Series,
        rul: pd.Series | None,
        label: str,
        rul_label: str,
        color: str,
    ) -> None:
        """Draw HI line + scatter and overlay true RUL when available."""
        ax.plot(x_axis, hi, color=color, label=label, linewidth=0.8)
        ax.scatter(x_axis, hi, color=color, s=3, alpha=0.7)
        ax.set_xlabel("Cycles Since New")
        ax.grid(True, alpha=0.3)

        if rul is not None:
            ax_rul = ax.twinx()
            ax_rul.plot(
                x_axis,
                rul,
                color="tab:red",
                linestyle="--",
                linewidth=1.0,
                alpha=0.9,
                label=rul_label,
            )
            h1, l1 = ax.get_legend_handles_labels()
            h2, l2 = ax_rul.get_legend_handles_labels()
            ax.legend(h1 + h2, l1 + l2, fontsize="small", loc="best")
        else:
            ax.legend(fontsize="small", loc="best")
