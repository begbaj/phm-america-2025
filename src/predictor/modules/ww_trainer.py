"""
ww_trainer.py - WWTrainer class.

Handles Water Wash (WW) prediction:
- Removes HPT/HPC maintenance effects from Sensed_T45 residuals.
- Detects WW events via rolling-mean deviation from linear trend.
- Extrapolates cycles to next WW from the end of available data.
- Plotting functions for WW detection results.
"""

from __future__ import annotations

from math import e as _e
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from modules import config as cfg, save_fig
from modules.data import Data
from modules.hi_trainer import HITrainer


class _FixedSlopeTrend:
    """Lightweight linear trend model with fixed slope and fitted intercept."""

    def __init__(self, slope: float, intercept: float) -> None:
        self.coef_ = np.array([slope], dtype=float)
        self.intercept_ = float(intercept)

    def predict(self, x: np.ndarray) -> np.ndarray:
        x_arr = np.asarray(x).reshape(-1)
        return self.coef_[0] * x_arr + self.intercept_


class WWTrainer:
    """Water Wash event detection and cycles-to-next-WW prediction.

    Parameters
    ----------
    hi_trainer : HITrainer
        Trained HITrainer (needed for ``calc_hi`` to detect HPT/HPC events).
    """

    def __init__(self, hi_trainer: HITrainer) -> None:
        self.hi = hi_trainer

        # Results cache: {esn: ww_result_dict}
        self.results_train: dict[Any, dict] = {}
        self.results_val: dict[Any, dict] = {}
        self.results_test: dict[Any, dict] = {}

        # Training slope statistics for global-slope inference
        self.training_slopes: list[float] = []
        self.training_mean_slope: float | None = None

    # ════════════════════════════════════════════════════════════════
    #  TRAINING SLOPE MANAGEMENT
    # ════════════════════════════════════════════════════════════════

    def clear_training_slopes(self) -> None:
        """Reset cached training slopes and global mean slope."""
        self.training_slopes = []
        self.training_mean_slope = None

    def register_training_slope(self, slope: float) -> None:
        """Store a valid per-engine training slope for global aggregation."""
        if np.isfinite(slope) and slope > 0:
            self.training_slopes.append(float(slope))

    def finalize_training_mean_slope(self) -> float | None:
        """Compute global training slope as mean of collected slopes."""
        if not self.training_slopes:
            self.training_mean_slope = None
            return None
        self.training_mean_slope = float(np.mean(self.training_slopes))
        return self.training_mean_slope

    @staticmethod
    def _fixed_trend_from_slope(
        x: np.ndarray,
        y: np.ndarray,
        slope: float,
    ) -> _FixedSlopeTrend:
        """Create a fixed-slope trend line with least-bias intercept."""
        x_arr = np.asarray(x).reshape(-1)
        y_arr = np.asarray(y).reshape(-1)
        intercept = float(np.mean(y_arr) - slope * np.mean(x_arr))
        return _FixedSlopeTrend(slope, intercept)

    @staticmethod
    def _apply_window_before_ww_predict(df: pd.DataFrame) -> pd.DataFrame:
        """Apply optional rolling window to T45 before WW prediction."""
        if not cfg.APPLY_WINDOW_BEFORE_WW_PREDICT:
            return df
        if "Sensed_T45" not in df.columns:
            return df

        window = int(cfg.WINDOW_BEFORE_WW_PREDICT)
        min_periods = max(1, int(cfg.WINDOW_BEFORE_WW_PREDICT_MIN_PERIODS))
        if window <= 1:
            return df

        out = df.copy()
        sort_cols: list[str] = []
        if "ESN" in out.columns:
            sort_cols.append("ESN")
        if "Cycles_Since_New" in out.columns:
            sort_cols.append("Cycles_Since_New")
        if "Snapshot" in out.columns:
            sort_cols.append("Snapshot")
        if sort_cols:
            out = out.sort_values(sort_cols).copy()

        if "ESN" in out.columns:
            out["Sensed_T45"] = out.groupby("ESN")["Sensed_T45"].transform(
                lambda x: x.rolling(window=window, min_periods=min_periods).mean()
            )
        else:
            out["Sensed_T45"] = out["Sensed_T45"].rolling(
                window=window,
                min_periods=min_periods,
            ).mean()
        return out

    # ════════════════════════════════════════════════════════════════
    #  EFFECT REMOVAL
    # ════════════════════════════════════════════════════════════════

    @staticmethod
    def remove_effect(df: pd.DataFrame, col: str) -> pd.DataFrame:
        """Remove Sensed_T45 jumps caused by known maintenance events.

        Uses a column like ``Cumulative_HPT_SVs`` to detect segment
        boundaries and subtract cumulative offsets.

        Parameters
        ----------
        df : pd.DataFrame
            Engine data with ``Sensed_T45`` and *col* columns.
        col : str
            Column indicating the cumulative maintenance count.
        """
        df = df.sort_values([col, "Cycles_Since_New", "Snapshot"]).copy()
        df = Data.remove_outliers(df, threshold=2.6, sensor_cols=["Sensed_T45"])
        df = df.groupby(["Cycles_Since_New"], as_index=False).median(numeric_only=True)

        grp = df.groupby(col)["Sensed_T45"].agg(["first", "last"])
        grp = grp.sort_index()
        grp["prev_last"] = grp["last"].shift(1)
        grp["jump"] = grp["first"] - grp["prev_last"]
        grp["jump"] = grp["jump"].fillna(0)
        grp["cumulative_offset"] = grp["jump"].cumsum()
        offset_map = grp["cumulative_offset"].to_dict()
        df["Sensed_T45"] = df["Sensed_T45"] - df[col].map(offset_map)
        return df

    def _get_events(
        self, df: pd.DataFrame, soglia: float = 10
    ) -> tuple[pd.Series, pd.Series]:
        """Detect sudden jumps in HI (proxy for maintenance events)."""
        hi_hpt, hi_hpc = self.hi.calc_hi(df)
        cond_hpt = (
            (hi_hpt.diff(1) > soglia)
            & (hi_hpt.diff(2) > soglia)
            & (hi_hpt.diff(3) > soglia)
        )
        cond_hpc = (
            (hi_hpc.diff(1) > soglia)
            & (hi_hpc.diff(2) > soglia)
            & (hi_hpc.diff(3) > soglia)
        )
        return hi_hpt[cond_hpt], hi_hpc[cond_hpc]

    def remove_effect_hard_core(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove maintenance effects when Cumulative_*_SVs are unavailable.

        Detects events via HI jumps and subtracts cumulative offsets.
        """
        df = df.sort_values(["Cycles_Since_New", "Snapshot"]).copy()
        df = Data.remove_outliers(df, threshold=2.6, sensor_cols=["Sensed_T45"])

        num_cols = df.select_dtypes(include=[np.number]).columns
        df[num_cols] = df[num_cols].rolling(window=5, min_periods=1).median()

        ehpt, _ = self._get_events(df)
        df["Event_Group"] = 0
        df.loc[ehpt.index, "Event_Group"] = 1
        df["Event_Group"] = df["Event_Group"].cumsum()

        grp = df.groupby("Event_Group")["Sensed_T45"].agg(["first", "last"])
        grp["prev_last"] = grp["last"].shift(1)
        grp["jump"] = grp["first"] - grp["prev_last"]
        grp["jump"] = grp["jump"].fillna(0)
        grp["cumulative_offset"] = grp["jump"].cumsum()
        offset_map = grp["cumulative_offset"].to_dict()
        df["Sensed_T45"] = df["Sensed_T45"] - df["Event_Group"].map(offset_map)
        df = df.drop(columns=["Event_Group"])
        return df

    # ════════════════════════════════════════════════════════════════
    #  WW EVENT DETECTION
    # ════════════════════════════════════════════════════════════════

    @staticmethod
    def detect_ww_events(
        t45_series: pd.Series,
        slope: float,
        window: int = cfg.WW_DETECTION_WINDOW,
        factor_mult: int = cfg.WW_FACTOR_MULT,
    ) -> dict[Any, float]:
        """Detect WW events based on rolling-mean deviation from trend.

        Parameters
        ----------
        t45_series : pd.Series
            Processed Sensed_T45 values.
        slope : float
            Global linear-trend slope.
        window : int
            Rolling-mean window.
        factor_mult : int
            Threshold multiplier: ``slope * e² * factor_mult``.

        Returns
        -------
        dict
            ``{index: t45_value}`` of detected events.
        """
        factor = (_e**2) * factor_mult
        sv: dict[Any, float] = {}
        counter: list[float] = []
        reference_r: float | None = None
        floor: float | None = None

        for idx, val_t45 in t45_series.items():
            if reference_r is None:
                reference_r = val_t45
                floor = val_t45
                counter.append(val_t45)
                continue

            counter.append(val_t45)
            if len(counter) < window:
                continue

            mc = np.mean(counter[-window:]) - floor
            msp = reference_r - floor

            if (mc - msp) > slope * factor:
                sv[idx] = val_t45
                reference_r = val_t45
                floor = val_t45

        return sv

    # ════════════════════════════════════════════════════════════════
    #  PREDICT — single engine
    # ════════════════════════════════════════════════════════════════

    def predict_ww(
        self,
        engine_df: pd.DataFrame,
        engine_res: pd.DataFrame,
        esn: Any,
        window: int = cfg.WW_DETECTION_WINDOW,
        factor_mult: int = cfg.WW_FACTOR_MULT,
        use_training_mean_slope: bool = cfg.WW_USE_TRAINING_MEAN_SLOPE,
    ) -> dict[str, Any]:
        """Full WW prediction pipeline for a single engine.

        1. Substitute sensors with residuals.
        2. Remove HPT/HPC maintenance effects on T45.
        3. Build one T45 signal.
        4. Fit global linear trend on that signal.
        5. Detect where the same signal deviates enough → WW event.

        Returns
        -------
        dict
            Keys: ``esn``, ``detected_events``, ``n_events``, ``slope``,
            ``wwdf``, ``regression``, ``is_training``.
        """
        wwdf = engine_df.copy()
        wwdf[cfg.DEGRAD_VARS] = engine_res[cfg.DEGRAD_VARS].values

        if "Cycles" in wwdf.columns and "Cycles_Since_New" not in wwdf.columns:
            wwdf = wwdf.rename(columns={"Cycles": "Cycles_Since_New"})

        wwdf = self._apply_window_before_ww_predict(wwdf)

        is_training = (
            "Cumulative_HPT_SVs" in wwdf.columns
            and "Cumulative_HPC_SVs" in wwdf.columns
        )

        if is_training:
            wwdf = self.remove_effect(wwdf, "Cumulative_HPT_SVs")
            wwdf = self.remove_effect(wwdf, "Cumulative_HPC_SVs")
        else:
            wwdf = self.remove_effect_hard_core(wwdf)

        wwdf = wwdf.dropna()

        if len(wwdf) == 0:
            return {
                "esn": esn,
                "detected_events": {},
                "n_events": 0,
                "slope": 0.0,
                "wwdf": wwdf,
                "t45_signal": pd.Series(dtype=float),
                "regression": None,
                "is_training": is_training,
                "used_training_mean_slope": False,
            }

        # Global linear trend
        X = wwdf["Cycles_Since_New"].values.reshape(-1, 1)
        t45_signal = wwdf["Sensed_T45"].copy()
        Y = t45_signal.values
        use_global_slope = (
            use_training_mean_slope
            and self.training_mean_slope is not None
            and self.training_mean_slope > 0
        )

        if use_global_slope:
            slope = float(self.training_mean_slope)
            reg = self._fixed_trend_from_slope(X, Y, slope)
        else:
            reg = LinearRegression().fit(X, Y)
            slope = float(reg.coef_[0])

        # Detect events
        if is_training:
            sv = self._detect_training_events(
                wwdf,
                t45_signal,
                slope,
                window,
                factor_mult,
            )
        else:
            sv = self.detect_ww_events(t45_signal, slope, window, factor_mult)

        return {
            "esn": esn,
            "detected_events": sv,
            "n_events": len(sv),
            "slope": slope,
            "wwdf": wwdf,
            "t45_signal": t45_signal,
            "regression": reg,
            "is_training": is_training,
            "used_training_mean_slope": use_global_slope,
        }

    def _detect_training_events(
        self,
        wwdf: pd.DataFrame,
        t45_series: pd.Series,
        slope: float,
        window: int,
        factor_mult: int,
    ) -> dict[Any, float]:
        """Detect WW events in training data, resetting at real WW boundaries."""
        sv: dict[Any, float] = {}
        counter: list[float] = []
        reference_r: float | None = None
        floor: float | None = None
        last_cum = 0
        factor = (_e**2) * factor_mult

        det_df = pd.DataFrame(
            {
                "Sensed_T45": t45_series.reindex(wwdf.index),
                "Cumulative_WWs": wwdf["Cumulative_WWs"],
            }
        ).dropna()

        for idx, row in det_df.iterrows():
            val_t45 = row["Sensed_T45"]
            val_cum = row["Cumulative_WWs"]

            if reference_r is None:
                reference_r = val_t45
                floor = val_t45
                counter.append(val_t45)
                continue

            counter.append(val_t45)
            if len(counter) < window:
                continue

            mc = np.mean(counter[-window:]) - floor
            msp = reference_r - floor

            if val_cum > last_cum:
                last_cum = val_cum
                reference_r = val_t45
                floor = val_t45

            if (mc - msp) > slope * factor:
                sv[idx] = val_t45
                reference_r = val_t45
                floor = val_t45

        return sv

    # ════════════════════════════════════════════════════════════════
    #  EXTRAPOLATION — cycles to next WW
    # ════════════════════════════════════════════════════════════════

    @staticmethod
    def cycles_to_next_ww_from_end(
        ww_result: dict[str, Any],
        factor_mult: int = cfg.WW_FACTOR_MULT,
    ) -> float:
        """Estimate cycles remaining until the next Water Wash.

        Parameters
        ----------
        ww_result : dict
            Output of ``predict_ww()``.
        factor_mult : int
            Same value used in detection.

        Returns
        -------
        float
            Estimated cycles >= 0.
        """
        wwdf = ww_result["wwdf"]
        slope = ww_result["slope"]
        sv = ww_result["detected_events"]

        if len(wwdf) == 0 or slope <= 0:
            return 0.0

        trigger_threshold = slope * (_e**2) * factor_mult

        t45 = ww_result.get("t45_signal", wwdf["Sensed_T45"])

        if sv:
            last_event_idx = max(sv.keys())
            floor = sv[last_event_idx]
        else:
            floor = t45.iloc[0]

        current_t45 = t45.iloc[-1]
        current_rise = current_t45 - floor
        remaining_rise = trigger_threshold - current_rise

        if remaining_rise <= 0:
            return 0.0

        return max(0.0, remaining_rise / slope)

    # ════════════════════════════════════════════════════════════════
    #  BATCH PREDICTION
    # ════════════════════════════════════════════════════════════════

    def predict_batch(
        self,
        engine_list: list[pd.DataFrame],
        label: str = "engines",
        use_training_mean_slope: bool = cfg.WW_USE_TRAINING_MEAN_SLOPE,
    ) -> dict[Any, dict]:
        """Run WW prediction for every engine in a list.

        Returns dict keyed by ESN.
        """
        print(f"=== WW PREDICTION — {label.upper()} ===")
        results: dict[Any, dict] = {}

        for engine_df in engine_list:
            for esn in engine_df["ESN"].unique():
                edf = engine_df[engine_df["ESN"] == esn]
                engine_res = self.hi.residuals_single(edf)
                if engine_res is None:
                    continue
                result = self.predict_ww(
                    edf,
                    engine_res,
                    esn,
                    use_training_mean_slope=use_training_mean_slope,
                )
                results[esn] = result
        return results

    def predict_all(self, data: Data) -> None:
        """Run WW prediction on training, validation, and test."""
        # Training (use full train df, per-ESN)
        print("=== WW PREDICTION — TRAINING ===")
        if cfg.WW_USE_TRAINING_MEAN_SLOPE:
            self.clear_training_slopes()
        for esn in data.train["ESN"].unique():
            edf = data.train[data.train["ESN"] == esn]
            engine_res = self.hi.residuals_single(edf)
            if engine_res is None:
                continue
            result = self.predict_ww(
                edf,
                engine_res,
                esn,
                use_training_mean_slope=False,
            )
            if cfg.WW_USE_TRAINING_MEAN_SLOPE:
                self.register_training_slope(float(result["slope"]))
            self.results_train[esn] = result
            self.plot_ww_prediction(result)

        if cfg.WW_USE_TRAINING_MEAN_SLOPE:
            self.finalize_training_mean_slope()

        self.results_val = self.predict_batch(
            data.validation,
            "validation",
            use_training_mean_slope=cfg.WW_USE_TRAINING_MEAN_SLOPE,
        )
        for r in self.results_val.values():
            self.plot_ww_prediction(r)

        self.results_test = self.predict_batch(
            data.test,
            "test",
            use_training_mean_slope=cfg.WW_USE_TRAINING_MEAN_SLOPE,
        )
        for r in self.results_test.values():
            self.plot_ww_prediction(r)

    # ════════════════════════════════════════════════════════════════
    #  PLOTTING
    # ════════════════════════════════════════════════════════════════

    def plot_ww_prediction(self, ww_result: dict[str, Any]) -> None:
        """Plot WW detection result for a single engine.

        Two subplots:
        1. T45 residuals + detected events + trend line.
        2. Detrended T45 showing fouling accumulation.
        """
        esn = ww_result["esn"]
        sv = ww_result["detected_events"]
        wwdf = ww_result["wwdf"]
        reg = ww_result["regression"]
        slope = ww_result["slope"]
        is_training = ww_result["is_training"]

        if len(wwdf) == 0 or reg is None:
            print(f"ESN {esn}: no data for WW plot")
            return

        X = wwdf["Cycles_Since_New"].values.reshape(-1, 1)

        subplots = []
        if cfg.PLOT_WW_T45_EVENTS:
            subplots.append(("t45", self._plot_t45_with_events))
        if cfg.PLOT_WW_DETRENDED:
            subplots.append(("detrended", self._plot_detrended))
        if not subplots:
            return

        n = len(subplots)
        fig, axs = plt.subplots(1, n, figsize=(11 * n, 6))
        fig.suptitle(f"WW Prediction — ESN {esn}", fontsize=16)
        if n == 1:
            axs = [axs]
        for ax, (kind, fn) in zip(axs, subplots):
            if kind == "t45":
                fn(ax, wwdf, reg, X, sv, slope, is_training)
            else:
                fn(ax, wwdf, reg, X, sv)

        plt.tight_layout()
        save_fig(fig, f"ww_prediction_esn_{esn}")

        if is_training and "Cumulative_WWs" in wwdf.columns:
            n_real = len(wwdf["Cumulative_WWs"].unique())
            print(
                f"  ESN {esn}: Detected={len(sv)}, "
                f"Real WW cycles={n_real}, Slope={slope:.6f}"
            )
        else:
            print(f"  ESN {esn}: Detected={len(sv)}, Slope={slope:.6f}")

    @staticmethod
    def _plot_t45_with_events(
        ax: plt.Axes,
        wwdf: pd.DataFrame,
        reg: LinearRegression,
        X: np.ndarray,
        sv: dict,
        slope: float,
        is_training: bool,
    ) -> None:
        """Subplot: T45 residuals with detected events and trend."""
        ax.scatter(
            wwdf["Cycles_Since_New"],
            wwdf["Sensed_T45"],
            color="tab:blue",
            alpha=0.6,
            s=3,
            label="T45 (residuals)",
            zorder=2,
        )
        ax.plot(
            wwdf["Cycles_Since_New"],
            reg.predict(X),
            color="red",
            linewidth=2,
            label=f"Trend (slope={slope:.6f})",
            zorder=3,
        )

        if sv:
            sv_cycles = wwdf.loc[list(sv.keys()), "Cycles_Since_New"]
            ax.vlines(
                x=sv_cycles,
                ymin=wwdf["Sensed_T45"].min(),
                ymax=wwdf["Sensed_T45"].max(),
                colors="green",
                linestyles="dashed",
                alpha=0.7,
                label=f"Detected WW ({len(sv)})",
                zorder=1,
            )

        if is_training and "Cumulative_WWs" in wwdf.columns:
            ww_bounds = wwdf["Cycles_Since_New"].groupby(wwdf["Cumulative_WWs"]).last()
            n_real = len(wwdf["Cumulative_WWs"].unique())
            ax.vlines(
                x=ww_bounds,
                ymin=wwdf["Sensed_T45"].min(),
                ymax=wwdf["Sensed_T45"].max(),
                colors="gray",
                linestyles="dotted",
                alpha=0.5,
                label=f"Real WW boundaries ({n_real})",
                zorder=1,
            )

        ax.set_title("T45 Residuals + WW Detection")
        ax.set_xlabel("Cycles Since New")
        ax.set_ylabel("Sensed T45 (residuals)")
        ax.legend(fontsize="small")
        ax.grid(True, alpha=0.3)

    @staticmethod
    def _plot_detrended(
        ax: plt.Axes,
        wwdf: pd.DataFrame,
        reg: LinearRegression,
        X: np.ndarray,
        sv: dict,
    ) -> None:
        """Subplot: detrended T45 showing fouling accumulation."""
        detrended = wwdf["Sensed_T45"].values - reg.predict(X).flatten()
        ax.scatter(
            wwdf["Cycles_Since_New"],
            detrended,
            color="tab:purple",
            alpha=0.6,
            s=3,
            label="T45 detrended",
            zorder=2,
        )
        ax.axhline(y=0, color="black", linewidth=0.5, linestyle="-")

        if sv:
            sv_cycles = wwdf.loc[list(sv.keys()), "Cycles_Since_New"]
            sv_detrended = [
                detrended[wwdf["Cycles_Since_New"].values == c][0]
                if len(detrended[wwdf["Cycles_Since_New"].values == c]) > 0
                else 0
                for c in sv_cycles
            ]
            ax.scatter(
                sv_cycles,
                sv_detrended,
                color="green",
                s=50,
                marker="v",
                zorder=3,
                label=f"WW events ({len(sv)})",
                edgecolors="black",
                linewidth=0.5,
            )

        ax.set_title("T45 Detrended — Fouling Accumulation")
        ax.set_xlabel("Cycles Since New")
        ax.set_ylabel("T45 - Trend")
        ax.legend(fontsize="small")
        ax.grid(True, alpha=0.3)
