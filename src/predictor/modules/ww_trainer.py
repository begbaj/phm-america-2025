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
    ) -> dict[str, Any]:
        """Full WW prediction pipeline for a single engine.

        1. Substitute sensors with residuals.
        2. Remove HPT/HPC maintenance effects on T45.
        3. Fit global linear trend.
        4. Detect where T45 deviates enough → WW event.

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
                "regression": None,
                "is_training": is_training,
            }

        # Global linear trend
        X = wwdf["Cycles_Since_New"].values.reshape(-1, 1)
        Y = wwdf["Sensed_T45"].values
        reg = LinearRegression().fit(X, Y)
        slope = reg.coef_[0]

        # Detect events
        if is_training:
            sv = self._detect_training_events(wwdf, slope, window, factor_mult)
        else:
            sv = self.detect_ww_events(wwdf["Sensed_T45"], slope, window, factor_mult)

        return {
            "esn": esn,
            "detected_events": sv,
            "n_events": len(sv),
            "slope": slope,
            "wwdf": wwdf,
            "regression": reg,
            "is_training": is_training,
        }

    def _detect_training_events(
        self,
        wwdf: pd.DataFrame,
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

        for idx, row in wwdf[["Sensed_T45", "Cumulative_WWs"]].iterrows():
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

        if sv:
            last_event_idx = max(sv.keys())
            floor = sv[last_event_idx]
        else:
            floor = wwdf["Sensed_T45"].iloc[0]

        current_t45 = wwdf["Sensed_T45"].iloc[-1]
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
                result = self.predict_ww(edf, engine_res, esn)
                results[esn] = result
        return results

    def predict_all(self, data: Data) -> None:
        """Run WW prediction on training, validation, and test."""
        # Training (use full train df, per-ESN)
        print("=== WW PREDICTION — TRAINING ===")
        for esn in data.train["ESN"].unique():
            edf = data.train[data.train["ESN"] == esn]
            engine_res = self.hi.residuals_single(edf)
            if engine_res is None:
                continue
            result = self.predict_ww(edf, engine_res, esn)
            self.results_train[esn] = result
            self.plot_ww_prediction(result)

        self.results_val = self.predict_batch(data.validation, "validation")
        for r in self.results_val.values():
            self.plot_ww_prediction(r)

        self.results_test = self.predict_batch(data.test, "test")
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

        fig, axs = plt.subplots(1, 2, figsize=(22, 6))
        fig.suptitle(f"WW Prediction — ESN {esn}", fontsize=16)

        self._plot_t45_with_events(axs[0], wwdf, reg, X, sv, slope, is_training)
        self._plot_detrended(axs[1], wwdf, reg, X, sv)

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

        rolling_mean = pd.Series(detrended).rolling(window=20, min_periods=1).mean()
        ax.plot(
            wwdf["Cycles_Since_New"].values,
            rolling_mean.values,
            color="tab:orange",
            linewidth=1.5,
            alpha=0.8,
            label="Rolling mean (20)",
        )

        ax.set_title("T45 Detrended — Fouling Accumulation")
        ax.set_xlabel("Cycles Since New")
        ax.set_ylabel("T45 - Trend")
        ax.legend(fontsize="small")
        ax.grid(True, alpha=0.3)
