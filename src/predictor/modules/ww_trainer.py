"""
ww_trainer.py - WWTrainer class.
Handles Water Wash (WW) prediction
"""

from __future__ import annotations

from math import e as _e
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

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
        self.trained_slope: float | None = None
        self.trained_gap: float = 21.0  # avg T45_res rise between WW events

    # ════════════════════════════════════════════════════════════════
    #  EFFECT REMOVAL
    # ════════════════════════════════════════════════════════════════

    @staticmethod
    def _preprocess_t45(df: pd.DataFrame) -> pd.DataFrame:
        """Sort, remove outliers, median per cycle."""
        df = df.sort_values(["Cycles_Since_New", "Snapshot"]).copy()
        df = Data.remove_outliers(df, threshold=2.6, sensor_cols=["Sensed_T45"])
        df = df.groupby(["Cycles_Since_New"], as_index=False).median(numeric_only=True)
        return df

    @staticmethod
    def _compute_combined_offset(
        df: pd.DataFrame, cols: list[str]
    ) -> pd.Series:
        """Compute the combined cumulative offset for multiple effect columns.

        For each column in *cols*, detects jumps at group boundaries
        **on the original, unmodified** ``Sensed_T45`` and accumulates
        an offset series.  All columns are evaluated on the same
        original data so they don't interfere with each other.

        Returns a Series aligned with *df*'s index containing the total
        offset to subtract from ``Sensed_T45``.
        """
        total_offset = pd.Series(0.0, index=df.index)
        for col in cols:
            if col not in df.columns:
                continue
            grp = df.groupby(col)["Sensed_T45"].agg(["first", "last"])
            grp = grp.sort_index()
            grp["prev_last"] = grp["last"].shift(1)
            grp["jump"] = grp["first"] - grp["prev_last"]
            grp["jump"] = grp["jump"].fillna(0)
            grp["cumulative_offset"] = grp["jump"].cumsum()
            offset_map = grp["cumulative_offset"].to_dict()
            total_offset = total_offset + df[col].map(offset_map).fillna(0)
        return total_offset

    @classmethod
    def remove_all_effects(cls, df: pd.DataFrame) -> pd.DataFrame:
        """Remove HPT + HPC + WW effects in one pass.

        Pre-computes all jump offsets on the original signal, then
        subtracts them once.
        """
        df = cls._preprocess_t45(df)
        cols = ["Cumulative_HPT_SVs", "Cumulative_HPC_SVs", "Cumulative_WWs"]
        offset = cls._compute_combined_offset(df, cols)
        df = df.copy()
        df["Sensed_T45"] = df["Sensed_T45"] - offset
        return df

    @classmethod
    def remove_hpt_hpc_effects(cls, df: pd.DataFrame) -> pd.DataFrame:
        """Remove only HPT + HPC effects in one pass.

        Pre-computes all jump offsets on the original signal, then
        subtracts them once.  WW effects are preserved.
        """
        df = cls._preprocess_t45(df)
        cols = ["Cumulative_HPT_SVs", "Cumulative_HPC_SVs"]
        offset = cls._compute_combined_offset(df, cols)
        df = df.copy()
        df["Sensed_T45"] = df["Sensed_T45"] - offset
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

        Chains two methods:
        1. HI-based detection (HPT + HPC jump events)
        2. Statistical anomaly correction (residual jumps)
        """
        df = self._preprocess_t45(df)
        df = self._remove_effects_via_hi(df)
        df = self._remove_effects_statistical(df)
        return df

    def _remove_effects_via_hi(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove effects detected via HI jumps (HPT + HPC).

        Uses ``calc_hi`` to compute HPT and HPC health indicators,
        then detects sudden jumps in both.  Each set of jumps defines
        event groups, and cumulative offsets are subtracted.
        """
        df = df.copy()

        # Detect HPT and HPC events
        ehpt, ehpc = self._get_events(df)

        # Merge all event indices
        all_events = set(ehpt.index) | set(ehpc.index)

        if not all_events:
            return df

        # Build event groups from merged events
        df["Event_Group"] = 0
        df.loc[df.index.isin(all_events), "Event_Group"] = 1
        df["Event_Group"] = df["Event_Group"].cumsum()

        # Compute and subtract cumulative offsets
        grp = df.groupby("Event_Group")["Sensed_T45"].agg(["first", "last"])
        grp["prev_last"] = grp["last"].shift(1)
        grp["jump"] = grp["first"] - grp["prev_last"]
        grp["jump"] = grp["jump"].fillna(0)
        grp["cumulative_offset"] = grp["jump"].cumsum()
        offset_map = grp["cumulative_offset"].to_dict()
        df["Sensed_T45"] = df["Sensed_T45"] - df["Event_Group"].map(offset_map)
        df = df.drop(columns=["Event_Group"])
        return df

    @staticmethod
    def _remove_effects_statistical(
        df: pd.DataFrame,
        z_threshold: float = 3.0,
        min_jump: float = 2.0,
    ) -> pd.DataFrame:
        """Remove residual jumps via statistical anomaly detection.

        Computes the first difference of ``Sensed_T45`` and flags
        any point whose absolute difference exceeds *z_threshold*
        standard deviations from the mean AND whose absolute value
        is above *min_jump*.  Each flagged jump is subtracted
        cumulatively to stitch the signal.

        Parameters
        ----------
        z_threshold : float
            Number of standard deviations for outlier detection.
        min_jump : float
            Minimum absolute jump size to correct (avoids over-fixing).
        """
        df = df.copy()
        t45 = df["Sensed_T45"].values.astype(float)

        diffs = np.diff(t45, prepend=t45[0])

        mu = np.mean(diffs)
        sigma = np.std(diffs)
        if sigma < 1e-8:
            return df

        z_scores = np.abs((diffs - mu) / sigma)

        # Flag anomalous jumps
        anomalies = (z_scores > z_threshold) & (np.abs(diffs) > min_jump)

        # Accumulate corrections: at each anomaly, subtract the jump
        corrections = np.where(anomalies, diffs - mu, 0.0)
        cumulative_correction = np.cumsum(corrections)

        df["Sensed_T45"] = t45 - cumulative_correction
        return df

    @staticmethod
    def _remove_outliers_dbscan(
        df: pd.DataFrame,
        eps: float = 0.5,
        min_samples: int = 5,
    ) -> pd.DataFrame:
        """Remove outliers using DBSCAN on (Cycles_Since_New, Sensed_T45).

        Points classified as noise (label = -1) are removed.
        Features are normalised with StandardScaler so that both
        dimensions contribute equally to distance.

        Parameters
        ----------
        eps : float
            DBSCAN neighbourhood radius (on scaled data).
        min_samples : int
            Minimum points to form a dense region.
        """
        if len(df) < min_samples:
            return df

        # We must drop NaNs for DBSCAN, but we only want to drop rows
        # where the specific features are NaN to avoid losing data based
        # on other columns.
        feature_cols = ["Cycles_Since_New", "Sensed_T45"]
        valid_idx = df.dropna(subset=feature_cols).index

        if len(valid_idx) < min_samples:
            return df

        features = df.loc[valid_idx, feature_cols].values
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)

        labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(
            features_scaled
        )

        # Keep valid points that are not noise, plus points that were
        # skipped (if we want to be conservative). Here we drop noise.
        noise_idx = valid_idx[labels == -1]
        return df.drop(index=noise_idx).copy()

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
        # factor = (_e**2) * factor_mult

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

            if (mc - msp) > 21:
                sv[idx] = val_t45
                reference_r = val_t45
                floor = val_t45

        return sv

    # ════════════════════════════════════════════════════════════════
    #  PREDICT — single engine
    # ════════════════════════════════════════════════════════════════
    def train_ww(
        self,
        engine_dfs: list[pd.DataFrame],
    ):
        """Train WW prediction parameters from training engines.

        Two-phase process:
        1. Remove HPT + HPC + WW effects from T45_res → fit global slope.
        2. Remove only HPT + HPC effects from T45_res → measure the average
           T45_res rise between WW events (the detection gap).
        """

        # ── Compute residuals for each engine ─────────────────────
        # train_ww receives raw engine DataFrames; we need to replace
        # Sensed_T45 (and other DEGRAD_VARS) with residuals from the
        # linear model, same as predict_ww does.
        res_dfs: list[pd.DataFrame] = []
        for edf in engine_dfs:
            engine_res = self.hi.residuals_single(edf)
            if engine_res is None:
                continue
            rdf = edf.copy()
            rdf[cfg.DEGRAD_VARS] = engine_res[cfg.DEGRAD_VARS].values
            if "Cycles" in rdf.columns and "Cycles_Since_New" not in rdf.columns:
                rdf = rdf.rename(columns={"Cycles": "Cycles_Since_New"})
            res_dfs.append(rdf)

        # ── PHASE 1: slope from fully-cleaned signal ─────────────
        print("\n  [Phase 1] Computing slope (HPT + HPC + WW removed)...")
        clean_esns: list[pd.DataFrame] = []
        for rdf in res_dfs:
            edf_c = self.remove_all_effects(rdf)
            edf_c = edf_c.dropna()
            clean_esns.append(edf_c)
        df_clean = pd.concat(clean_esns)

        X = df_clean["Cycles_Since_New"].values.reshape(-1, 1)
        Y = df_clean["Sensed_T45"].values
        reg = LinearRegression().fit(X, Y)
        self.trained_slope = float(reg.coef_[0])
        print(f"  Slope = {self.trained_slope:.5f}")

        # ── PHASE 2: gap from HPT/HPC-only cleaned signal ────────
        print("\n  [Phase 2] Computing WW gap (only HPT + HPC removed)...")
        partial_esns: list[pd.DataFrame] = []
        for rdf in res_dfs:
            edf_p = self.remove_hpt_hpc_effects(rdf)
            edf_p = edf_p.dropna()
            partial_esns.append(edf_p)
        df_partial = pd.concat(partial_esns)

        # Measure per-engine average rise between WW events
        all_rises: list[float] = []
        for esn_id in df_partial["ESN"].unique():
            edf = df_partial[df_partial["ESN"] == esn_id].copy()
            if "Cumulative_WWs" not in edf.columns:
                continue

            grp = edf.groupby("Cumulative_WWs")["Sensed_T45"].agg(
                ["first", "last"]
            )
            grp["rise"] = grp["last"] - grp["first"]
            # Skip the first segment (may be incomplete) and
            # non-positive rises (artefacts)
            valid = grp["rise"].iloc[1:]
            valid = valid[valid > 0]

            engine_avg = float(valid.mean()) if len(valid) > 0 else 0.0
            all_rises.extend(valid.tolist())

            print(
                f"    ESN {esn_id}: "
                f"{len(valid)} WW segments, "
                f"avg rise = {engine_avg:.2f}"
            )

            # ── Diagnostic plot per engine ────────────────────
            if cfg.PLOT_WW:
                self._plot_ww_training_gap(edf, esn_id, grp)

        if all_rises:
            self.trained_gap = float(np.mean(all_rises))
        else:
            self.trained_gap = 21.0  # fallback

        print(f"\n  ── Summary ──")
        print(f"  Slope    = {self.trained_slope:.5f}")
        print(f"  WW Gap   = {self.trained_gap:.2f} (avg T45_res rise)")

        return df_partial

    def _plot_ww_training_gap(
        self,
        edf: pd.DataFrame,
        esn_id,
        grp: pd.DataFrame,
    ) -> None:
        """Plot T45_res_c with WW event markers and rise values."""
        fig, ax = plt.subplots(figsize=(14, 6))

        # Scatter T45_res_c
        ax.scatter(
            edf["Cycles_Since_New"],
            edf["Sensed_T45"],
            s=2, alpha=0.6, color="tab:blue", label="T45_res_c",
        )

        # Mark WW boundaries and annotate rise values
        if "Cumulative_WWs" in edf.columns:
            ww_groups = edf.groupby("Cumulative_WWs")
            colors = plt.cm.Set2(np.linspace(0, 1, len(grp)))

            for i, (ww_id, ww_group) in enumerate(ww_groups):
                cycle_start = ww_group["Cycles_Since_New"].iloc[0]
                t45_start = ww_group["Sensed_T45"].iloc[0]
                cycle_end = ww_group["Cycles_Since_New"].iloc[-1]
                t45_end = ww_group["Sensed_T45"].iloc[-1]

                # Vertical line at WW event
                ax.axvline(
                    x=cycle_start, color="gray",
                    linestyle="--", alpha=0.5, linewidth=0.8,
                )

                # Mark start/end points
                ax.scatter(
                    [cycle_start], [t45_start],
                    color="green", s=40, zorder=5, marker="^",
                )
                ax.scatter(
                    [cycle_end], [t45_end],
                    color="red", s=40, zorder=5, marker="v",
                )

                # Annotate rise
                if ww_id in grp.index:
                    rise = grp.loc[ww_id, "rise"]
                    mid_cycle = (cycle_start + cycle_end) / 2
                    ax.annotate(
                        f"Δ={rise:.1f}",
                        xy=(mid_cycle, (t45_start + t45_end) / 2),
                        fontsize=7, ha="center",
                        bbox=dict(boxstyle="round,pad=0.2",
                                  fc="white", alpha=0.8),
                    )

        ax.set_title(f"WW Training Gap — ESN {esn_id}")
        ax.set_xlabel("Cycles Since New")
        ax.set_ylabel("T45_res_c (HPT+HPC removed)")
        ax.legend(fontsize="small")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        save_fig(fig, f"ww_training_gap_esn_{esn_id}")

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
            wwdf = self.remove_all_effects(wwdf)
        else:
            wwdf = self._preprocess_t45(wwdf)
            wwdf = self._remove_effects_via_hi(wwdf)

        # Remove outliers from T45_res_c via DBSCAN
        wwdf = self._remove_outliers_dbscan(wwdf)
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

        # Engine-specific linear trend (always fit for plotting/context)
        X = wwdf["Cycles_Since_New"].values.reshape(-1, 1)
        Y = wwdf["Sensed_T45"].values
        reg = LinearRegression().fit(X, Y)
        engine_slope = float(reg.coef_[0])

        slope = engine_slope
        if (
            not is_training
            and self.trained_slope is not None
            and not cfg.WW_DYNAMIC_SLOPE_DETECTION
        ):
            slope = self.trained_slope

        # Detect events using the trained gap
        sv = self._detect_events_real(wwdf, self.trained_gap)

        return {
            "esn": esn,
            "detected_events": sv,
            "n_events": len(sv),
            "slope": slope,
            "wwdf": wwdf,
            "regression": reg,
            "is_training": is_training,
        }

    def _detect_events_real(self, df: pd.DataFrame, gap: float) -> dict[Any, float]:
        sv: dict[Any, float] = {}
        reference_r: float | None = None
        for idx, row in df[["Sensed_T45"]].iterrows():
            val_t45 = row["Sensed_T45"]
            if reference_r is None:
                reference_r = val_t45
                floor = val_t45
                continue

            if val_t45 - floor > gap:
                sv[idx] = val_t45
                reference_r = val_t45
                floor = val_t45
        return sv

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

    def cycles_to_next_ww_from_end(
        self,
        ww_result: dict[str, Any],
    ) -> float:
        """Estimate cycles remaining until the next Water Wash.

        Uses the trained gap (average T45_res rise per WW cycle) as
        the trigger threshold, consistent with the detection logic.

        Parameters
        ----------
        ww_result : dict
            Output of ``predict_ww()``.

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

        trigger_threshold = self.trained_gap

        t45 = wwdf["Sensed_T45"]

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
        train_slopes: list[float] = []
        for esn in data.train["ESN"].unique():
            edf = data.train[data.train["ESN"] == esn]
            engine_res = self.hi.residuals_single(edf)
            if engine_res is None:
                continue
            result = self.predict_ww(edf, engine_res, esn)
            self.results_train[esn] = result
            train_slopes.append(float(result["slope"]))
            self.plot_ww_prediction(result)

        if train_slopes:
            self.trained_slope = float(np.mean(train_slopes))
            print(f"Global WW slope (training mean): {self.trained_slope:.6f}")
        else:
            self.trained_slope = None
            print("Global WW slope not available (no valid training engines).")

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

        subplots = []
        if cfg.PLOT_WW_T45_EVENTS:
            subplots.append(("t45", self._plot_t45_with_events))
        if cfg.PLOT_WW_T45_DEFECTED:
            subplots.append(("t45_res_defected", self._plot_t45_res_defected))
        if cfg.PLOT_WW_DETRENDED:
            subplots.append(("detrended", self._plot_detrended))
        if not subplots:
            return

        n = len(subplots)
        fig, axs = plt.subplots(1, n, figsize=(15 * n, 9))
        fig.suptitle(f"WW Prediction — ESN {esn}", fontsize=16)
        if n == 1:
            axs = [axs]
        for ax, (kind, fn) in zip(axs, subplots):
            if kind == "t45":
                fn(ax, wwdf, reg, X, sv, slope, is_training)
            elif kind == "t45_res_defected":
                fn(ax, wwdf, is_training)
            elif kind == "detrended":
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
    def _plot_t45_res_defected(
        ax: plt.Axes,
        wwdf: pd.DataFrame,
        is_training: bool,
    ) -> None:
        """Subplot: T45 residuals with effects removed."""

        ax.scatter(
            wwdf["Cycles_Since_New"],
            wwdf["Sensed_T45"],
            color="tab:blue",
            alpha=0.6,
            s=3,
            label="T45 (residuals)",
            zorder=2,
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

        ax.set_title("T45_{res} + WW events")
        ax.set_xlabel("Cycles Since New")
        ax.set_ylabel("T45_{res}")
        ax.legend(fontsize="small")
        ax.grid(True, alpha=0.3)

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
