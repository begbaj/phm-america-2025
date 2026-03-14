"""
data.py - Data class.

Contains the original DataFrames of loaded datasets and auxiliary methods
for manipulating, accessing, checking, and preprocessing them.
Preprocessing methods contain only atomic operations.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import NearestNeighbors

from modules import config as cfg


class Data:
    """Holds train/val/test DataFrames and exposes atomic preprocessing
    and data-access helpers.

    Attributes
    ----------
    train : pd.DataFrame
        Full training set (all ESNs).
    train_data : pd.DataFrame
        Training subset (after LOO / healthy-cycle filtering).
    test_loo : pd.DataFrame
        Leave-one-out testing ESN from training set.
    validation : list[pd.DataFrame]
        Per-engine validation DataFrames.
    test : list[pd.DataFrame]
        Per-engine test DataFrames.
    """

    def __init__(self) -> None:
        self.train: pd.DataFrame = pd.DataFrame()
        self.train_data: pd.DataFrame = pd.DataFrame()
        self.test_loo: pd.DataFrame = pd.DataFrame()
        self.validation: list[pd.DataFrame] = []
        self.test: list[pd.DataFrame] = []

    # ──────────────────────── Atomic preprocessing ───────────────────

    @staticmethod
    def aggregate_by_cycle(
        df: pd.DataFrame,
        cycle_col: str | None = None,
        sensor_cols: list[str] | None = None,
    ) -> pd.DataFrame:
        """Aggregate snapshots to one row per engine-cycle.

        Sensor columns are averaged; label/metadata columns keep the first
        value within each engine-cycle block.
        """
        if df.empty:
            return df

        if cycle_col is None:
            if "Cycles_Since_New" in df.columns:
                cycle_col = "Cycles_Since_New"
            elif "Cycles" in df.columns:
                cycle_col = "Cycles"
            else:
                return df.copy()

        group_cols: list[str] = []
        if "ESN" in df.columns:
            group_cols.append("ESN")
        if cycle_col in df.columns:
            group_cols.append(cycle_col)

        if not group_cols:
            return df.copy()

        if sensor_cols is None:
            base_sensors = list(cfg.SENSORS)
        else:
            base_sensors = list(sensor_cols)

        sensor_mean_cols = {
            c
            for c in (base_sensors + list(cfg.OPERATING_VARS) + list(cfg.DEGRAD_VARS))
            if c in df.columns and c not in group_cols
        }

        agg_map: dict[str, str] = {}
        for col in df.columns:
            if col in group_cols:
                continue
            if col in sensor_mean_cols and pd.api.types.is_numeric_dtype(df[col]):
                agg_map[col] = "mean"
            elif col.startswith("Cumulative_") or col.startswith("Cycles_to_"):
                agg_map[col] = "first"
            elif col == "Snapshot":
                agg_map[col] = "first"
            elif pd.api.types.is_numeric_dtype(df[col]):
                agg_map[col] = "mean"
            else:
                agg_map[col] = "first"

        out = df.groupby(group_cols, as_index=False).agg(agg_map)
        return out.sort_values(group_cols).reset_index(drop=True)

    @staticmethod
    def remove_outliers(
        df: pd.DataFrame,
        sensor_cols: list[str] | None = None,
        threshold: float = 3,
        method: str = "zscore",
    ) -> pd.DataFrame:
        """Replace outlier values with NaN.

        Parameters
        ----------
        df : pd.DataFrame
        sensor_cols : list[str] | None
            Columns to check. Defaults to ``cfg.SENSORS``.
        threshold : float
            Z-score or IQR multiplier threshold.
        method : str
            ``'zscore'``, ``'iqr'``, or ``'isoforest'``.
        """
        df_out = df.copy()

        if sensor_cols is None:
            target_sensors = list(cfg.SENSORS)
        else:
            target_sensors = list(sensor_cols)
        target_sensors = [s for s in target_sensors if s in df_out.columns]

        if method == "zscore":
            for sensor in target_sensors:
                series = df_out[sensor]
                if series.dropna().empty:
                    continue
                z_scores = np.abs(stats.zscore(series, nan_policy="omit"))
                df_out.loc[z_scores > threshold, sensor] = np.nan

        elif method == "iqr":
            for sensor in target_sensors:
                q1 = df_out[sensor].quantile(0.25)
                q3 = df_out[sensor].quantile(0.75)
                iqr = q3 - q1
                lower = q1 - threshold * iqr
                upper = q3 + threshold * iqr
                df_out.loc[
                    (df_out[sensor] < lower) | (df_out[sensor] > upper), sensor
                ] = np.nan

        elif method == "isoforest":
            for sensor in target_sensors:
                series_nonan = df_out[sensor].dropna()
                if series_nonan.empty:
                    continue
                data = series_nonan.values.reshape(-1, 1)
                iso = IsolationForest(contamination="auto", random_state=42)
                preds = iso.fit_predict(data)
                outlier_indices = series_nonan.index[preds == -1]
                df_out.loc[outlier_indices, sensor] = np.nan

        return df_out

    @staticmethod
    def missingfill(
        df: pd.DataFrame,
        align_cols: list[str] | None = None,
        sensor_cols: list[str] | None = None,
    ) -> pd.DataFrame:
        """Fill missing values using interpolation, fleet mean and ffill/bfill.

        Parameters
        ----------
        df : pd.DataFrame
        align_cols : list[str] | None
            Columns for grouping when computing fleet mean.
            Tries ``['Snapshot', 'Cycles_Since_New']``,
            ``['Snapshot', 'Cycles']``, then cycle-only alternatives.
        sensor_cols : list[str] | None
            Columns to fill. Defaults to ``cfg.SENSORS``.
        """
        align_candidates = [
            ["Snapshot", "Cycles_Since_New"],
            ["Snapshot", "Cycles"],
            ["Cycles_Since_New"],
            ["Cycles"],
        ]

        if align_cols is None:
            align_cols = []
            for cand in align_candidates:
                if all(c in df.columns for c in cand):
                    align_cols = cand
                    break

        if sensor_cols is None:
            raw_sensors = list(cfg.SENSORS)
        else:
            raw_sensors = list(sensor_cols)
        valid_cols = [s for s in raw_sensors if s in df.columns]
        if not valid_cols:
            return df

        df_out = df.copy()

        cycle_col = None
        if "Cycles_Since_New" in df_out.columns:
            cycle_col = "Cycles_Since_New"
        elif "Cycles" in df_out.columns:
            cycle_col = "Cycles"

        if "ESN" in df_out.columns and cycle_col is not None:
            df_out = df_out.sort_values(["ESN", cycle_col]).copy()
        elif cycle_col is not None:
            df_out = df_out.sort_values([cycle_col]).copy()

        # Per-ESN interpolation
        if "ESN" in df_out.columns:
            for col in valid_cols:
                df_out[col] = df_out.groupby("ESN")[col].transform(
                    lambda x: x.interpolate(method="linear", limit_direction="both")
                )
        elif cycle_col is not None:
            df_out[valid_cols] = df_out[valid_cols].interpolate(
                method="linear", limit_direction="both"
            )

        # Fleet mean fill
        if align_cols and all(c in df_out.columns for c in align_cols):
            try:
                fleet_means = df_out.groupby(align_cols)[valid_cols].transform("mean")
                df_out[valid_cols] = df_out[valid_cols].fillna(fleet_means)
            except Exception:
                pass

        # Per-ESN forward + backward fill
        if "ESN" in df_out.columns:
            for col in valid_cols:
                df_out[col] = df_out.groupby("ESN")[col].transform(lambda x: x.ffill())
                df_out[col] = df_out.groupby("ESN")[col].transform(lambda x: x.bfill())

        return df_out

    @staticmethod
    def minmax(df: pd.DataFrame, column: str) -> pd.Series:
        """Min-max normalise a single column."""
        col_min = df[column].min()
        col_max = df[column].max()
        denom = col_max - col_min
        if denom == 0:
            return pd.Series(0.0, index=df.index)
        return (df[column] - col_min) / denom

    @staticmethod
    def minmax_all(df: pd.DataFrame) -> pd.DataFrame:
        """Min-max normalise every column."""
        newdf = pd.DataFrame()
        for column in df.columns:
            col_min = df[column].min()
            col_max = df[column].max()
            denom = col_max - col_min
            if denom == 0:
                newdf[column] = 0.0
            else:
                newdf[column] = (df[column] - col_min) / denom
        return newdf

    @staticmethod
    def normalize(col: pd.Series) -> pd.DataFrame:
        """Normalise a Series into a 1-column DataFrame."""
        col_min, col_max = col.min(), col.max()
        denom = col_max - col_min
        if denom == 0:
            normed = pd.Series(0.0, index=col.index)
        else:
            normed = (col - col_min) / denom
        return normed.to_frame()

    @staticmethod
    def median_norm(df: pd.DataFrame) -> pd.DataFrame:
        """Subtract column medians (first 7 columns)."""
        df = df.copy()
        for i in range(min(7, df.shape[1])):
            m = df.iloc[:, i].median()
            df.iloc[:, i] -= m
        return df

    @staticmethod
    def get_slope(y) -> float:
        """Linear regression slope for a rolling window."""
        if len(y) < 2:
            return 0.0
        if np.all(y == y[0]) or not np.all(np.isfinite(y)):
            return 0.0
        x = np.arange(len(y))
        x_mean = x.mean()
        y_mean = y.mean()
        denom = np.sum((x - x_mean) ** 2)
        if denom == 0:
            return 0.0
        return float(np.sum((x - x_mean) * (y - y_mean)) / denom)

    # ──────────────────────── Data preparation ──────────────────────

    def prepare_training(self) -> None:
        """Build ``train_data`` and ``test_loo`` from ``train``,
        applying INCLUDE_TEST, CYCLES_HEALTHY, and SMOTE settings from config.
        """
        if cfg.INCLUDE_TEST:
            self.train_data = self.train.copy()
        else:
            self.train_data = self.train[self.train["ESN"] != cfg.TESTING_ESN].copy()

        self.test_loo = self.train[self.train["ESN"] == cfg.TESTING_ESN].copy()

        if cfg.CYCLES_HEALTHY > 0:
            self.train_data = (
                self.train_data.groupby("ESN")
                .head(cfg.CYCLES_HEALTHY)
                .reset_index(drop=True)
                .copy()
            )
        else:
            self.train_data = self.train_data.sort_values(
                ["ESN", "Cycles_Since_New", "Snapshot"]
            )

        if cfg.SMOTE:
            self._apply_smote()

    def _apply_smote(self) -> None:
        """SMOTE-based data augmentation on ``train_data``."""
        newdf: list[pd.DataFrame] = []
        for esn in self.train_data["ESN"].unique():
            cur = self.train_data[self.train_data["ESN"] == esn]
            nbrs = NearestNeighbors(n_neighbors=min(5, len(cur))).fit(cur[cfg.ALL_VARS])
            _, idx = nbrs.kneighbors(cur[cfg.ALL_VARS])
            for i in range(cfg.AUGMENTED_COUNT):
                neighbor_offsets = np.random.randint(1, idx.shape[1], size=len(cur))
                neighbor_indices = idx[np.arange(len(cur)), neighbor_offsets]
                diff = (
                    self.train_data.iloc[neighbor_indices][cfg.ALL_VARS].values
                    - cur[cfg.ALL_VARS].values
                )
                new_vals = cur[cfg.ALL_VARS].values + diff * np.random.rand(len(cur), 1)
                aug_df = cur.copy()
                aug_df[cfg.ALL_VARS] = new_vals
                aug_df["ESN"] = f"aug_{i}_{esn}"
                newdf.append(aug_df)
        self.train_data = pd.concat([self.train_data] + newdf, ignore_index=True)

    # ──────────────────────── Accessors ─────────────────────────────

    def esn_list(self, dataset: str = "train") -> list:
        """Return unique ESNs from the chosen dataset."""
        ds = self._resolve_dataset(dataset)
        if isinstance(ds, list):
            esns = set()
            for d in ds:
                esns.update(d["ESN"].unique())
            return sorted(esns)
        return sorted(ds["ESN"].unique())

    def get_engine(self, esn, dataset: str = "train") -> pd.DataFrame | None:
        """Return rows for a single ESN from the chosen dataset."""
        ds = self._resolve_dataset(dataset)
        if isinstance(ds, list):
            for d in ds:
                mask = d["ESN"] == esn
                if mask.any():
                    return d[mask].copy()
            return None
        mask = ds["ESN"] == esn
        if mask.any():
            return ds[mask].copy()
        return None

    def _resolve_dataset(self, name: str):
        mapping = {
            "train": self.train,
            "train_data": self.train_data,
            "test_loo": self.test_loo,
            "validation": self.validation,
            "test": self.test,
        }
        ds = mapping.get(name)
        if ds is None:
            raise ValueError(
                f"Unknown dataset '{name}'. Choose from {list(mapping.keys())}."
            )
        return ds
