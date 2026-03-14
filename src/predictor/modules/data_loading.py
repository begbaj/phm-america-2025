"""
data_loading.py - DataLoading class.

Handles loading of training, validation, and test data from CSV files.
Populates a ``Data`` instance with preprocessed DataFrames.
"""

from __future__ import annotations

import pandas as pd

from modules import config as cfg
from modules.data import Data


class DataLoading:
    """Load train / validation / test CSVs and populate a ``Data`` object.

    Parameters
    ----------
    data : Data
        The ``Data`` container to populate.
    """

    def __init__(self, data: Data) -> None:
        self.data = data

    # ──────────────────────── Public API ─────────────────────────────

    def load_all(
        self,
        val_range: range | list[int] | None = None,
        test_range: range | list[int] | None = None,
    ) -> Data:
        """Load training, validation, and test datasets.

        Parameters
        ----------
        val_range : range | list[int] | None
            Indices for validation files (default 0..47).
        test_range : range | list[int] | None
            Indices for test files (default 0..51).

        Returns
        -------
        Data
            The populated ``Data`` instance.
        """
        if val_range is None:
            val_range = range(0, 48)
        if test_range is None:
            test_range = range(0, 52)

        self.load_training()
        self.load_validation(val_range)
        self.load_testing(test_range)
        self._align_common_sensors()

        return self.data

    def load_training(self) -> None:
        """Load and preprocess the training CSV."""
        df = pd.read_csv(cfg.DATA_TRAINING_DATA)
        df = Data.aggregate_by_cycle(df)
        df = Data.remove_outliers(df, cfg.SENSORS)
        df = Data.missingfill(df).dropna()
        self.data.train = df

    def load_validation(self, indices: range | list[int]) -> None:
        """Load validation files as a list of per-engine DataFrames."""
        self.data.validation = self._load_multi(
            indices,
            cfg.DATA_VALIDATION_PATH,
            prefix="val",
        )

    def load_testing(self, indices: range | list[int]) -> None:
        """Load test files as a list of per-engine DataFrames."""
        self.data.test = self._load_multi(
            indices,
            cfg.DATA_TESTING_PATH,
            prefix="test",
        )

    # ──────────────────────── Private helpers ───────────────────────

    @staticmethod
    def _load_multi(
        indices: range | list[int],
        base_path: str,
        prefix: str,
    ) -> list[pd.DataFrame]:
        """Load multiple CSV files, preprocess, and return as list."""
        frames: list[pd.DataFrame] = []
        for i in indices:
            path = f"{base_path}{prefix}_{i}.csv"
            df = pd.read_csv(path)
            df = Data.aggregate_by_cycle(df)
            df = Data.remove_outliers(df, cfg.SENSORS)
            df = Data.missingfill(df).dropna()
            frames.append(df)
        return frames

    def _align_common_sensors(self) -> None:
        """Keep only sensors present in train, validation and test splits."""
        datasets: list[pd.DataFrame] = [self.data.train] + self.data.validation + self.data.test
        if not datasets:
            return

        original_sensors = list(cfg.SENSORS)
        common = set(original_sensors)
        for df in datasets:
            common &= set(df.columns)

        common_sensors = [s for s in original_sensors if s in common]
        if not common_sensors:
            raise ValueError("No common sensor columns across train/validation/test.")

        dropped = [s for s in original_sensors if s not in common_sensors]
        if dropped:
            print(f"  Dropping non-common sensors: {dropped}")

        sensor_set = set(original_sensors)

        self.data.train = self._drop_sensor_columns(self.data.train, common_sensors, sensor_set)
        self.data.validation = [
            self._drop_sensor_columns(df, common_sensors, sensor_set)
            for df in self.data.validation
        ]
        self.data.test = [
            self._drop_sensor_columns(df, common_sensors, sensor_set)
            for df in self.data.test
        ]

        cfg.SENSORS = common_sensors
        cfg.OPERATING_VARS = [s for s in cfg.OPERATING_VARS if s in common_sensors]
        cfg.DEGRAD_VARS = [s for s in cfg.DEGRAD_VARS if s in common_sensors]
        cfg.ALL_VARS = cfg.OPERATING_VARS + cfg.DEGRAD_VARS

        if not cfg.OPERATING_VARS or not cfg.DEGRAD_VARS:
            raise ValueError(
                "Sensor intersection removed all operating or degradation variables."
            )

    @staticmethod
    def _drop_sensor_columns(
        df: pd.DataFrame,
        keep_sensors: list[str],
        sensor_set: set[str],
    ) -> pd.DataFrame:
        """Drop sensor columns that are not in the shared sensor set."""
        drop_cols = [c for c in df.columns if c in sensor_set and c not in keep_sensors]
        if not drop_cols:
            return df
        return df.drop(columns=drop_cols)
