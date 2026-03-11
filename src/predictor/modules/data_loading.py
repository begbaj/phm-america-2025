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

        return self.data

    def load_training(self) -> None:
        """Load and preprocess the training CSV."""
        df = pd.read_csv(cfg.DATA_TRAINING_DATA)
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
            df = Data.remove_outliers(df, cfg.SENSORS)
            df = Data.missingfill(df, align_cols=["Snapshot", "Cycles"]).dropna()
            frames.append(df)
        return frames
