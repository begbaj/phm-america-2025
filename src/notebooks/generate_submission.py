from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline

TARGET_COLS = ["Cycles_to_WW", "Cycles_to_HPC_SV", "Cycles_to_HPT_SV"]

BASE_FEATURE_CANDIDATES = [
    "Cycles",
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
]


def _normalize_cycle_column(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "Cycles_Since_New" in out.columns:
        out = out.rename(columns={"Cycles_Since_New": "Cycles"})
    if "Cycles" not in out.columns:
        raise ValueError("Cycle column not found: expected 'Cycles' or 'Cycles_Since_New'.")
    return out


def _sort_test_files(files: list[Path]) -> list[Path]:
    def _key(p: Path) -> int:
        match = re.search(r"(?:test|val)_(\d+)\.csv$", p.name)
        return int(match.group(1)) if match else 10**9

    return sorted(files, key=_key)


def _prepare_features(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    out = _normalize_cycle_column(df)

    for col in feature_cols:
        if col not in out.columns:
            out[col] = np.nan

    if "ESN" in out.columns:
        out[feature_cols] = out.groupby("ESN", dropna=False)[feature_cols].transform(
            lambda s: s.ffill().bfill()
        )

    out[feature_cols] = out[feature_cols].astype(float)
    return out


def _build_cycle_train(train_df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    work = _prepare_features(train_df, feature_cols)

    agg_map = {col: "median" for col in feature_cols + TARGET_COLS}
    cycle_df = work.groupby(["ESN", "Cycles"], as_index=False).agg(agg_map)
    cycle_df = cycle_df.dropna(subset=TARGET_COLS, how="any").reset_index(drop=True)
    return cycle_df


def _train_models(train_cycle_df: pd.DataFrame, feature_cols: list[str]) -> dict[str, Pipeline]:
    x_train = train_cycle_df[feature_cols]
    models: dict[str, Pipeline] = {}

    for target in TARGET_COLS:
        y_train = train_cycle_df[target].astype(float).values

        model = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "regressor",
                    LGBMRegressor(
                        objective="regression",
                        n_estimators=500,
                        learning_rate=0.05,
                        num_leaves=31,
                        min_child_samples=20,
                        subsample=0.9,
                        colsample_bytree=0.9,
                        random_state=42,
                        n_jobs=-1,
                    ),
                ),
            ]
        )

        model.fit(x_train, y_train)
        pred_train = np.clip(model.predict(x_train), 0.0, None)
        rmse = np.sqrt(mean_squared_error(y_train, pred_train))
        mae = mean_absolute_error(y_train, pred_train)
        print(f"{target}: train RMSE={rmse:.2f} MAE={mae:.2f}")

        models[target] = model

    return models


def _predict_single_file(file_path: Path, models: dict[str, Pipeline], feature_cols: list[str]) -> dict[str, float]:
    raw_df = pd.read_csv(file_path)
    work = _prepare_features(raw_df, feature_cols)

    cycle_df = work.groupby(["ESN", "Cycles"], as_index=False)[feature_cols].median()
    if cycle_df.empty:
        return {target: 0.0 for target in TARGET_COLS}

    last_per_esn = cycle_df.sort_values(["ESN", "Cycles"]).groupby("ESN", as_index=False).tail(1)

    out: dict[str, float] = {}
    for target, model in models.items():
        pred = model.predict(last_per_esn[feature_cols])
        out[target] = float(np.clip(np.nanmedian(pred), 0.0, None))

    return out


def _predict_files(file_list: list[Path], models: dict[str, Pipeline], feature_cols: list[str]) -> pd.DataFrame:
    """Predict for a list of individual engine files (test or validation)."""
    rows = []
    for file_path in file_list:
        preds = _predict_single_file(file_path, models, feature_cols)
        rows.append(
            {
                "file": file_path.stem,
                "Cycles_to_WW": preds["Cycles_to_WW"],
                "Cycles_to_HPC_SV": preds["Cycles_to_HPC_SV"],
                "Cycles_to_HPT_SV": preds["Cycles_to_HPT_SV"],
            }
        )
    return pd.DataFrame(rows, columns=["file", "Cycles_to_WW", "Cycles_to_HPC_SV", "Cycles_to_HPT_SV"])


def _build_training_submission(train_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Per il training usiamo i valori ground-truth: per ogni ESN prendiamo
    l'ultimo ciclo e leggiamo direttamente Cycles_to_WW/HPC_SV/HPT_SV.
    """
    work = _normalize_cycle_column(train_raw)
    # Ordiniamo e prendiamo l'ultimo snapshot dell'ultimo ciclo per ogni ESN
    work = work.sort_values(["ESN", "Cycles", "Snapshot"])
    last_rows = work.groupby("ESN", as_index=False).tail(1)

    rows = []
    for _, row in last_rows.iterrows():
        esn = int(row["ESN"])
        rows.append({
            "file": f"training_esn_{esn}",
            "Cycles_to_WW": row["Cycles_to_WW"],
            "Cycles_to_HPC_SV": row["Cycles_to_HPC_SV"],
            "Cycles_to_HPT_SV": row["Cycles_to_HPT_SV"],
        })
    return pd.DataFrame(rows, columns=["file", "Cycles_to_WW", "Cycles_to_HPC_SV", "Cycles_to_HPT_SV"])


def main() -> None:
    project_root = Path(__file__).resolve().parents[2]
    train_path = project_root / "Data" / "PHM2025_training_data" / "training_data.csv"
    test_dir = project_root / "Data" / "PHM2025_test_data"
    val_dir = project_root / "Data" / "PHM2025_validation_data"
    output_dir = project_root

    train_raw = pd.read_csv(train_path)

    feature_cols = [c for c in BASE_FEATURE_CANDIDATES if c in set(train_raw.columns) | {"Cycles"}]
    if "Cycles" not in feature_cols:
        feature_cols = ["Cycles", *feature_cols]

    train_cycle_df = _build_cycle_train(train_raw, feature_cols)
    print(f"Train cycle rows: {len(train_cycle_df)}")
    print(f"Feature columns ({len(feature_cols)}): {feature_cols}")

    models = _train_models(train_cycle_df, feature_cols)

    # ===== TRAINING =====
    # Ground-truth: ultimo ciclo di ogni ESN nel training
    print("\n=== TRAINING ===")
    train_sub = _build_training_submission(train_raw)
    train_out = output_dir / "submission_training.csv"
    train_sub.to_csv(train_out, index=False)
    print(f"Saved: {train_out}")
    print(train_sub.to_string(index=False))

    # ===== VALIDATION =====
    print("\n=== VALIDATION ===")
    val_files = _sort_test_files(list(val_dir.glob("val_*.csv")))
    val_sub = _predict_files(val_files, models, feature_cols)
    val_out = output_dir / "submission_validation.csv"
    val_sub.to_csv(val_out, index=False)
    print(f"Saved: {val_out}  ({len(val_sub)} files)")
    print(val_sub.head(5).to_string(index=False))

    # ===== TEST =====
    print("\n=== TEST ===")
    test_files = _sort_test_files(list(test_dir.glob("test_*.csv")))
    test_sub = _predict_files(test_files, models, feature_cols)
    test_out = output_dir / "submission.csv"
    test_sub.to_csv(test_out, index=False)
    print(f"Saved: {test_out}  ({len(test_sub)} files)")
    print(test_sub.head(5).to_string(index=False))


if __name__ == "__main__":
    main()
