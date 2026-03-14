#!/usr/bin/env python3
"""
main.py - Full PHM America 2025 prediction pipeline.

Orchestrates every step of the prediction pipeline:
    1. Load data
    2. Prepare training subset
    3. Train nominal-behaviour linear models (ensemble)
    4. Compute residuals for all datasets
    5. Optimise HI coefficients (differential_evolution)
    6. Train LGBM cycle classifier
    7. Train LGBM gap correction
    8. Infer Cycles_to_HPT_SV / Cycles_to_HPC_SV on val & test
    9. Predict Water Wash (WW) events
   10. Assemble and save submission.csv

Usage
-----
    cd src/notebooks
    python -m predictor.main
"""

from __future__ import annotations
import argparse
import os
import sys
from pathlib import Path

from modules.ww_trainer import WWTrainer
from modules.lgbm_gap_correction import LGBMGapCorrection
from modules.lgbm_classifier import LGBMCycleClassifier
from modules.hi_trainer import HITrainer
from modules.data_loading import DataLoading
from modules.data import Data
from modules import config as cfg

import time
import warnings
import pandas as pd
import numpy as np

warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="PHM America 2025 prediction pipeline")

    # Skip options for optimization
    p.add_argument("--skip-validation", action="store_true",
                    help="Skip validation step")
    p.add_argument("--skip-test", action="store_true", 
                    help="Skip test step")
    p.add_argument("--skip-ww", action="store_true",
                    help="Skip water wash prediction")
    p.add_argument("--skip-submission", action="store_true",
                    help="Skip submission generation")
    p.add_argument("--skip-plotting", action="store_true",
                    help="Skip all plotting")
    p.add_argument("--windowtraining-mode", action="store_true",
                    help="Output RMSE/MAE via LOEO CV on training set (implies --skip-validation --skip-test --skip-ww --skip-submission)")
    p.add_argument("--windowtraining-include-ww", action="store_true",
                    help="Include WW proxy metric in LOEO windowtraining output")
    
    return p.parse_args()


def _cycle_col(df: pd.DataFrame) -> str:
    """Return cycle column name available in dataframe."""
    if "Cycles_Since_New" in df.columns:
        return "Cycles_Since_New"
    if "Cycles" in df.columns:
        return "Cycles"
    raise ValueError("No cycle column found (Cycles_Since_New/Cycles).")


def _final_targets(engine_df: pd.DataFrame) -> tuple[float, float]:
    """Return final ground-truth HPT/HPC targets for one engine dataframe."""
    cycle_col = _cycle_col(engine_df)
    sorted_df = engine_df.sort_values(cycle_col)
    return (
        float(sorted_df["Cycles_to_HPT_SV"].iloc[-1]),
        float(sorted_df["Cycles_to_HPC_SV"].iloc[-1]),
    )


def _prepare_fold_train_data(train_df: pd.DataFrame, holdout_esn) -> pd.DataFrame:
    """Build training subset for one LOEO fold."""
    fold_train = train_df[train_df["ESN"] != holdout_esn].copy()
    if fold_train.empty:
        return fold_train

    if cfg.CYCLES_HEALTHY > 0:
        fold_train = (
            fold_train.groupby("ESN")
            .head(cfg.CYCLES_HEALTHY)
            .reset_index(drop=True)
            .copy()
        )
    else:
        cycle_col = _cycle_col(fold_train)
        sort_cols = ["ESN", cycle_col]
        if "Snapshot" in fold_train.columns:
            sort_cols.append("Snapshot")
        fold_train = fold_train.sort_values(sort_cols).reset_index(drop=True)

    return fold_train


def _proxy_true_cycles_to_ww(engine_df: pd.DataFrame) -> float:
    """Estimate a proxy target for cycles-to-next-WW from historical WW intervals."""
    if "Cumulative_WWs" not in engine_df.columns:
        return float("nan")

    cycle_col = _cycle_col(engine_df)
    sdf = engine_df.sort_values(cycle_col)
    ww_transitions = sdf.loc[
        sdf["Cumulative_WWs"].diff().fillna(0) > 0,
        cycle_col,
    ].to_numpy(dtype=float)

    if len(ww_transitions) < 2:
        return float("nan")

    intervals = np.diff(ww_transitions)
    intervals = intervals[intervals > 0]
    if len(intervals) == 0:
        return float("nan")

    expected_interval = float(np.median(intervals))
    end_cycle = float(sdf[cycle_col].iloc[-1])
    elapsed_since_last_ww = end_cycle - float(ww_transitions[-1])
    return max(expected_interval - elapsed_since_last_ww, 0.0)


def _fold_metrics_from_results(
    fold_df: pd.DataFrame,
    include_ww: bool = False,
) -> dict[str, dict[str, float]]:
    """Compute RMSE/MAE from fold prediction records."""
    metrics: dict[str, dict[str, float]] = {}
    mapping = {
        "Cycles_to_HPT_SV": ("pred_hpt", "true_hpt"),
        "Cycles_to_HPC_SV": ("pred_hpc", "true_hpc"),
    }
    if include_ww:
        mapping["Cycles_to_WW"] = ("pred_ww", "true_ww")

    for target, (pred_col, true_col) in mapping.items():
        if pred_col not in fold_df.columns or true_col not in fold_df.columns:
            metrics[target] = {"RMSE": float("inf"), "MAE": float("inf")}
            continue

        valid = fold_df[[pred_col, true_col]].dropna()
        if len(valid):
            finite_mask = np.isfinite(valid[pred_col].to_numpy()) & np.isfinite(
                valid[true_col].to_numpy()
            )
            valid = valid.loc[finite_mask]
        if valid.empty:
            metrics[target] = {"RMSE": float("inf"), "MAE": float("inf")}
            continue

        pred_arr = valid[pred_col].to_numpy(dtype=float)
        true_arr = valid[true_col].to_numpy(dtype=float)
        rmse = float(np.sqrt(np.mean((true_arr - pred_arr) ** 2)))
        mae = float(np.mean(np.abs(true_arr - pred_arr)))
        metrics[target] = {"RMSE": rmse, "MAE": mae}

    return metrics


def run_windowtraining_loeo_cv(
    train_df: pd.DataFrame,
    include_ww: bool = False,
) -> tuple[dict[str, dict[str, float]], pd.DataFrame, dict[str, int]]:
    """Run leave-one-engine-out CV using full training pipeline per fold."""
    esn_values = list(train_df["ESN"].dropna().unique())
    if len(esn_values) < 2:
        raise ValueError("LOEO requires at least 2 distinct ESN values in training data.")

    fold_records: list[dict[str, object]] = []
    fold_errors = 0

    print(f"  LOEO folds: {len(esn_values)}")
    for fold_idx, holdout_esn in enumerate(esn_values, start=1):
        print(f"  Fold {fold_idx}/{len(esn_values)} — holdout ESN {holdout_esn}")
        holdout_df = train_df[train_df["ESN"] == holdout_esn].copy()
        fold_train = _prepare_fold_train_data(train_df, holdout_esn)

        if holdout_df.empty or fold_train.empty:
            fold_errors += 1
            print(f"    Skip fold ESN {holdout_esn}: empty holdout/train subset")
            continue

        try:
            fold_data = Data()
            fold_data.train = fold_train
            fold_data.train_data = fold_train.copy()
            fold_data.test_loo = holdout_df.copy()
            fold_data.validation = []
            fold_data.test = []

            if cfg.SMOTE:
                fold_data._apply_smote()

            fold_hi = HITrainer(fold_data)
            fold_hi.train_linear_models()
            fold_hi.train_coefficients()

            fold_clf = LGBMCycleClassifier(fold_hi)
            fold_clf.train()

            fold_gap = LGBMGapCorrection(fold_hi, fold_clf)
            fold_gap.train(run_logo_validation=False)

            pred_df = fold_gap.predict_all_engines([holdout_df])
            if pred_df.empty:
                fold_errors += 1
                print(f"    Fold ESN {holdout_esn}: no predictions")
                continue

            pred_row = pred_df.iloc[0]
            true_hpt, true_hpc = _final_targets(holdout_df)

            pred_ww = float("nan")
            true_ww = float("nan")
            if include_ww:
                fold_ww = WWTrainer(fold_hi)

                if cfg.WW_USE_TRAINING_MEAN_SLOPE:
                    fold_ww.clear_training_slopes()
                    for tr_esn in fold_train["ESN"].dropna().unique():
                        tr_df = fold_train[fold_train["ESN"] == tr_esn]
                        tr_res = fold_hi.residuals_single(tr_df)
                        if tr_res is None:
                            continue
                        tr_ww_result = fold_ww.predict_ww(
                            tr_df,
                            tr_res,
                            tr_esn,
                            use_training_mean_slope=False,
                        )
                        fold_ww.register_training_slope(float(tr_ww_result["slope"]))
                    fold_ww.finalize_training_mean_slope()

                holdout_res = fold_hi.residuals_single(holdout_df)
                if holdout_res is not None:
                    ww_result = fold_ww.predict_ww(
                        holdout_df,
                        holdout_res,
                        holdout_esn,
                        use_training_mean_slope=cfg.WW_USE_TRAINING_MEAN_SLOPE,
                    )
                    pred_ww = float(fold_ww.cycles_to_next_ww_from_end(ww_result))
                true_ww = _proxy_true_cycles_to_ww(holdout_df)

            fold_records.append(
                {
                    "ESN": holdout_esn,
                    "pred_hpt": float(pred_row["Cycles_to_HPT_SV"]),
                    "pred_hpc": float(pred_row["Cycles_to_HPC_SV"]),
                    "true_hpt": true_hpt,
                    "true_hpc": true_hpc,
                    "pred_ww": pred_ww,
                    "true_ww": true_ww,
                    "confidence": str(pred_row.get("confidence", "ok")),
                }
            )
        except Exception as exc:
            fold_errors += 1
            print(f"    Fold ESN {holdout_esn}: ERROR {exc}")

    fold_df = pd.DataFrame(fold_records)
    metrics = _fold_metrics_from_results(fold_df, include_ww=include_ww)
    meta = {
        "folds_total": len(esn_values),
        "folds_success": len(fold_df),
        "folds_error": fold_errors,
        "fallback_count": int((fold_df["confidence"] != "ok").sum()) if "confidence" in fold_df.columns else 0,
        "ww_valid": int(
            fold_df[["pred_ww", "true_ww"]].dropna().shape[0]
        ) if include_ww and {"pred_ww", "true_ww"}.issubset(fold_df.columns) else 0,
    }
    return metrics, fold_df, meta


def main() -> None:
    args = parse_args()

    # Handle windowtraining mode
    if args.windowtraining_mode:
        args.skip_validation = True
        args.skip_test = True
        args.skip_ww = True
        args.skip_submission = True

    # Handle skip plotting
    if args.skip_plotting:
        cfg.PLOT_RESIDUALS = False
        cfg.PLOT_TRAINING_HI = False
        cfg.PLOT_GAP_BEFORE_AFTER = False
        cfg.PLOT_GAP_RESULTS = False
        cfg.PLOT_WW = False

    if args.windowtraining_mode:
        print(
            "  WINDOWTRAINING MODE: LOEO on training set, output RMSE/MAE"
            f" (include WW={args.windowtraining_include_ww})"
        )
    else:
        print(f"  Skip validation: {args.skip_validation}")
        print(f"  Skip test: {args.skip_test}")
        print(f"  Skip WW: {args.skip_ww}")
        print(f"  Skip submission: {args.skip_submission}")
        print(f"  Skip plotting: {args.skip_plotting}")

    t0 = time.time()

    # ──────────────────────────────────────────────────────────────
    # 1. LOAD DATA
    # ──────────────────────────────────────────────────────────────
    print("=" * 60)
    print("STEP 1 — LOAD DATA")
    print("=" * 60)
    data = Data()
    loader = DataLoading(data)
    loader.load_all(
        val_range=range(0, 48),
        test_range=range(0, 52),
    )
    print(
        f"  Training rows:    {len(data.train)}\n"
        f"  Validation files: {len(data.validation)}\n"
        f"  Test files:       {len(data.test)}"
    )

    # ──────────────────────────────────────────────────────────────
    # 2. PREPARE TRAINING SUBSET
    # ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 2 — PREPARE TRAINING SUBSET")
    print("=" * 60)
    data.prepare_training()
    print(
        f"  train_data rows: {len(data.train_data)}\n"
        f"  test_loo rows:   {len(data.test_loo)}\n"
        f"  TESTING_ESN:     {cfg.TESTING_ESN}\n"
        f"  CYCLES_HEALTHY:  {cfg.CYCLES_HEALTHY}\n"
        f"  ENSAMBLE:        {cfg.ENSAMBLE}\n"
        f"  SEPARATE_MODELS: {cfg.SEPARATE_MODELS}"
    )

    # ──────────────────────────────────────────────────────────────
    # 3-5. HITrainer: LINEAR MODELS + RESIDUALS + HI COEFFICIENTS
    # ──────────────────────────────────────────────────────────────
    hi = HITrainer(data)
    _hi_saved = Path(cfg.MODELS_DIR, "hi_models.pkl").exists()

    if cfg.LOAD_HI_TRAINER and _hi_saved:
        print("\n" + "=" * 60)
        print("STEPS 3-5 — LOADING HITrainer")
        print("=" * 60)
        hi.load()
        print(f"  Models: {list(hi.models.keys())}")
        print(f"  chpt = {hi.chpt}")
        print(f"  chpc = {hi.chpc}")
        hi.compute_all_residuals()
    else:
        if cfg.LOAD_HI_TRAINER and not _hi_saved:
            print("\n  [!] LOAD_HI_TRAINER=True but no saved models — training...")

        # 3. TRAIN LINEAR MODELS
        print("\n" + "=" * 60)
        print("STEP 3 — TRAIN LINEAR MODELS")
        print("=" * 60)
        hi.train_linear_models()
        print(f"  Models: {list(hi.models.keys())}")

        # 4. COMPUTE RESIDUALS
        print("\n" + "=" * 60)
        print("STEP 4 — COMPUTE RESIDUALS")
        print("=" * 60)
        hi.compute_all_residuals()
        if cfg.PLOT_RESIDUALS:
            if hi._res_train_healthy is not None:
                hi.plot_residuals(hi._res_train_healthy, title_suffix="Training (healthy)")
            if hi._res_train is not None:
                hi.plot_residuals(hi._res_train, title_suffix="Training (full)")
            if hi._res_test_loo is not None:
                hi.plot_residuals(hi._res_test_loo, title_suffix="Leave-One-Out ESN")

        # 5. OPTIMISE HI COEFFICIENTS
        print("\n" + "=" * 60)
        print("STEP 5 — HI COEFFICIENT OPTIMISATION")
        print("=" * 60)
        hi.train_coefficients()
        print(f"  chpt = {hi.chpt}")
        print(f"  chpc = {hi.chpc}")
        hi.plot_training_hi() if cfg.PLOT_TRAINING_HI else None
        hi.save()

    # ──────────────────────────────────────────────────────────────
    # 6. LGBM CYCLE CLASSIFIER
    # ──────────────────────────────────────────────────────────────
    classifier = LGBMCycleClassifier(hi)
    _clf_saved = Path(cfg.MODELS_DIR, "clf_hpt.pkl").exists()

    if cfg.LOAD_LGBM_CLASSIFIER and _clf_saved:
        print("\n" + "=" * 60)
        print("STEP 6 — LOADING LGBM CYCLE CLASSIFIER")
        print("=" * 60)
        classifier.load()
    else:
        if cfg.LOAD_LGBM_CLASSIFIER and not _clf_saved:
            print("\n  [!] LOAD_LGBM_CLASSIFIER=True but no saved models — training...")
        print("\n" + "=" * 60)
        print("STEP 6 — TRAIN LGBM CYCLE CLASSIFIER")
        print("=" * 60)
        classifier.train()
        classifier.save()

    # ──────────────────────────────────────────────────────────────
    # 7. LGBM GAP CORRECTION
    # ──────────────────────────────────────────────────────────────
    gap = LGBMGapCorrection(hi, classifier)
    _gap_saved = Path(cfg.MODELS_DIR, "lgbm_gap_hpt.pkl").exists()

    if cfg.LOAD_LGBM_GAP and _gap_saved:
        print("\n" + "=" * 60)
        print("STEP 7 — LOADING LGBM GAP CORRECTION")
        print("=" * 60)
        gap.load()
    else:
        if cfg.LOAD_LGBM_GAP and not _gap_saved:
            print("\n  [!] LOAD_LGBM_GAP=True but no saved models — training...")
        print("\n" + "=" * 60)
        print("STEP 7 — TRAIN LGBM GAP CORRECTION")
        print("=" * 60)
        gap.train()
        gap.save()

    # Plot training before/after gap correction
    if cfg.PLOT_GAP_BEFORE_AFTER:
        gap.plot_training_before_after()

    # ──────────────────────────────────────────────────────────────
    # 8. PREDICT HPT / HPC (validation + test)
    # ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 8 — PREDICT CYCLES TO SV (HPT / HPC)")
    print("=" * 60)

    results_val = None
    results_test = None
    results_df = None

    if not args.skip_validation:
        print("\n=== VALIDATION ===")
        results_val = gap.predict_all_engines(
            data.validation,
            apply_submission_window=False,
        )
        
        # Quality check for validation
        print("\n=== QUALITY CHECK (VALIDATION) ===")
        n_ok = (results_val["confidence"] == "ok").sum()
        n_fallback = (results_val["confidence"] != "ok").sum()
        print(f"Validation: {n_ok} ok, {n_fallback} fallback")
        print(
            f"  HPT: mean={results_val['Cycles_to_HPT_SV'].mean():.0f}, "
            f"std={results_val['Cycles_to_HPT_SV'].std():.0f}, "
            f"min={results_val['Cycles_to_HPT_SV'].min():.0f}, "
            f"max={results_val['Cycles_to_HPT_SV'].max():.0f}"
        )
        print(
            f"  HPC: mean={results_val['Cycles_to_HPC_SV'].mean():.0f}, "
            f"std={results_val['Cycles_to_HPC_SV'].std():.0f}, "
            f"min={results_val['Cycles_to_HPC_SV'].min():.0f}, "
            f"max={results_val['Cycles_to_HPC_SV'].max():.0f}"
        )
        
        # Combine with empty test results for compatibility
        results_df = results_val.copy()
    else:
        print("\n=== VALIDATION SKIPPED ===")

    if not args.skip_test:
        print("\n=== TEST ===")
        results_test = gap.predict_all_engines(
            data.test,
            apply_submission_window=True,
        )

        # Offset test file_idx so they don't collide with validation
        n_val = len(data.validation)
        results_test = results_test.copy()
        results_test["file_idx"] = results_test["file_idx"] + n_val
        
        if results_val is not None:
            results_df = pd.concat([results_val, results_test], ignore_index=True)
        else:
            results_df = results_test

        # Quality check for test
        print("\n=== QUALITY CHECK (TEST) ===")
        n_ok = (results_test["confidence"] == "ok").sum()
        n_fallback = (results_test["confidence"] != "ok").sum()
        print(f"Test: {n_ok} ok, {n_fallback} fallback")
        print(
            f"  HPT: mean={results_test['Cycles_to_HPT_SV'].mean():.0f}, "
            f"std={results_test['Cycles_to_HPT_SV'].std():.0f}, "
            f"min={results_test['Cycles_to_HPT_SV'].min():.0f}, "
            f"max={results_test['Cycles_to_HPT_SV'].max():.0f}"
        )
        print(
            f"  HPC: mean={results_test['Cycles_to_HPC_SV'].mean():.0f}, "
            f"std={results_test['Cycles_to_HPC_SV'].std():.0f}, "
            f"min={results_test['Cycles_to_HPC_SV'].min():.0f}, "
            f"max={results_test['Cycles_to_HPC_SV'].max():.0f}"
        )
    else:
        print("\n=== TEST SKIPPED ===")

    # Plot results only if not skipping and we have results
    if not args.skip_plotting and cfg.PLOT_GAP_RESULTS and results_df is not None:
        gap.plot_results(results_df, data.validation, data.test)

    # ──────────────────────────────────────────────────────────────
    # 9. PREDICT WATER WASH (WW)
    # ──────────────────────────────────────────────────────────────
    ww_results_test_final: dict = {}
    
    if not args.skip_ww:
        print("\n" + "=" * 60)
        print("STEP 9 — WATER WASH PREDICTION")
        print("=" * 60)
        ww = WWTrainer(hi)

        if cfg.WW_USE_TRAINING_MEAN_SLOPE:
            ww.clear_training_slopes()

        # --- Training WW ---
        print("\n--- Training WW ---")
        for esn in data.train["ESN"].unique():
            edf = data.train[data.train["ESN"] == esn]
            engine_res = hi.residuals_single(edf)
            if engine_res is None:
                print(f"  ESN {esn}: residuals None, skip")
                continue
            ww_result = ww.predict_ww(
                edf,
                engine_res,
                esn,
                use_training_mean_slope=False,
            )
            ww.results_train[esn] = ww_result
            if cfg.WW_USE_TRAINING_MEAN_SLOPE:
                ww.register_training_slope(float(ww_result["slope"]))
            if cfg.PLOT_WW and not args.skip_plotting:
                ww.plot_ww_prediction(ww_result)

        if cfg.WW_USE_TRAINING_MEAN_SLOPE:
            mean_slope = ww.finalize_training_mean_slope()
            if mean_slope is not None:
                print(f"  Training mean WW slope: {mean_slope:.6f}")
            else:
                print("  Training mean WW slope unavailable, fallback to per-engine slope.")

        # --- Validation WW ---
        if not args.skip_validation:
            print("\n--- Validation WW ---")
            for i, engine_df in enumerate(data.validation):
                for esn in engine_df["ESN"].unique():
                    edf = engine_df[engine_df["ESN"] == esn]
                    engine_res = hi.residuals_single(edf)
                    if engine_res is None:
                        print(f"  val_{i} (ESN {esn}): residuals None, skip")
                        continue
                    ww_result = ww.predict_ww(
                        edf,
                        engine_res,
                        esn,
                        use_training_mean_slope=cfg.WW_USE_TRAINING_MEAN_SLOPE,
                    )
                    ww.results_val[esn] = ww_result
                    cycles_ww = ww.cycles_to_next_ww_from_end(ww_result)
                    print(
                        f"  val_{i} (ESN {esn}): events={ww_result['n_events']}  "
                        f"slope={ww_result['slope']:.5f}  Cycles_to_WW≈{cycles_ww:.0f}"
                    )
                    if cfg.PLOT_WW and not args.skip_plotting:
                        ww.plot_ww_prediction(ww_result)

        # --- Test WW ---
        if not args.skip_test:
            print("\n--- Test WW ---")
            for i, engine_df in enumerate(data.test):
                for esn in engine_df["ESN"].unique():
                    edf = engine_df[engine_df["ESN"] == esn]
                    engine_res = hi.residuals_single(edf)
                    if engine_res is None:
                        print(f"  test_{i} (ESN {esn}): residuals None, skip")
                        continue
                    ww_result = ww.predict_ww(
                        edf,
                        engine_res,
                        esn,
                        use_training_mean_slope=cfg.WW_USE_TRAINING_MEAN_SLOPE,
                    )
                    ww.results_test[esn] = ww_result
                    ww_results_test_final[i] = ww_result  # keyed by file index
                    cycles_ww = ww.cycles_to_next_ww_from_end(ww_result)
                    print(
                        f"  test_{i} (ESN {esn}): events={ww_result['n_events']}  "
                        f"slope={ww_result['slope']:.5f}  Cycles_to_WW≈{cycles_ww:.0f}"
                    )
    else:
        print("\n" + "=" * 60)
        print("STEP 9 — WATER WASH PREDICTION (SKIPPED)")
        print("=" * 60)
        ww = WWTrainer(hi)  # Create for compatibility

    # ──────────────────────────────────────────────────────────────
    # 10. WINDOWTRAINING MODE OR ASSEMBLE SUBMISSION CSV
    # ──────────────────────────────────────────────────────────────
    
    if args.windowtraining_mode:
        print("\n" + "=" * 60)
        print("STEP 10 — WINDOWTRAINING MODE: OUTPUT RMSE/MAE")
        print("=" * 60)

        try:
            metrics, cv_folds, cv_meta = run_windowtraining_loeo_cv(
                data.train,
                include_ww=args.windowtraining_include_ww,
            )
        except Exception as exc:
            print(f"ERROR: LOEO windowtraining failed: {exc}")
            sys.exit(1)

        metric_targets = ["Cycles_to_HPT_SV", "Cycles_to_HPC_SV"]
        if args.windowtraining_include_ww:
            metric_targets.append("Cycles_to_WW")

        for target in metric_targets:
            vals = metrics.get(target, {"RMSE": float("inf"), "MAE": float("inf")})
            rmse = vals["RMSE"]
            mae = vals["MAE"]
            if np.isfinite(rmse) and np.isfinite(mae):
                print(f"  {target}: RMSE={rmse:.2f}, MAE={mae:.2f}")
            else:
                print(f"  {target}: No valid LOEO predictions available")

        print(
            "  LOEO summary: "
            f"total={cv_meta['folds_total']}  "
            f"success={cv_meta['folds_success']}  "
            f"errors={cv_meta['folds_error']}  "
            f"fallbacks={cv_meta['fallback_count']}"
        )
        if args.windowtraining_include_ww:
            print(f"  WW proxy valid folds: {cv_meta['ww_valid']}")
        if not cv_folds.empty:
            print("  Fold predictions (first rows):")
            print(cv_folds.head().to_string(index=False))
        
        # Create windowtraining directory and save metrics
        os.makedirs("windowtraining", exist_ok=True)
        
        # Find next sequential number for windowtraining
        existing = [f for f in os.listdir("windowtraining") if f.startswith("metrics-") and f.endswith(".csv")]
        nums = []
        for f in existing:
            try:
                nums.append(int(f.replace("metrics-", "").replace(".csv", "")))
            except ValueError:
                pass
        run_id = max(nums, default=0) + 1
        
        # Save metrics
        metrics_data = []
        for target, values in metrics.items():
            metrics_data.append({
                "target": target,
                "RMSE": values["RMSE"],
                "MAE": values["MAE"],
                "FOLDS_TOTAL": cv_meta["folds_total"],
                "FOLDS_SUCCESS": cv_meta["folds_success"],
                "FOLD_ERRORS": cv_meta["folds_error"],
                "FALLBACK_COUNT": cv_meta["fallback_count"],
            })
        
        metrics_df = pd.DataFrame(metrics_data)
        metrics_path = f"windowtraining/metrics-{run_id}.csv"
        metrics_df.to_csv(metrics_path, index=False)
        
        # Save config
        import shutil
        cfg_path = f"windowtraining/config-{run_id}.py"
        shutil.copy2("modules/config.py", cfg_path)
        
        elapsed = time.time() - t0
        print(f"\nWindowtraining metrics saved to: {metrics_path}")
        print(f"Config saved to: {cfg_path}")
        print(f"Total elapsed time: {elapsed:.1f}s")
        
    elif not args.skip_submission:
        print("\n" + "=" * 60)
        print("STEP 10 — ASSEMBLE SUBMISSION")
        print("=" * 60)
        
        if results_test is None:
            print("ERROR: Cannot create submission without test results. Use --skip-test=false or --skip-submission")
            sys.exit(1)

        _fallback_hpt = float(results_test["Cycles_to_HPT_SV"].mean())
        _fallback_hpc = float(results_test["Cycles_to_HPC_SV"].mean())

        rows: list[dict] = []
        for i, engine_df in enumerate(data.test):
            file_name = f"test_{i}"
            esn = engine_df["ESN"].iloc[0]

            # --- HPT and HPC (match by file index, not by ESN) ---
            n_val = len(data.validation) if not args.skip_validation else 0
            file_idx = i + n_val  # consistent with the offset applied above
            mask = results_test["file_idx"] == file_idx
            if mask.any():
                cycles_hpt = float(results_test.loc[mask, "Cycles_to_HPT_SV"].values[0])
                cycles_hpc = float(results_test.loc[mask, "Cycles_to_HPC_SV"].values[0])
            else:
                cycles_hpt = _fallback_hpt
                cycles_hpc = _fallback_hpc
                print(
                    f"  {file_name}: FALLBACK HPT/HPC (file_idx {file_idx} not found)"
                )

            # --- WW (keyed by file index) ---
            if not args.skip_ww and i in ww_results_test_final:
                cycles_ww = ww.cycles_to_next_ww_from_end(ww_results_test_final[i])
            else:
                cycles_ww = 0
                if not args.skip_ww:
                    print(f"  {file_name}: WW not available, set to 0")

            rows.append(
                {
                    "file": file_name,
                    "Cycles_to_WW": cycles_ww,
                    "Cycles_to_HPC_SV": cycles_hpc,
                    "Cycles_to_HPT_SV": cycles_hpt,
                }
            )

            print(
                f"  {file_name} (ESN {esn}):  "
                f"WW={cycles_ww:.0f}  HPC={cycles_hpc:.0f}  HPT={cycles_hpt:.0f}"
            )

        submission_df = pd.DataFrame(
            rows,
            columns=[
                "file",
                "Cycles_to_WW",
                "Cycles_to_HPC_SV",
                "Cycles_to_HPT_SV",
            ],
        )

        # 1. Crea la cartella se non esiste
        os.makedirs("results", exist_ok=True)

        # 2. Find next sequential number
        existing = [f for f in os.listdir("results") if f.startswith("submission-") and f.endswith(".csv")]
        nums = []
        for f in existing:
            try:
                nums.append(int(f.replace("submission-", "").replace(".csv", "")))
            except ValueError:
                pass
        run_id = max(nums, default=0) + 1

        sub_path = f"results/submission-{run_id}.csv"
        cfg_path = f"results/config-{run_id}.csv"

        # 4. Salva la submission
        submission_df.to_csv(sub_path, index=False)

        # 5. Salva una copia del file config.py con i valori effettivamente usati
        import shutil
        cfg_path = cfg_path.replace(".csv", ".py")
        shutil.copy2("modules/config.py", cfg_path)

        elapsed = time.time() - t0

        # 6. Output finale
        print(f"\nSubmission saved to: {sub_path}")
        print(f"Config saved to: {cfg_path}")
        print(submission_df.to_string(index=False))
        print(f"\nTotal elapsed time: {elapsed:.1f}s")
    else:
        print("\n" + "=" * 60)
        print("STEP 10 — SUBMISSION GENERATION (SKIPPED)")
        print("=" * 60)
        elapsed = time.time() - t0
        print(f"Total elapsed time: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
