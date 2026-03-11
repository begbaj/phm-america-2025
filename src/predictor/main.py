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

warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn")


def main() -> None:
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
    # 3. TRAIN LINEAR MODELS (nominal-behaviour regressors)
    # ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 3 — TRAIN LINEAR MODELS")
    print("=" * 60)
    hi = HITrainer(data)
    hi.train_linear_models()
    print(f"  Models: {list(hi.models.keys())}")

    # ──────────────────────────────────────────────────────────────
    # 4. COMPUTE RESIDUALS (all datasets)
    # ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 4 — COMPUTE RESIDUALS")
    print("=" * 60)
    hi.compute_all_residuals()

    # Plot training residuals
    hi.plot_residuals(hi._res_train_healthy, title_suffix="Training (healthy)")
    hi.plot_residuals(hi._res_train, title_suffix="Training (full)")
    hi.plot_residuals(hi._res_test_loo, title_suffix="Leave-One-Out ESN")

    # ──────────────────────────────────────────────────────────────
    # 5. OPTIMISE HI COEFFICIENTS
    # ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 5 — HI COEFFICIENT OPTIMISATION")
    print("=" * 60)
    hi.train_coefficients()
    print(f"  chpt = {hi.chpt}")
    print(f"  chpc = {hi.chpc}")

    # Plot HI on training data
    hi.plot_training_hi()

    # ──────────────────────────────────────────────────────────────
    # 6. TRAIN LGBM CYCLE CLASSIFIER
    # ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 6 — LGBM CYCLE CLASSIFIER")
    print("=" * 60)
    classifier = LGBMCycleClassifier(hi)
    classifier.train()
    classifier.save()

    # ──────────────────────────────────────────────────────────────
    # 7. TRAIN LGBM GAP CORRECTION
    # ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 7 — LGBM GAP CORRECTION")
    print("=" * 60)
    gap = LGBMGapCorrection(hi, classifier)
    gap.train()
    gap.save()

    # Plot training before/after gap correction
    gap.plot_training_before_after()

    # ──────────────────────────────────────────────────────────────
    # 8. PREDICT HPT / HPC (validation + test)
    # ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 8 — PREDICT CYCLES TO SV (HPT / HPC)")
    print("=" * 60)

    print("\n=== VALIDATION ===")
    results_val = gap.predict_all_engines(data.validation)

    print("\n=== TEST ===")
    results_test = gap.predict_all_engines(data.test)

    results_df = pd.concat([results_val, results_test], ignore_index=True)

    # Quality check
    print("\n=== QUALITY CHECK ===")
    for label, rdf in [
        ("Validation", results_val),
        ("Test", results_test),
    ]:
        n_ok = (rdf["confidence"] == "ok").sum()
        n_fallback = (rdf["confidence"] != "ok").sum()
        print(f"{label}: {n_ok} ok, {n_fallback} fallback")
        print(
            f"  HPT: mean={rdf['Cycles_to_HPT_SV'].mean():.0f}, "
            f"std={rdf['Cycles_to_HPT_SV'].std():.0f}, "
            f"min={rdf['Cycles_to_HPT_SV'].min():.0f}, "
            f"max={rdf['Cycles_to_HPT_SV'].max():.0f}"
        )
        print(
            f"  HPC: mean={rdf['Cycles_to_HPC_SV'].mean():.0f}, "
            f"std={rdf['Cycles_to_HPC_SV'].std():.0f}, "
            f"min={rdf['Cycles_to_HPC_SV'].min():.0f}, "
            f"max={rdf['Cycles_to_HPC_SV'].max():.0f}"
        )

    # Plot results
    gap.plot_results(results_df, data.validation, data.test)

    # ──────────────────────────────────────────────────────────────
    # 9. PREDICT WATER WASH (WW)
    # ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 9 — WATER WASH PREDICTION")
    print("=" * 60)
    ww = WWTrainer(hi)

    # --- Training WW ---
    print("\n--- Training WW ---")
    for esn in data.train["ESN"].unique():
        edf = data.train[data.train["ESN"] == esn]
        engine_res = hi.residuals_single(edf)
        if engine_res is None:
            print(f"  ESN {esn}: residuals None, skip")
            continue
        ww_result = ww.predict_ww(edf, engine_res, esn)
        ww.results_train[esn] = ww_result
        ww.plot_ww_prediction(ww_result)

    # --- Validation WW ---
    print("\n--- Validation WW ---")
    for i, engine_df in enumerate(data.validation):
        for esn in engine_df["ESN"].unique():
            edf = engine_df[engine_df["ESN"] == esn]
            engine_res = hi.residuals_single(edf)
            if engine_res is None:
                print(f"  val_{i} (ESN {esn}): residuals None, skip")
                continue
            ww_result = ww.predict_ww(edf, engine_res, esn)
            ww.results_val[esn] = ww_result
            cycles_ww = ww.cycles_to_next_ww_from_end(ww_result)
            print(
                f"  val_{i} (ESN {esn}): events={ww_result['n_events']}  "
                f"slope={ww_result['slope']:.5f}  Cycles_to_WW≈{cycles_ww:.0f}"
            )
            ww.plot_ww_prediction(ww_result)

    # --- Test WW ---
    print("\n--- Test WW ---")
    ww_results_test_final: dict = {}
    for i, engine_df in enumerate(data.test):
        for esn in engine_df["ESN"].unique():
            edf = engine_df[engine_df["ESN"] == esn]
            engine_res = hi.residuals_single(edf)
            if engine_res is None:
                print(f"  test_{i} (ESN {esn}): residuals None, skip")
                continue
            ww_result = ww.predict_ww(edf, engine_res, esn)
            ww.results_test[esn] = ww_result
            ww_results_test_final[esn] = ww_result
            ww_results_test_final[i] = ww_result
            cycles_ww = ww.cycles_to_next_ww_from_end(ww_result)
            print(
                f"  test_{i} (ESN {esn}): events={ww_result['n_events']}  "
                f"slope={ww_result['slope']:.5f}  Cycles_to_WW≈{cycles_ww:.0f}"
            )

    # ──────────────────────────────────────────────────────────────
    # 10. ASSEMBLE SUBMISSION CSV
    # ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 10 — ASSEMBLE SUBMISSION")
    print("=" * 60)

    _fallback_hpt = float(results_test["Cycles_to_HPT_SV"].mean())
    _fallback_hpc = float(results_test["Cycles_to_HPC_SV"].mean())

    rows: list[dict] = []
    for i, engine_df in enumerate(data.test):
        file_name = f"test_{i}"
        esn = engine_df["ESN"].iloc[0]

        # --- HPT and HPC ---
        mask = results_test["ESN"] == esn
        if mask.any():
            cycles_hpt = float(results_test.loc[mask, "Cycles_to_HPT_SV"].values[0])
            cycles_hpc = float(results_test.loc[mask, "Cycles_to_HPC_SV"].values[0])
        else:
            cycles_hpt = _fallback_hpt
            cycles_hpc = _fallback_hpc
            print(
                f"  {file_name}: FALLBACK HPT/HPC (ESN {esn} not found in results_test)"
            )

        # --- WW ---
        if esn in ww_results_test_final:
            cycles_ww = ww.cycles_to_next_ww_from_end(ww_results_test_final[esn])
        elif i in ww_results_test_final:
            cycles_ww = ww.cycles_to_next_ww_from_end(ww_results_test_final[i])
        else:
            cycles_ww = 0
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

    submission_df.to_csv(cfg.SUBMISSION_OUTPUT, index=False)
    elapsed = time.time() - t0
    print(f"\nSubmission saved to: {cfg.SUBMISSION_OUTPUT}")
    print(submission_df.to_string(index=False))
    print(f"\nTotal elapsed time: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
