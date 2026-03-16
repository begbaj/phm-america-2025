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
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn")


# ══════════════════════════════════════════════════════════════════
#  STEP 1-2: DATA LOADING & PREPARATION
# ══════════════════════════════════════════════════════════════════


def load_data() -> Data:
    """Load all datasets and prepare the training subset.

    Returns
    -------
    Data
        Populated Data instance with train/val/test loaded and
        training subset prepared.
    """
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

    return data


# ══════════════════════════════════════════════════════════════════
#  STEPS 3-5: HITrainer (linear models + residuals + HI coefs)
# ══════════════════════════════════════════════════════════════════


def train_hi(data: Data) -> HITrainer:
    """Train or load the HITrainer (linear models, residuals, HI
    coefficients).

    Parameters
    ----------
    data : Data
        Populated Data instance.

    Returns
    -------
    HITrainer
        Ready-to-use HITrainer with models and coefficients.
    """
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
            hi.plot_residuals(hi._res_train_healthy, title_suffix="Training (healthy)")
            hi.plot_residuals(hi._res_train, title_suffix="Training (full)")
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

    return hi


# ══════════════════════════════════════════════════════════════════
#  STEPS 6-8: HPT/HPC SV PREDICTION (classifier + gap + predict)
# ══════════════════════════════════════════════════════════════════


def predict_hpc_hpt(
    hi: HITrainer, data: Data
) -> tuple[pd.DataFrame, pd.DataFrame, int]:
    """Run the full HPT/HPC SV prediction pipeline.

    Steps 6-8: LGBM classifier, gap correction, and inference on
    validation + test sets.

    Parameters
    ----------
    hi : HITrainer
        Trained HITrainer.
    data : Data
        Populated Data instance.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame, int]
        (results_val, results_test, n_val)
    """
    # ── 6. LGBM CYCLE CLASSIFIER ─────────────────────────────────
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

    # ── 7. LGBM GAP CORRECTION ───────────────────────────────────
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

    if cfg.PLOT_GAP_BEFORE_AFTER:
        gap.plot_training_before_after()

    # ── 8. PREDICT HPT / HPC (validation + test) ─────────────────
    print("\n" + "=" * 60)
    print("STEP 8 — PREDICT CYCLES TO SV (HPT / HPC)")
    print("=" * 60)

    print("\n=== VALIDATION ===")
    results_val = gap.predict_all_engines(data.validation)

    print("\n=== TEST ===")
    results_test = gap.predict_all_engines(data.test)

    # Offset test file_idx so they don't collide with validation
    n_val = len(data.validation)
    results_test = results_test.copy()
    results_test["file_idx"] = results_test["file_idx"] + n_val
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

    if cfg.PLOT_GAP_RESULTS:
        gap.plot_results(results_df, data.validation, data.test)

    return results_val, results_test, n_val


# ══════════════════════════════════════════════════════════════════
#  STEP 9: WATER WASH PREDICTION
# ══════════════════════════════════════════════════════════════════


def predict_ww(
    hi: HITrainer, data: Data
) -> tuple[WWTrainer, dict]:
    """Run the full WW prediction pipeline.

    Trains the global WW slope/gap on training engines, then predicts
    WW events for validation and test sets.

    Parameters
    ----------
    hi : HITrainer
        Trained HITrainer.
    data : Data
        Populated Data instance.

    Returns
    -------
    tuple[WWTrainer, dict]
        (ww_trainer, ww_results_test_final) where
        ww_results_test_final is keyed by test file index.
    """
    print("\n" + "=" * 60)
    print("STEP 9 — WATER WASH PREDICTION")
    print("=" * 60)

    ww = WWTrainer(hi)

    # ── Train WW global slope + gap ──────────────────────────────
    print("\n--- Training WW global slope ---")
    engine_training = []
    for esn in data.train["ESN"].unique():
        edf = data.train[data.train["ESN"] == esn].copy()
        if edf.empty:
            continue
        engine_training.append(edf)
    ww.train_ww(engine_training)

    # ── Training WW (predict + plot) ─────────────────────────────
    print("\n--- Training WW ---")
    for edf in engine_training:
        esn = edf["ESN"].iloc[0]
        engine_res = hi.residuals_single(edf)
        if engine_res is None:
            continue
        ww_result = ww.predict_ww(edf, engine_res, esn)
        ww.results_train[esn] = ww_result
        if cfg.PLOT_WW:
            ww.plot_ww_prediction(ww_result)

    # ── Validation WW ────────────────────────────────────────────
    print("\n--- Validation WW ---")
    for i, engine_df in enumerate(data.validation):
        esn = engine_df["ESN"].iloc[0]
        engine_res = hi.residuals_single(engine_df)
        if engine_res is None:
            continue

        ww_result = ww.predict_ww(engine_df, engine_res, esn)
        ww.results_val[esn] = ww_result
        if cfg.PLOT_WW:
            ww.plot_ww_prediction(ww_result)
        cycles_ww = ww.cycles_to_next_ww_from_end(ww_result)

        print(
            f"val_{i} ESN {esn}: "
            f"events={ww_result['n_events']} "
            f"slope={ww_result['slope']:.5f} "
            f"WW≈{cycles_ww:.0f}"
        )

    # ── Test WW ──────────────────────────────────────────────────
    print("\n--- Test WW ---")
    ww_results_test_final = {}

    for i, engine_df in enumerate(data.test):
        esn = engine_df["ESN"].iloc[0]
        engine_res = hi.residuals_single(engine_df)
        if engine_res is None:
            continue

        ww_result = ww.predict_ww(engine_df, engine_res, esn)
        ww.results_test[esn] = ww_result
        ww_results_test_final[i] = ww_result
        if cfg.PLOT_WW:
            ww.plot_ww_prediction(ww_result)
        cycles_ww = ww.cycles_to_next_ww_from_end(ww_result)

        print(
            f"test_{i} ESN {esn}: "
            f"events={ww_result['n_events']} "
            f"slope={ww_result['slope']:.5f} "
            f"WW≈{cycles_ww:.0f}"
        )

    return ww, ww_results_test_final


# ══════════════════════════════════════════════════════════════════
#  STEP 10: ASSEMBLE SUBMISSION
# ══════════════════════════════════════════════════════════════════


def assemble_submission(
    data: Data,
    results_test: pd.DataFrame,
    n_val: int,
    ww: WWTrainer,
    ww_results_test_final: dict,
) -> pd.DataFrame:
    """Assemble and save the final submission CSV.

    Parameters
    ----------
    data : Data
        Populated Data instance.
    results_test : pd.DataFrame
        HPT/HPC prediction results for test set.
    n_val : int
        Number of validation files (used for file_idx offset).
    ww : WWTrainer
        Trained WWTrainer instance.
    ww_results_test_final : dict
        WW results keyed by test file index.

    Returns
    -------
    pd.DataFrame
        The submission DataFrame.
    """
    print("\n" + "=" * 60)
    print("STEP 10 — ASSEMBLE SUBMISSION")
    print("=" * 60)

    _fallback_hpt = float(results_test["Cycles_to_HPT_SV"].mean())
    _fallback_hpc = float(results_test["Cycles_to_HPC_SV"].mean())

    rows: list[dict] = []
    for i, engine_df in enumerate(data.test):
        file_name = f"test_{i}"
        esn = engine_df["ESN"].iloc[0]

        # --- HPT and HPC (match by file index, not by ESN) ---
        file_idx = i + n_val  # consistent with the offset applied above
        mask = results_test["file_idx"] == file_idx
        if mask.any():
            cycles_hpt = float(results_test.loc[mask, "Cycles_to_HPT_SV"].values[0])
            cycles_hpc = float(results_test.loc[mask, "Cycles_to_HPC_SV"].values[0])
        else:
            cycles_hpt = _fallback_hpt
            cycles_hpc = _fallback_hpc
            print(f"  {file_name}: FALLBACK HPT/HPC (file_idx {file_idx} not found)")

        # --- WW (keyed by file index) ---
        if i in ww_results_test_final:
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
    print(f"\nSubmission saved to: {cfg.SUBMISSION_OUTPUT}")
    print(submission_df.to_string(index=False))

    return submission_df


# ══════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════


def main() -> None:
    t0 = time.time()

    data = load_data()
    hi = train_hi(data)

    # HPT/HPC SV prediction
    results_val, results_test, n_val = predict_hpc_hpt(hi, data)

    # Water Wash prediction
    ww, ww_results_test_final = predict_ww(hi, data)

    # Assemble submission
    assemble_submission(data, results_test, n_val, ww, ww_results_test_final)

    elapsed = time.time() - t0
    print(f"\nTotal elapsed time: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
