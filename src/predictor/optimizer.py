#!/usr/bin/env python3
"""
optimizer.py - Ternary search per le smoothing window ottimali.

Ottimizza SMOOTHING_WINDOW_HPT, SMOOTHING_WINDOW_HPC e SMOOTHING_WINDOW_WW
uno alla volta (coordinate descent) usando ternary search per minimizzare
RMSE/MAE (windowtraining mode) o |mediana| dello scarto rispetto alla submission di riferimento.

Uso
---
    python optimizer.py [--windowtraining]

Configurazione in fondo al file (SEARCH_RANGES, MAX_ITER, ecc.)
"""

from __future__ import annotations

import subprocess
import sys
import os
import time
import pandas as pd
import numpy as np
import argparse

# ── Riferimento per submission mode ──────────────────────────────────────────────
BASE_SUBMISSION = "results/submission_compare.csv"

# ── Range di ricerca per ogni parametro ──────────────────────────────────────────
SEARCH_RANGES = {
    "hpt": (3, 100),
    "hpc": (3, 100),
    "ww":  (3, 100),
}


def get_search_range(param: str) -> tuple[int, int]:
    """Return search range for a parameter, including shared HPT/HPC mode."""
    if param == "hpt_hpc":
        lo = max(SEARCH_RANGES["hpt"][0], SEARCH_RANGES["hpc"][0])
        hi = min(SEARCH_RANGES["hpt"][1], SEARCH_RANGES["hpc"][1])
        if lo > hi:
            raise ValueError("Invalid search range for shared HPT/HPC optimization")
        return lo, hi
    return SEARCH_RANGES[param]

# ── Ternary search ──────────────────────────────────────────────────────────────
MAX_ITER = 12          # iterazioni per parametro (precisione ≈ range / 3^MAX_ITER)
TOLERANCE = 10          # se hi - lo < TOLERANCE, ferma la ricerca
COORD_ROUNDS = 1       # round di coordinate descent (ripetere il giro)

# ── Defaults iniziali (punto di partenza) ────────────────────────────────────────
INITIAL = {"hpt": 100, "hpc": 100, "ww": 100}


# ═════════════════════════════════════════════════════════════════
#  RUNNER
# ═════════════════════════════════════════════════════════════════

def run_main(
    hpt: int,
    hpc: int,
    ww: int,
    windowtraining_mode: bool = False,
    include_ww_objective: bool = False,
) -> str | None:
    """Lancia main.py con le window date. Ritorna il path del file di output."""
    cmd = [
        sys.executable, "main.py",
        "--smoothing-hpt", str(hpt),
        "--smoothing-hpc", str(hpc),
        "--smoothing-ww",  str(ww),
        "--skip-plotting",  # Skip plotting per velocità
    ]
    
    if windowtraining_mode:
        cmd.append("--windowtraining-mode")
        if include_ww_objective:
            cmd.append("--windowtraining-include-ww")
    
    print(f"\n{'─'*60}")
    run_suffix = "(windowtraining)" if windowtraining_mode else ""
    if windowtraining_mode and include_ww_objective:
        run_suffix = "(windowtraining+ww)"
    print(f"  RUN: HPT={hpt}  HPC={hpc}  WW={ww} {run_suffix}")
    print(f"{'─'*60}")
    t0 = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True)
    elapsed = time.time() - t0

    if result.returncode != 0:
        print(f"  ERRORE (exit {result.returncode}):")
        print(result.stderr[-500:] if result.stderr else "(no stderr)")
        return None

    # Trova il path del file dall'output
    if windowtraining_mode:
        for line in result.stdout.splitlines():
            if "Windowtraining metrics saved to:" in line:
                path = line.split("Windowtraining metrics saved to:")[-1].strip()
                print(f"  OK in {elapsed:.0f}s → {path}")
                return path
    else:
        for line in result.stdout.splitlines():
            if "Submission saved to:" in line:
                path = line.split("Submission saved to:")[-1].strip()
                print(f"  OK in {elapsed:.0f}s → {path}")
                return path

    print(f"  WARNING: non trovo il path del file nell'output")
    return None


# ═════════════════════════════════════════════════════════════════
#  METRICA
# ═════════════════════════════════════════════════════════════════

def compute_windowtraining_metric(metrics_path: str, target: str) -> float:
    """Calcola RMSE per la colonna target da file windowtraining.
    
    target: "hpt", "hpc", "ww" (simplified names)
    Returns RMSE or inf if error
    """
    target_map = {
        "hpt": "Cycles_to_HPT_SV",
        "hpc": "Cycles_to_HPC_SV",
        "ww": "Cycles_to_WW"
    }
    
    if target not in target_map:
        return float("inf")
    
    try:
        metrics_df = pd.read_csv(metrics_path)
        target_col = target_map[target]
        target_rows = metrics_df[metrics_df["target"] == target_col]
        
        if len(target_rows) == 0:
            return float("inf")
        
        rmse = target_rows["RMSE"].values[0]
        return float(rmse)
    except Exception as e:
        print(f"  ERROR reading metrics from {metrics_path}: {e}")
        return float("inf")

def compute_submission_metric(submission_path: str, target: str) -> float:
    """Calcola |mediana| dello scarto per la colonna target.

    target: "Cycles_to_HPT_SV", "Cycles_to_HPC_SV", "Cycles_to_WW"
    """
    try:
        base = pd.read_csv(BASE_SUBMISSION)
        sub = pd.read_csv(submission_path)
        diff = sub[target] - base[target]
        return float(abs(diff.median()))
    except Exception as e:
        print(f"  ERROR computing submission metric: {e}")
        return float("inf")

def compute_total_windowtraining_metric(
    metrics_path: str,
    include_ww: bool = False,
) -> float:
    """Metrica aggregata: somma RMSE per HPT/HPC (+WW opzionale)."""
    total = 0.0
    params = ["hpt", "hpc"]
    if include_ww:
        params.append("ww")
    for param in params:
        total += compute_windowtraining_metric(metrics_path, param)
    return total

def compute_total_submission_metric(submission_path: str) -> float:
    """Metrica aggregata: somma |mediana| per tutte e tre le colonne."""
    try:
        base = pd.read_csv(BASE_SUBMISSION)
        sub = pd.read_csv(submission_path)
        total = 0.0
        for col in ["Cycles_to_HPT_SV", "Cycles_to_HPC_SV", "Cycles_to_WW"]:
            diff = sub[col] - base[col]
            total += abs(float(diff.median()))
        return total
    except Exception as e:
        print(f"  ERROR computing total submission metric: {e}")
        return float("inf")


# ═════════════════════════════════════════════════════════════════
#  TERNARY SEARCH (per singolo parametro)
# ═════════════════════════════════════════════════════════════════

PARAM_TO_COL_SUBMISSION = {
    "hpt": "Cycles_to_HPT_SV",
    "hpc": "Cycles_to_HPC_SV",
    "ww":  "Cycles_to_WW",
}


def ternary_search_param(
    param: str,
    current: dict[str, int],
    windowtraining_mode: bool = False,
    include_ww_objective: bool = False,
) -> int:
    """Ternary search su un singolo parametro, tenendo fissi gli altri.

    Ritorna il valore ottimale trovato.
    """
    lo, hi = get_search_range(param)
    print(f"\n{'═'*60}")
    print(f"  TERNARY SEARCH: {param.upper()}  range=[{lo}, {hi}] {'(windowtraining)' if windowtraining_mode else '(submission)'}")
    print(f"{'═'*60}")

    # Cache per evitare run duplicati
    cache: dict[int, float] = {}

    def evaluate(val: int) -> float:
        if val in cache:
            return cache[val]
        params = current.copy()
        if param == "hpt_hpc":
            params["hpt"] = val
            params["hpc"] = val
        else:
            params[param] = val
        path = run_main(
            params["hpt"],
            params["hpc"],
            params["ww"],
            windowtraining_mode,
            include_ww_objective,
        )
        if path is None:
            cache[val] = float("inf")
            return float("inf")
            
        if windowtraining_mode:
            if param == "hpt_hpc":
                metric_hpt = compute_windowtraining_metric(path, "hpt")
                metric_hpc = compute_windowtraining_metric(path, "hpc")
                metric = metric_hpt + metric_hpc
                print(
                    f"    HPT=HPC={val}  →  RMSE_HPT={metric_hpt:.2f}  "
                    f"RMSE_HPC={metric_hpc:.2f}  SUM={metric:.2f}"
                )
            else:
                metric = compute_windowtraining_metric(path, param)
                print(f"    {param.upper()}={val}  →  RMSE={metric:.2f}")
        else:
            if param == "hpt_hpc":
                metric_hpt = compute_submission_metric(path, PARAM_TO_COL_SUBMISSION["hpt"])
                metric_hpc = compute_submission_metric(path, PARAM_TO_COL_SUBMISSION["hpc"])
                metric = metric_hpt + metric_hpc
                print(
                    f"    HPT=HPC={val}  →  |mediana|_HPT={metric_hpt:.2f}  "
                    f"|mediana|_HPC={metric_hpc:.2f}  SUM={metric:.2f}"
                )
            else:
                col = PARAM_TO_COL_SUBMISSION[param]
                metric = compute_submission_metric(path, col)
                print(f"    {param.upper()}={val}  →  |mediana|={metric:.2f}")
            
        cache[val] = metric
        return metric

    for iteration in range(MAX_ITER):
        if hi - lo < TOLERANCE:
            break

        m1 = lo + (hi - lo) // 3
        m2 = hi - (hi - lo) // 3

        # Evita che m1 == m2
        if m1 == m2:
            m2 = min(m1 + 1, hi)
        if m1 == m2:
            break

        print(f"\n  iter {iteration+1}/{MAX_ITER}: lo={lo} m1={m1} m2={m2} hi={hi}")

        f1 = evaluate(m1)
        f2 = evaluate(m2)

        if f1 < f2:
            hi = m2
        else:
            lo = m1

    # Valore ottimale = punto medio del range finale
    best_val = (lo + hi) // 2

    # Valuta anche il best_val finale (potrebbe non essere nella cache)
    evaluate(best_val)

    # Trova il reale minimo tra tutti i valori testati
    best_val = min(cache, key=lambda key: cache[key])
    best_metric = cache[best_val]
    metric_name = "RMSE" if windowtraining_mode else "|mediana|"
    print(f"\n  ✓ {param.upper()} ottimale = {best_val}  ({metric_name} = {best_metric:.2f})")
    return best_val


# ═════════════════════════════════════════════════════════════════
#  MAIN
# ═════════════════════════════════════════════════════════════════

def parse_args():
    parser = argparse.ArgumentParser(description="Optimizer for smoothing window parameters")
    parser.add_argument("--windowtraining", action="store_true",
                        help="Use windowtraining mode (RMSE/MAE optimization) instead of submission mode")
    parser.add_argument("--same-hpt-hpc", action="store_true",
                        help="Optimize one shared window value for both HPT and HPC (HPT=HPC)")
    parser.add_argument("--include-ww-objective", action="store_true",
                        help="In windowtraining mode, include WW proxy RMSE and optimize smoothing WW")
    return parser.parse_args()

def main() -> None:
    args = parse_args()
    windowtraining_mode = args.windowtraining
    same_hpt_hpc = args.same_hpt_hpc
    include_ww_objective = bool(args.include_ww_objective and windowtraining_mode)
    
    if not windowtraining_mode and not os.path.exists(BASE_SUBMISSION):
        print(f"ERRORE: file di riferimento {BASE_SUBMISSION} non trovato.")
        print("Usa --windowtraining per ottimizzare RMSE/MAE invece delle mediane.")
        sys.exit(1)

    current = INITIAL.copy()
    if same_hpt_hpc:
        shared_start = int(round((current["hpt"] + current["hpc"]) / 2))
        current["hpt"] = shared_start
        current["hpc"] = shared_start
        if windowtraining_mode:
            order = ["hpt_hpc"] + (["ww"] if include_ww_objective else [])
        else:
            order = ["hpt_hpc", "ww"]
    else:
        if windowtraining_mode:
            order = ["hpt", "hpc"] + (["ww"] if include_ww_objective else [])
        else:
            order = ["hpt", "hpc", "ww"]

    mode_str = "Windowtraining (RMSE/MAE)" if windowtraining_mode else "Submission (mediane)"
    if windowtraining_mode and include_ww_objective:
        mode_str = "Windowtraining (RMSE/MAE + WW proxy)"
    print("=" * 60)
    print(f"  OPTIMIZER — Ternary Search (coordinate descent)")
    print(f"  Modalità: {mode_str}")
    print(f"  Vincolo HPT=HPC: {same_hpt_hpc}")
    if windowtraining_mode:
        print(f"  Include WW objective: {include_ww_objective}")
    if not windowtraining_mode:
        print(f"  Riferimento: {BASE_SUBMISSION}")
    print(f"  Partenza: HPT={current['hpt']}  HPC={current['hpc']}  WW={current['ww']}")
    print(f"  Rounds: {COORD_ROUNDS}  ×  {len(order)} parametri  ×  ≤{MAX_ITER} iter")
    print("=" * 60)

    t0 = time.time()

    for rnd in range(COORD_ROUNDS):
        print(f"\n\n{'▓'*60}")
        print(f"  ROUND {rnd+1}/{COORD_ROUNDS}")
        print(f"{'▓'*60}")

        for param in order:
            best = ternary_search_param(
                param,
                current,
                windowtraining_mode,
                include_ww_objective,
            )
            if param == "hpt_hpc":
                current["hpt"] = best
                current["hpc"] = best
            else:
                current[param] = best

        print(f"\n  Fine round {rnd+1}: HPT={current['hpt']}  HPC={current['hpc']}  WW={current['ww']}")

    # Run finale con i parametri ottimali
    print(f"\n\n{'█'*60}")
    print(f"  RISULTATO FINALE")
    print(f"  HPT={current['hpt']}  HPC={current['hpc']}  WW={current['ww']}")
    print(f"{'█'*60}")

    final_path = run_main(
        current["hpt"],
        current["hpc"],
        current["ww"],
        windowtraining_mode,
        include_ww_objective,
    )
    if final_path:
        if windowtraining_mode:
            total = compute_total_windowtraining_metric(
                final_path,
                include_ww=include_ww_objective,
            )
            if include_ww_objective:
                print(f"\n  Metrica totale (somma RMSE HPT+HPC+WW): {total:.2f}")
            else:
                print(f"\n  Metrica totale (somma RMSE HPT+HPC): {total:.2f}")
        else:
            total = compute_total_submission_metric(final_path)
            print(f"\n  Metrica totale (somma |mediane|): {total:.2f}")

    elapsed = time.time() - t0
    print(f"\n  Tempo totale ottimizzazione: {elapsed:.0f}s ({elapsed/60:.1f}min)")


if __name__ == "__main__":
    main()
