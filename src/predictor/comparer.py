import pandas as pd
import os
import re
import argparse
import math
from numpy import sort

# --- Codici Colore ANSI ---
RESET = "\033[0m"
WHITE = "\033[97m"
BLUE = "\033[94m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
BEST_OVERALL_BG = "\033[48;2;36;84;36m"
BEST_COLUMN_BG = "\033[48;2;28;64;120m"
BOLD = "\033[1m"

def gradient_color_error(med_val, all_medians):
    """Colore gradiente basato su range osservato (min → max).
    
    Blu (min) → Verde/Giallo → Rosso (max).
    """
    numeric_medians = []
    for value in all_medians:
        if value == "N/A":
            continue
        try:
            numeric_value = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(numeric_value):
            numeric_medians.append(numeric_value)

    if not numeric_medians:
        return RESET

    try:
        current_value = float(med_val)
    except (TypeError, ValueError):
        return RESET

    if not math.isfinite(current_value):
        return RESET

    min_med = min(numeric_medians)
    max_med = max(numeric_medians)
    
    if max_med == min_med:
        return RESET
    
    # Normalizza a [0, 1]: min = blu, max = rosso
    t = (current_value - min_med) / (max_med - min_med)
    t = max(0.0, min(1.0, t))
    
    # Gradiente: Blu (0) → Verde (0.33) → Giallo (0.66) → Rosso (1.0)
    if t <= 0.33:
        # Blu → Verde
        s = t / 0.33
        r = int(0)
        g = int(50 + 205 * s)
        b = int(255 - 255 * s)
    elif t <= 0.66:
        # Verde → Giallo
        s = (t - 0.33) / 0.33
        r = int(255 * s)
        g = int(255)
        b = int(0)
    else:
        # Giallo → Rosso
        s = (t - 0.66) / 0.34
        r = int(255)
        g = int(255 - 255 * s)
        b = int(0)
    
    return f"\033[38;2;{r};{g};{b}m"


def fmt_error(med, std, all_medians):
    """Formatta l'errore con gradiente dinamico."""
    color = gradient_color_error(med, all_medians)
    text = f"{med:.2f} ± {std:.2f}"
    return f"{color}{text:<20}{RESET}", color  # ritorna sia la stringa che il colore

def fmt_metric(value, all_values):
    """Formatta una metrica (RMSE/MAE) con gradiente dinamico."""
    color = gradient_color_error(value, all_values)
    text = f"{value:.2f}"
    return f"{color}{text:<10}{RESET}", color


def fmt_file_with_best_bg(filename, is_best):
    """Formatta il nome file con background per il best match."""
    if is_best:
        return f"{BEST_OVERALL_BG}{WHITE}{filename:<25}{RESET}"
    return f"{filename:<25}"


def strip_ansi(text):
    """Rimuove codici ANSI da una stringa."""
    return re.sub(r"\x1B\[[0-?]*[ -/]*[@-~]", "", text)


def highlight_cell(text, is_best, bg_color=BEST_COLUMN_BG):
    """Applica un background a una cella già formattata."""
    if not is_best:
        return text
    clean_text = strip_ansi(text)
    return f"{bg_color}{WHITE}{BOLD}{clean_text}{RESET}"


def min_finite(values):
    """Ritorna il minimo valore finito o None."""
    finite_vals = []
    for value in values:
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(numeric):
            finite_vals.append(numeric)
    return min(finite_vals) if finite_vals else None


def is_best_value(value, best_value, tol=1e-9):
    """Confronto robusto per evidenziare il best value."""
    if best_value is None:
        return False
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return False
    if not math.isfinite(numeric):
        return False
    return math.isclose(numeric, float(best_value), rel_tol=tol, abs_tol=tol)

# --- NUOVA Funzione per leggere il config tramite Regex ---
def read_config_vars(filepath):
    """Legge un file .py come testo ed estrae le variabili target tramite Regex."""
    config_dict = {}
    if os.path.exists(filepath):
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Cerca le variabili di nostro interesse (nuovi nomi separati)
            for var in ["SMOOTHING_WINDOW_WW", "SMOOTHING_WINDOW_HPT", "SMOOTHING_WINDOW_HPC"]:
                # Gestisce sia "VAR = 50" che "VAR: int = 50"
                match = re.search(rf"{var}\s*(?::\s*\w+\s*)?=\s*([^#\n]+)", content)
                if match:
                    val_str = match.group(1).strip()
                    try:
                        config_dict[var] = float(val_str)
                    except ValueError:
                        config_dict[var] = val_str

            # Fallback: vecchio config con singolo SMOOTHING_WINDOW
            if not any(v in config_dict for v in ["SMOOTHING_WINDOW_WW", "SMOOTHING_WINDOW_HPT", "SMOOTHING_WINDOW_HPC"]):
                match = re.search(r"SMOOTHING_WINDOW\s*(?::\s*\w+\s*)?=\s*([^#\n]+)", content)
                if match:
                    val_str = match.group(1).strip()
                    try:
                        val = float(val_str)
                    except ValueError:
                        val = val_str
                    config_dict["SMOOTHING_WINDOW_HPT"] = val
                    config_dict["SMOOTHING_WINDOW_HPC"] = val
                    config_dict["SMOOTHING_WINDOW_WW"] = val
        except Exception as e:
            print(f"Avviso: Impossibile leggere il config {filepath}: {e}")
    return config_dict

# --- Impostazioni di Base ---
BASE_SUBMISSION = "results/submission_compare.csv"
BASE_CONFIG = "results/config-2.py" # Cambia se il tuo config base si chiama diversamente

submission_compare = pd.read_csv(BASE_SUBMISSION)
base_config_vars = read_config_vars(BASE_CONFIG)

TARGET_VARS = ["SMOOTHING_WINDOW_HPT", "SMOOTHING_WINDOW_HPC", "SMOOTHING_WINDOW_WW"]
SHORT_NAMES = {"SMOOTHING_WINDOW_HPT": "S_HPT", "SMOOTHING_WINDOW_HPC": "S_HPC", "SMOOTHING_WINDOW_WW": "S_WW"}


def fmt_smoothing_with_error_color(curr_val, error_color):
    """Formatta il valore smoothing con il colore dell'errore mediano."""
    val_str = f"{int(curr_val)}" if curr_val != "N/A" else "N/A"
    return f"{error_color}{val_str:<7}{RESET}"

def read_windowtraining_metrics(filepath):
    """Legge un file metrics CSV dalla directory windowtraining."""
    try:
        return pd.read_csv(filepath)
    except Exception as e:
        print(f"Avviso: Impossibile leggere il file metrics {filepath}: {e}")
        return None

def compare_windowtraining():
    """Confronta i risultati della windowtraining directory."""
    print("\n" + "=" * 80)
    print("WINDOWTRAINING MODE - RMSE & MAE COMPARISON")
    print("=" * 80)
    
    if not os.path.exists("windowtraining"):
        print("Directory 'windowtraining' non trovata.")
        return
    
    dirs = natural_sort(os.listdir("windowtraining/"))
    metrics_files = [f for f in dirs if f.startswith("metrics-") and f.endswith(".csv")]
    
    if not metrics_files:
        print("Nessun file metrics trovato in windowtraining/")
        return
    
    entries = []  # lista di (filename, metrics_df, config_vars)
    
    # Raccogli tutti i valori di metriche per il gradiente colore
    all_rmse = {"HPT": [], "HPC": []}
    all_mae = {"HPT": [], "HPC": []}
    all_smoothing = {var: [] for var in TARGET_VARS}
    
    for metrics_file in metrics_files:
        metrics_path = f"windowtraining/{metrics_file}"
        metrics_df = read_windowtraining_metrics(metrics_path)
        
        if metrics_df is None:
            continue
        
        # Leggi il config corrispondente
        config_filename = metrics_file.replace("metrics-", "config-").replace(".csv", ".py")
        config_filepath = f"windowtraining/{config_filename}"
        curr_config_vars = read_config_vars(config_filepath)
        
        # Raccogli metriche per gradiente
        for _, row in metrics_df.iterrows():
            target = row["target"]
            if target == "Cycles_to_HPT_SV":
                all_rmse["HPT"].append(row["RMSE"])
                all_mae["HPT"].append(row["MAE"])
            elif target == "Cycles_to_HPC_SV":
                all_rmse["HPC"].append(row["RMSE"])
                all_mae["HPC"].append(row["MAE"])
        
        # Raccogli valori smoothing per gradiente
        for var in TARGET_VARS:
            val = curr_config_vars.get(var, "N/A")
            if val != "N/A":
                all_smoothing[var].append(val)
        
        entries.append((metrics_file, metrics_df, curr_config_vars))
    
    # Header per la tabella
    header = (
        f"{'File':<25} | {'HPT RMSE':<10} | {'HPT MAE':<10} | {'S_HPT':<7} | "
        f"{'HPC RMSE':<10} | {'HPC MAE':<10} | {'S_HPC':<7} | {'S_WW':<7}"
    )
    
    # Best match = minima somma RMSE+MAE su HPT/HPC
    best_file = None
    best_score = float("inf")
    for metrics_file, metrics_df, _ in entries:
        hpt_metrics = metrics_df[metrics_df["target"] == "Cycles_to_HPT_SV"]
        hpc_metrics = metrics_df[metrics_df["target"] == "Cycles_to_HPC_SV"]
        hpt_rmse = hpt_metrics["RMSE"].values[0] if len(hpt_metrics) > 0 else float("inf")
        hpt_mae = hpt_metrics["MAE"].values[0] if len(hpt_metrics) > 0 else float("inf")
        hpc_rmse = hpc_metrics["RMSE"].values[0] if len(hpc_metrics) > 0 else float("inf")
        hpc_mae = hpc_metrics["MAE"].values[0] if len(hpc_metrics) > 0 else float("inf")
        score = hpt_rmse + hpt_mae + hpc_rmse + hpc_mae
        if score < best_score:
            best_score = score
            best_file = metrics_file

    best_hpt_rmse = min_finite(all_rmse["HPT"])
    best_hpt_mae = min_finite(all_mae["HPT"])
    best_hpc_rmse = min_finite(all_rmse["HPC"])
    best_hpc_mae = min_finite(all_mae["HPC"])

    print("Best overall (green) + best per-column values (olive) highlighted.")
    print(header)
    print("-" * len(header))
    
    # Stampa ogni riga
    for metrics_file, metrics_df, curr_config_vars in entries:
        # Estrai metriche per HPT e HPC
        hpt_metrics = metrics_df[metrics_df["target"] == "Cycles_to_HPT_SV"]
        hpc_metrics = metrics_df[metrics_df["target"] == "Cycles_to_HPC_SV"]
        
        hpt_rmse = hpt_metrics["RMSE"].values[0] if len(hpt_metrics) > 0 else float("inf")
        hpt_mae = hpt_metrics["MAE"].values[0] if len(hpt_metrics) > 0 else float("inf")
        hpc_rmse = hpc_metrics["RMSE"].values[0] if len(hpc_metrics) > 0 else float("inf")
        hpc_mae = hpc_metrics["MAE"].values[0] if len(hpc_metrics) > 0 else float("inf")
        
        # Formatta metriche con colori
        hpt_rmse_fmt, hpt_rmse_color = fmt_metric(hpt_rmse, all_rmse["HPT"])
        hpt_mae_fmt, hpt_mae_color = fmt_metric(hpt_mae, all_mae["HPT"])
        hpc_rmse_fmt, hpc_rmse_color = fmt_metric(hpc_rmse, all_rmse["HPC"])
        hpc_mae_fmt, hpc_mae_color = fmt_metric(hpc_mae, all_mae["HPC"])

        hpt_rmse_fmt = highlight_cell(hpt_rmse_fmt, is_best_value(hpt_rmse, best_hpt_rmse))
        hpt_mae_fmt = highlight_cell(hpt_mae_fmt, is_best_value(hpt_mae, best_hpt_mae))
        hpc_rmse_fmt = highlight_cell(hpc_rmse_fmt, is_best_value(hpc_rmse, best_hpc_rmse))
        hpc_mae_fmt = highlight_cell(hpc_mae_fmt, is_best_value(hpc_mae, best_hpc_mae))
        
        # Estrai valori smoothing
        s_hpt = curr_config_vars.get("SMOOTHING_WINDOW_HPT", "N/A")
        s_hpc = curr_config_vars.get("SMOOTHING_WINDOW_HPC", "N/A")
        s_ww = curr_config_vars.get("SMOOTHING_WINDOW_WW", "N/A")
        
        # Usa il colore della metrica migliore (RMSE) per il valore di smoothing
        s_hpt_fmt = fmt_smoothing_with_error_color(s_hpt, hpt_rmse_color)
        s_hpc_fmt = fmt_smoothing_with_error_color(s_hpc, hpc_rmse_color)
        # Per WW, usiamo il colore neutro dato che non abbiamo metriche WW in windowtraining mode
        s_ww_fmt = fmt_smoothing_with_error_color(s_ww, RESET)
        
        # Costruisci riga
        file_fmt = fmt_file_with_best_bg(metrics_file, metrics_file == best_file)
        row = (
            f"{file_fmt} | {hpt_rmse_fmt} | {hpt_mae_fmt} | {s_hpt_fmt} | "
            f"{hpc_rmse_fmt} | {hpc_mae_fmt} | {s_hpc_fmt} | {s_ww_fmt}"
        )
        print(row)

def compare_submissions():
    """Confronta i risultati della directory results (modalità originale)."""
    print("\n" + "=" * 80)
    print("SUBMISSION MODE - MEDIAN ERROR COMPARISON")
    print("=" * 80)
    
    # Impostazioni di Base
    BASE_SUBMISSION = "results/submission_compare.csv"
    BASE_CONFIG = "results/config-2.py" # Cambia se il tuo config base si chiama diversamente

    if not os.path.exists(BASE_SUBMISSION):
        print(f"File di riferimento {BASE_SUBMISSION} non trovato.")
        return

    submission_compare = pd.read_csv(BASE_SUBMISSION)
    base_config_vars = read_config_vars(BASE_CONFIG)

    # ═══════════════════════════════════════════════════════════════
    #  PRIMA PASSATA: raccogli tutti i dati
    # ═══════════════════════════════════════════════════════════════

    dirs = natural_sort(os.listdir("results/"))
    entries = []  # lista di (filename, submission_df, config_vars, differences)

    # Raccogli tutti i valori di smoothing per parametro (per il gradiente)
    all_smoothing = {var: [] for var in TARGET_VARS}
    all_medians = {"WW": [], "HPT": [], "HPC": []}  # per gradiente errori

    for d in dirs:
        if "submission_compare.csv" in d or "config-" in d or ".py" in d:
            continue

        try:
            submission = pd.read_csv(f"results/{d}")
        except Exception:
            continue

        config_filename = d.replace("submission-", "config-").replace(".csv", ".py")
        config_filepath = f"results/{config_filename}"
        curr_config_vars = read_config_vars(config_filepath)

        # Calcola differenze
        differences = {
            "WW": submission["Cycles_to_WW"] - submission_compare["Cycles_to_WW"],
            "HPT": submission["Cycles_to_HPT_SV"] - submission_compare["Cycles_to_HPT_SV"],
            "HPC": submission["Cycles_to_HPC_SV"] - submission_compare["Cycles_to_HPC_SV"]
        }

        # Raccogliere mediane per gradiente
        for label in ["WW", "HPT", "HPC"]:
            med = differences[label].median()
            all_medians[label].append(med)

        entries.append((d, curr_config_vars, differences))

        for var in TARGET_VARS:
            val = curr_config_vars.get(var, "N/A")
            if val != "N/A":
                all_smoothing[var].append(val)

    # ═══════════════════════════════════════════════════════════════
    #  SECONDA PASSATA: stampa con gradienti
    # ═══════════════════════════════════════════════════════════════

    # Best match = minima somma |mediana| su WW/HPT/HPC
    best_file = None
    best_score = float("inf")
    for d, _, differences in entries:
        score = (
            abs(float(differences["WW"].median()))
            + abs(float(differences["HPT"].median()))
            + abs(float(differences["HPC"].median()))
        )
        if score < best_score:
            best_score = score
            best_file = d

    best_ww_abs = min_finite([abs(float(diff["WW"].median())) for _, _, diff in entries])
    best_hpt_abs = min_finite([abs(float(diff["HPT"].median())) for _, _, diff in entries])
    best_hpc_abs = min_finite([abs(float(diff["HPC"].median())) for _, _, diff in entries])

    header = (
        f"{'File':<25} | {'WW (med ± std)':<20} | {'S_WW':<7} | "
        f"{'HPT (med ± std)':<20} | {'S_HPT':<7} | {'HPC (med ± std)':<20} | {'S_HPC':<7}"
    )
    print("Best overall (green) + best per-column values (olive) highlighted.")
    print(header)
    print("-" * len(header))

    for d, curr_config_vars, differences in entries:
        # Calcolo scarti con colori
        error_data = {}  # {label: (formatted_string, color)}
        smoothing_data = {}  # {label: current_value}

        for label_idx, label in enumerate(["WW", "HPT", "HPC"]):
            diff = differences[label]
            med = diff.median()
            std = diff.std()
            formatted_error, error_color = fmt_error(med, std, all_medians[label])

            if label == "WW":
                formatted_error = highlight_cell(formatted_error, is_best_value(abs(float(med)), best_ww_abs))
            elif label == "HPT":
                formatted_error = highlight_cell(formatted_error, is_best_value(abs(float(med)), best_hpt_abs))
            elif label == "HPC":
                formatted_error = highlight_cell(formatted_error, is_best_value(abs(float(med)), best_hpc_abs))

            error_data[label] = (formatted_error, error_color)

        # Estrai valori smoothing
        smoothing_data["WW"] = curr_config_vars.get("SMOOTHING_WINDOW_WW", "N/A")
        smoothing_data["HPT"] = curr_config_vars.get("SMOOTHING_WINDOW_HPT", "N/A")
        smoothing_data["HPC"] = curr_config_vars.get("SMOOTHING_WINDOW_HPC", "N/A")

        # Formatta smoothing con il COLORE DELL'ERRORE (stesso colore della mediana)
        smoothing_formatted = {}
        for label in ["WW", "HPT", "HPC"]:
            _, error_color = error_data[label]
            smoothing_formatted[label] = fmt_smoothing_with_error_color(smoothing_data[label], error_color)

        # Costruisci riga con nuovo ordine: Media - Window - Media - Window - Media - Window
        file_fmt = fmt_file_with_best_bg(d, d == best_file)
        row = (
            f"{file_fmt} | {error_data['WW'][0]} | {smoothing_formatted['WW']} | "
            f"{error_data['HPT'][0]} | {smoothing_formatted['HPT']} | "
            f"{error_data['HPC'][0]} | {smoothing_formatted['HPC']}"
        )
        print(row)

# ═══════════════════════════════════════════════════════════════
#  MAIN LOGIC
# ═══════════════════════════════════════════════════════════════

def parse_args():
    parser = argparse.ArgumentParser(description="Compare results from results/ or windowtraining/ directories")
    parser.add_argument("--mode", choices=["submission", "windowtraining", "auto"], default="auto",
                        help="Comparison mode: submission (results/), windowtraining (windowtraining/), or auto-detect")
    return parser.parse_args()

def natural_sort(l): 
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(l, key=alphanum_key)

# Impostazioni globali (per compatibilità con il codice esistente)
TARGET_VARS = ["SMOOTHING_WINDOW_HPT", "SMOOTHING_WINDOW_HPC", "SMOOTHING_WINDOW_WW"]
SHORT_NAMES = {"SMOOTHING_WINDOW_HPT": "S_HPT", "SMOOTHING_WINDOW_HPC": "S_HPC", "SMOOTHING_WINDOW_WW": "S_WW"}

if __name__ == "__main__":
    args = parse_args()
    
    if args.mode == "auto":
        # Auto-detect mode based on available directories
        has_windowtraining = os.path.exists("windowtraining") and any(
            f.startswith("metrics-") and f.endswith(".csv") 
            for f in os.listdir("windowtraining")
        )
        has_results = os.path.exists("results") and any(
            f.startswith("submission-") and f.endswith(".csv") 
            for f in os.listdir("results")
        )
        
        if has_windowtraining and has_results:
            print("Both windowtraining and results directories found. Select mode:")
            print("1. Windowtraining (RMSE/MAE)")
            print("2. Submission (median errors)")
            choice = input("Enter choice (1/2): ").strip()
            if choice == "1":
                compare_windowtraining()
            else:
                compare_submissions()
        elif has_windowtraining:
            compare_windowtraining()
        elif has_results:
            compare_submissions()
        else:
            print("No valid comparison data found in windowtraining/ or results/ directories.")
    elif args.mode == "windowtraining":
        compare_windowtraining()
    elif args.mode == "submission":
        compare_submissions()

# Backwards compatibility: if imported not executed, run the old behavior
else:
    compare_submissions()