import pandas as pd
import numpy as np
from enum import Enum, auto
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as spst
from typing import Dict, List, Tuple
from . import utils as u
from scipy.optimize import minimize
from sklearn.linear_model import LinearRegression


custom_functions = {"rms": lambda x: np.sqrt(np.mean(x**2))}


class FPerformanceParameter(Enum):
    """
    Base class for engine performance parameters.
    Each member acts as a callable function that applies its formula to a DataFrame.
    """

    def __init__(self, description, formula, colname):
        self.description = description
        self.formula = formula
        self.colname = colname

    def __call__(self, df: pd.DataFrame) -> pd.Series:
        """
        Executes the formula string against the DataFrame columns.
        """
        return df.eval(self.formula, engine="python")


class FPressureRatio(FPerformanceParameter):
    """Rapporti di Pressione (PR)"""

    FAN = ("PR Fan", "Pt2 / Pamb", "PR_FAN")
    LPC = ("PR LPC (Bassa Pres.)", "P25 / Pt2", "PR_LPC")
    HPC = ("PR HPC (Alta Pres.)", "Ps3 / P25", "PR_HPC")
    COMPRESSOR_TOTAL = ("PR Compressore (Tot)",
                        "Ps3 / Pt2", "PR_COMPRESSOR_TOTAL")
    ENGINE_GLOBAL = ("PR Motore (Global)", "Ps3 / Pamb", "PR_ENGINE_GLOBAL")


class FThermalEfficiency(FPerformanceParameter):
    """Efficienza Termica e Turbine"""

    # DELTA_PR_TH_HPC = ("ΔPR/ΔTH HPC", "(Ps3 / P25) / ((T25 - T3) / T25)", "DP_TH_HPC")
    DELTA_PR_TH_HPC_2 = (
        "ΔPR/ΔTH HPC", "(Ps3 / P25) / (T3/T25)", "DP_TH_HPC_2")
    DELTA_T_HPT = ("ΔT Relativo HPT", "(T3 - T45) / T3", "THE_DELTA_T_HPT")
    DELTA_T_LPT = ("ΔT Relativo LPT", "(T45 - T5) / T45", "THE_DELTA_T_LPT")
    DELTA_HPC = ("ΔT Relativo HPC", "(T25-T3)/T25", "THE_DELTA_HPC")
    THERMAL_PROXY = (
        "Proxy Efficienza Termica",
        "(T5 - TAT) / TAT",
        "THE_THERMAL_PROXY",
    )
    SFC_PROXY = ("Proxy Consumo Specifico (SFC)",
                 "WFuel / (T5 - TAT)", "THE_SFC_PROXY")


class FCorrectedSpeed(FPerformanceParameter):
    """Velocità Corrette"""

    # Note: pandas eval handles 'sqrt' automatically if engine='python' or using local_dict
    FAN_SPEED = ("Fan Speed Corretta", "Fan_Speed / sqrt(TAT)", "CS_FAN_SPEED")
    CORE_SPEED = ("Core Speed Corretta",
                  "Core_Speed / sqrt(TAT)", "CS_CORE_SPEED")


def get_all_performance_colnames():
    all_colnames = []
    for cls in FPerformanceParameter.__subclasses__():
        all_colnames.extend([param.colname for param in cls])
    return all_colnames


def _feature_aggregator(
    dfi, group, incols, features, sortcol, window_size=None, step=1, outcols=None
):
    """
    Questa funzione fa cagare sotto tutti i punti di vista
    """
    v_incols = dfi[incols].select_dtypes(include=[np.number]).columns.tolist()
    s_list = [sortcol] if isinstance(sortcol, str) else sortcol

    def groupapply(fn):
        return (
            dfi.groupby(group)[v_incols]
            .rolling(window_size, min_periods=step)
            .apply(fn)
        )

    customs = {"rms": lambda _: groupapply(lambda x: np.sqrt(x**2).mean())}

    dfo = dfi.copy()
    if window_size:
        for feat in features:
            fname = feat if isinstance(feat, str) else feat.__name__
            if fname in customs.keys():
                out = customs[fname](None)
            else:
                out = (
                    dfi.groupby(group)[v_incols]
                    .rolling(window=window_size, min_periods=step)
                    .agg(feat)
                )
            out = out.reset_index(level=0, drop=True)
            out.columns = [f"{c}_{fname}" for c in out.columns]
            dfo = pd.concat([dfo, out], axis=1)
    else:
        dfo = dfi.groupby(group, as_index=False).agg(
            {c: features for c in v_incols})
        dfo.columns = [
            f"{c[0]}_{c[1]}" if isinstance(c, tuple) and c[1] else c[0]
            for c in dfo.columns
        ]

    final_sort = [c for c in s_list if c in dfo.columns]
    if outcols:
        dfo = dfo[[c for c in outcols if c in dfo.columns]]

    return dfo.sort_values(final_sort).reset_index(drop=True)


def performance_features(df: pd.DataFrame, features: list[FPerformanceParameter]):
    if not features:
        features = [
            member
            for cls in [FPressureRatio, FThermalEfficiency, FCorrectedSpeed]
            for member in cls
        ]

    print("Verranno calcolate le seguenti feature:")
    for feat in features:
        print(f"{feat.description} - {feat.formula}")

    dfo = df.copy()
    for feat in features:
        dfo[feat.colname] = feat(dfo)

    return dfo


def calc_statistical_features(
    df: pd.DataFrame,
    features: list[str] = [],
    columns: list[str] = [],
    window_size: int = 0,
    step: int = 1,
    groupby: list[str] = ["ESN"],
    sortby: list[str] = ["esn_index"],
    outcols: list[str] = [],
) -> pd.DataFrame:
    """
    Applica features statistiche a un DataFrame, sia rolling che globali.

    Args:
        df: DataFrame di input
        features: Lista di funzioni statistiche da applicare (es. ['mean', 'std', 'rms'])
        columns: Colonne su cui calcolare le statistiche (default: tutte le numeriche)
        window_size: Dimensione finestra rolling (0 = statistiche globali per gruppo)
        step: Numero minimo di osservazioni richieste nella finestra
        groupby: Colonne per raggruppamento
        sortby: Colonne per ordinamento
        outcols: Colonne finali da restituire (default: tutte)

    Returns:
        DataFrame con features statistiche aggiunte
    """

    if not features:
        features = ["mean", "std", "min", "max",
                    "median", "rms", "kurtosis", "skew"]

    dfo = df.copy().sort_values(by=groupby + sortby).reset_index(drop=True)

    if not columns:
        columns = dfo.select_dtypes(include=[np.number]).columns.tolist()

    calc_cols = [
        c
        for c in columns
        if c in dfo.columns
        and c not in groupby
        and c not in sortby
        and c not in u.META_COLS
    ]

    if not calc_cols:
        return dfo

    results = []
    if window_size > 0:
        # Modalità Rolling Window
        grouped = dfo.groupby(groupby, sort=False)

        for feat in features:
            fname = feat.lower()

            # Applica rolling per ogni gruppo
            if fname in custom_functions:
                # Funzione custom
                res = (
                    grouped[calc_cols]
                    .rolling(window=window_size, min_periods=step)
                    .apply(custom_functions[fname], raw=True)
                )
            else:
                # Funzione standard pandas
                res = (
                    grouped[calc_cols]
                    .rolling(window=window_size, min_periods=step)
                    .agg(fname)
                )

            # Rimuovi i livelli di MultiIndex dal groupby
            if isinstance(res.index, pd.MultiIndex):
                res = res.droplevel(list(range(len(groupby))))

            # Rinomina colonne
            res.columns = [f"{c}_{fname.upper()}" for c in res.columns]
            results.append(res)

    else:
        # Modalità Statistiche Globali (per gruppo)
        grouped = dfo.groupby(groupby, sort=False)

        for feat in features:
            fname = feat.lower()

            if fname in custom_functions:
                # Funzione custom
                res = grouped[calc_cols].transform(custom_functions[fname])
            else:
                # Funzione standard
                res = grouped[calc_cols].transform(fname)  # type: ignore

            if isinstance(res, pd.Series):
                res = res.to_frame()

            res.columns = [f"{c}_{fname.upper()}" for c in res.columns]
            results.append(res)

    if results:
        dfo = pd.concat([dfo] + results, axis=1)

    if outcols:
        available_cols = [
            col
            for col in dfo.columns
            if any(col == req or col.startswith(f"{req}_") for req in outcols)
        ]
        if available_cols:
            dfo = dfo[available_cols]

    return dfo.reset_index(drop=True)


def evaluate_correlation(
    df: pd.DataFrame, target: str, groupby: list[str] = ["ESN"], cols: list[str] = []
):
    if cols == []:
        cols = df.select_dtypes(include=[np.number]).columns.tolist()
    for g in groupby:
        try:
            cols.remove(g)
        except ValueError:
            pass

    for g in u.META_COLS:
        try:
            cols.remove(g)
        except ValueError:
            pass

    results = []
    for col in cols:
        data = df[[col, target]].dropna()
        row = {"feature": col, "n_samples": len(data)}
        # pearson
        p_corr, p_pval = spst.pearsonr(data[col], data[target])
        row["pearson_corr"] = p_corr
        row["pearson_pval"] = p_pval
        row["pearson_abs"] = abs(p_corr)
        # spearman
        s_corr, s_pval = spst.spearmanr(data[col], data[target])
        row["spearman_corr"] = s_corr
        row["spearman_pval"] = s_pval
        row["spearman_abs"] = abs(s_corr)

        # # linear_regression
        # linreg = spst.linregress(data[col], data[target])
        # row['linear_regression'] = linreg

        row["tot_val"] = abs(s_corr) * 0.8 + abs(p_corr) * 0.2

        results.append(row)
    dfo = pd.DataFrame(results)
    del results

    return dfo


def evaluate_correlation_per_snap(df: pd.DataFrame, target: str, top_n: int = 5):
    """
    Calcola le top feature correlate al target per ogni singolo snapshot.
    Se la colonna 'Snapshot' non è presente, esegue un fallback su 'ESN' (global).
    """
    # Fallback rapido se non abbiamo la colonna Snapshot
    if "Snapshot" not in df.columns:
        # Esegui valutazione a livello di ESN
        df_global = evaluate_correlation(df, target=target, groupby=["ESN"])
        df_global["Snapshot"] = "ALL"
        return df_global.sort_values(by='tot_val', key=abs, ascending=False).head(top_n)

    all_snap_results = []

    # Identifichiamo i vari snapshot presenti
    unique_snaps = df["Snapshot"].unique()

    for snap_val in unique_snaps:
        snap_df = df[df["Snapshot"] == snap_val].copy()

        if len(snap_df) < 5:
            continue

        corr_results = evaluate_correlation(
            snap_df, target=target, groupby=["ESN", "Snapshot"]
        )
        corr_results["Snapshot"] = snap_val
        all_snap_results.append(corr_results)

    report_df = pd.concat(all_snap_results).reset_index(drop=True)
    cols = ["Snapshot", "feature", "spearman_corr",
            "pearson_corr", "tot_val", "n_samples"]
    return report_df[cols]


def pipeline_hpc(
    df: pd.DataFrame,
    features: list[FPerformanceParameter],
    cols,
    statistical_features: list[str] = ["mean", "rms"],
    window: int = 100,
    step: int = 25,
    stat_groupby: list[str] = ["ESN", "Snapshot"],
    stat_sortby: list[str] = ["ESN", "snap_index"],
    target="Cycles_to_HPC_SV",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    # dff = df.groupby(['ESN', "esn_index"], as_index=False).mean()
    dff = performance_features(df, features)
    dff = calc_statistical_features(
        dff,
        features=["mean", "rms"],
        columns=cols,
        groupby=["ESN"],
        window_size=window,
        step=step,
    ).dropna()
    val = evaluate_correlation(dff, target=target, groupby=["ESN"])
    return (dff, val)


def pipeline_hpt(
    df: pd.DataFrame,
    features: list[FPerformanceParameter],
    cols,
    statistical_features: list[str] = ["mean", "rms"],
    window: int = 100,
    step: int = 25,
    stat_groupby: list[str] = ["ESN", "Snapshot"],
    stat_sortby: list[str] = ["ESN", "snap_index"],
    target="Cycles_to_HPT_SV",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    dff = performance_features(df, features)
    dff = calc_statistical_features(
        dff,
        features=["mean", "rms"],
        columns=cols,
        groupby=["ESN"],
        window_size=window,
        step=step,
    ).dropna()
    val = evaluate_correlation_per_snap(dff, target=target)
    return (dff, val)


def pipeline_ww(
    df: pd.DataFrame,
    features: list[FPerformanceParameter],
    cols,
    statistical_features: list[str] = ["mean", "rms"],
    window: int = 100,
    step: int = 25,
    stat_groupby: list[str] = ["ESN"],
    stat_sortby: list[str] = ["ESN", "esn_index"],
    target="Cycles_to_WW",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Pipeline dedicata per Water Wash (WW).
    Identifica i punti di WW tramite `Cumulative_WWs` e valuta le feature correlate
    rispetto al target (Cycles_to_WW). Se non ci sono eventi WW, effettua il fallback
    a una valutazione globale per ESN.
    """
    dff = performance_features(df, features)
    dff = calc_statistical_features(
        dff,
        features=statistical_features if statistical_features else ["mean", "rms"],
        columns=cols,
        groupby=["ESN"],
        window_size=window,
        step=step,
    ).dropna()

    # Identifica i punti WW
    wws = u.df_get_step_points(df, "Cumulative_WWs")

    if wws is None or wws.empty:
        # fallback: valutazione globale
        val = evaluate_correlation(dff, target=target, groupby=["ESN"])
        return (dff, val)

    # Usa il campo global_index per mappare gli eventi (se presente)
    if "global_index" in wws.columns and "global_index" in dff.columns:
        ww_indices = set(wws["global_index"].unique())
        dff_ww = dff[dff["global_index"].isin(ww_indices)]
    else:
        # fallback: usa i punti trovati direttamente (match su ESN e snap_index se presenti)
        if set(["ESN", "snap_index"]).issubset(wws.columns) and set(["ESN", "snap_index"]).issubset(dff.columns):
            merged = pd.merge(wws[["ESN", "snap_index"]].drop_duplicates(), dff, on=["ESN", "snap_index"], how="inner")
            dff_ww = merged
        else:
            dff_ww = pd.DataFrame()

    if dff_ww.empty:
        val = evaluate_correlation(dff, target=target, groupby=["ESN"])
    else:
        val = evaluate_correlation(dff_ww, target=target, groupby=["ESN"])

    return (dff, val)


def calculate_hpt_health_index(
    df: pd.DataFrame, t3_col: str, t45_col: str, to_next_hpc_col: str
) -> pd.DataFrame:
    df["HI_HPT"] = np.nan

    for esn in df["ESN"].unique():
        engine_data = df[df["ESN"] == esn].sort_values("snap_index")

        # Individuo l'evento HPC
        zero_indices = engine_data.index[engine_data[to_next_hpc_col] == 0].tolist(
        )

        if zero_indices:
            idx_zero = zero_indices[-2]
            pos_zero = engine_data.index.get_indexer([idx_zero])[0]

            # Prendo una finestra a cavallo dell'evento di manutenzione dell'hpc
            start = max(0, pos_zero - 10)
            end = min(len(engine_data), pos_zero + 11)
            calib_window = engine_data.iloc[start:end]

            # Funzione obiettivo: minimizzare la deviazione standard dell'HI
            # mentre l'altro componente (HPC) viene sostituito
            def objective(a):
                hi = -a * calib_window[t3_col] - calib_window[t45_col]
                return np.std(hi)

            res = minimize(objective, x0=1.0)
            best_alpha = res.x[0]

        else:
            # Fallback se non c'è l'evento: uso tutto il motore
            best_alpha = 1.0  # o una media globale

        # Applico l'alpha trovato a tutto il motore
        df.loc[df["ESN"] == esn, "HI_HPT"] = (
            -best_alpha * engine_data[t3_col] - engine_data[t45_col]
        )

    return df


def calculate_hpt_health_index_all(
    df: pd.DataFrame, t3_col: str, t45_col: str, to_next_hpc_col: str
) -> pd.DataFrame:
    df["HI_HPT"] = np.nan

    for esn in df["ESN"].unique():
        engine_mask = df["ESN"] == esn
        engine_data = df[engine_mask].sort_values("snap_index")

        # Pulizia dati: rimuoviamo eventuali NaN per la calibrazione
        calib_data = engine_data.dropna(
            subset=[t3_col, t45_col, to_next_hpc_col])

        if not calib_data.empty:
            # FUNZIONE OBIETTIVO:
            # Vogliamo che HI_HPT non "segua" il trend dell'HPC.
            # Minimizziamo la deviazione standard dell'HI su tutta la vita.
            def objective(a):
                hi = -a * calib_data[t3_col] - calib_data[t45_col]
                return np.std(hi)

            res = minimize(objective, x0=1.0)
            best_alpha = res.x[0]
        else:
            best_alpha = 1.0

        # Applichiamo a tutto il motore
        df.loc[engine_mask, "HI_HPT"] = (
            -best_alpha * engine_data[t3_col] - engine_data[t45_col]
        )

    return df


def fit_hpt_mapping(
    df: pd.DataFrame, to_next_hpt_col: str
) -> tuple[pd.DataFrame, dict]:
    """
    Usa l'evento HPT per definire la retta di predizione
    assumendo cicli = 0 all'evento HPT.
    """
    engine_params_hpt = {}

    for esn in df["ESN"].unique():
        engine_data = df[df["ESN"] == esn].dropna(
            subset=["HI_HPT", to_next_hpt_col])

        if not engine_data.empty:
            # Regressione lineare: HI_HPT -> Cycles_to_HPT_SV
            X = engine_data[["HI_HPT"]].values
            y = engine_data[to_next_hpt_col].values

            model = LinearRegression().fit(X, y)

            # Salviamo i parametri specifici per questo motore
            engine_params_hpt[esn] = {
                "slope": model.coef_[0],
                "intercept": model.intercept_,
            }

            # Creiamo la colonna della predizione lineare
            df.loc[df["ESN"] == esn, "hpt_linear_pred"] = model.predict(X)

    return df, engine_params_hpt
