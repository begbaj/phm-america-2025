# Preprocessing Guide

Questa guida spiega i metodi di preprocessing disponibili e come utilizzarli nel flusso `tasks/01_preprocess.py`.

## Architettura del Flusso

Il task `01_preprocess.py` esegue una sequenza di step:

1. **Preprocess Data**: Applica outlier removal, missing value filling, e smoothing
2. **Aggregate Snapshots**: Aggrega i dati per snapshot (8 snapshot → 1 riga per ESN per ciclo)
3. **Feature Engineering Pipeline**: Calcola feature statistiche e correlazioni con il target (HPC/HPT/WW)
4. **Save Outputs**: Salva metadati e feature computate
5. **Plot Aggregated**: Visualizza feature per motore
6. **Plot Per-Snapshot**: Visualizza feature per snapshot
7. **Train Models**: Allena modelli predittivi

## Metodi di Preprocessing

### 1. Outlier Detection

Utilizzato in **Preprocess Data** per identificare e rimuovere valori anomali.

#### Isolation Forest (default)
- **Metodo**: Algoritmo non supervisionato basato su alberi random
- **Parametro**: `--outlier-method isoforest`, `--outlier-threshold <0.08>` (contamination rate)
- **Uso**: Migliore per distribuzioni non normali, tollerante alle scale diverse
- **Default threshold**: 0.08 (8% dei dati considerati outlier)
- **Comando**:
  ```bash
  python tasks/01_preprocess.py --steps preprocess --outlier-method isoforest --outlier-threshold 0.08
  ```

#### Z-score
- **Metodo**: Misura deviazione standard dalla media
- **Parametro**: `--outlier-method zscore`, `--outlier-threshold <3>` (numero di std deviations)
- **Uso**: Ideale per distribuzioni normali; sensibile a outlier estremi
- **Default threshold**: 3 (valori > 3σ dalla media sono outlier)
- **Comando**:
  ```bash
  python tasks/01_preprocess.py --steps preprocess --outlier-method zscore --outlier-threshold 3
  ```

#### IQR (Interquartile Range)
- **Metodo**: Basato su quartili (Q1, Q3)
- **Parametro**: `--outlier-method iqr`, `--outlier-threshold <1.5>` (moltiplicatore IQR)
- **Uso**: Robusto, indipendente dalla distribuzione
- **Default threshold**: 1.5 (outlier = valori fuori Q1 - 1.5*IQR e Q3 + 1.5*IQR)
- **Comando**:
  ```bash
  python tasks/01_preprocess.py --steps preprocess --outlier-method iqr --outlier-threshold 1.5
  ```

### 2. Missing Values Filling

Utilizzato **prima e dopo** il rilevamento outlier per riempire valori NaN.

#### Strategia implementata
1. **Fleet Mean**: Calcola la media del gruppo (altri motori) per lo stesso (Snapshot, Cycles_Since_New)
2. **Forward Fill (per ESN)**: Riempie NaN residui propagando valori anteriori
3. **Backward Fill (per ESN)**: Riempie NaN all'inizio della serie propagando valori successivi

```
Missing Value → Fleet Mean (se disponibile)
             → Forward Fill per ESN (se ancora NaN)
             → Backward Fill per ESN (se ancora NaN)
```

**Parametri**: Nessun parametro configurabile (strategia fissa)
**Comando**: Automatico in ogni esecuzione di `--steps preprocess`

### 3. Smoothing (Data Smoothing)

Utilizzato in **Preprocess Data** per ridurre il rumore dai sensori.

#### Rolling Mean (media mobile semplice, default)
- **Metodo**: Calcola media su finestra scorrevole
- **Parametri**: 
  - `--smoothing-method rolling_mean`
  - `--smoothing-window <100>` (dimensione finestra, default 100)
  - `--smoothing-step <25>` (minimo numero di punti per calcolare media, default 25)
- **Uso**: Semplice, veloce, mantiene continuità; può smussare troppo
- **Comando**:
  ```bash
  python tasks/01_preprocess.py --steps preprocess --smoothing-method rolling_mean --smoothing-window 100 --smoothing-step 25
  ```

#### Exponential Smoothing (EMA - Exponential Moving Average)
- **Metodo**: Media mobile ponderata esponenzialmente (recenti più pesanti)
- **Parametri**:
  - `--smoothing-method exponential`
  - `--smoothing-window <100>` (interpretato come `span` in pandas ewm)
  - `--smoothing-step` (ignorato)
- **Uso**: Ideale per trend, reagisce meglio a cambiamenti recenti
- **Comando**:
  ```bash
  python tasks/01_preprocess.py --steps preprocess --smoothing-method exponential --smoothing-window 100
  ```

#### Savitzky-Golay (filtro polinomiale)
- **Metodo**: Stima polinomio locale su finestra, mantiene derivate
- **Parametri**:
  - `--smoothing-method savitzky_golay`
  - `--smoothing-window <75>` (deve essere dispari; se pari diventa dispari-1)
  - `--smoothing-step` (ignorato)
- **Uso**: Migliore per preservare feature locali e pendenze; richiede window maggiore
- **Comando**:
  ```bash
  python tasks/01_preprocess.py --steps preprocess --smoothing-method savitzky_golay --smoothing-window 75
  ```

## Feature Engineering Pipeline

Eseguito in **Feature Engineering Pipeline** (step 3) su dati aggregati.

- **Target**: Scegli HPC, HPT, o WW (colonna `Cycles_to_*`)
- **Statistical Features**: Comma-separated list (es. `mean,rms`) calcolate con rolling window
- **Rolling Window**: Dimensione finestra per calcolare feature statistiche (default 100)
- **Min Period**: Minimo numero di punti per calcolare feature (default 25)

**Nota**: Pipeline target e rolling window sono **sempre** utilizzati se esegui `--steps pipeline`. Non sono configurabili se skippi il pipeline step.

## Comandi Comuni

### Preprocess Completo (Default)
```bash
python tasks/01_preprocess.py --steps preprocess,aggregate,pipeline,save \
  --outlier-method isoforest --outlier-threshold 0.08 \
  --smoothing-method rolling_mean --smoothing-window 100 \
  --pipeline-target HPC --pipeline-stats mean,rms
```

### Preprocess Solo (senza Feature Engineering)
```bash
python tasks/01_preprocess.py --steps preprocess,aggregate \
  --outlier-method zscore --outlier-threshold 3 \
  --smoothing-method exponential --smoothing-window 50
```

### Feature Engineering su Dati Preprocessati Esistenti
```bash
python tasks/01_preprocess.py --steps pipeline,save \
  --pipeline-target HPT --pipeline-stats mean,rms,kurtosis
```

### Con Training Modelli
```bash
python tasks/01_preprocess.py --steps preprocess,aggregate,pipeline,save,train_models \
  --pipeline-target HPC --models linear,xgb
```

## GUI Web

Accedi a `src/notebooks/web_gui/main.py` e naviga a `http://localhost:8000`:

- **Preprocessing Parameters**: Outlier method, threshold, smoothing method, window, min period
- **Feature Engineering Pipeline**: Target (HPC/HPT/WW), statistical features, rolling window parameters
- **Execution Steps**: Seleziona quali step eseguire (preprocess, aggregate, pipeline, save, plot_agg, plot_snap, train_models)
- **Models to Train**: Seleziona modelli da allenare (Linear, RF, XGBoost, Transformer)

## Output Files

Dopo esecuzione, i seguenti file sono generati:

- `Data/snapshot_tables/training.csv`: Dati preprocessati (16 sensori, indici vari, colonne fault)
- `Data/snapshot_tables/averaged_final.csv`: Dati aggregati per snapshot (8 snapshot → 1 riga per ESN × ciclo)
- `Data/features/training_feature_<TARGET>_data.csv`: Feature computate (con solo top feature)
- `Data/features/training_feature_<TARGET>_metadata.csv`: Metadati e correlazioni feature-target

## Troubleshooting

### "Nessuna colonna sensore valida trovata"
- I sensori nel dataset hanno prefisso `Sensed_` (es. `Sensed_T3`)
- Il codice dovrebbe gestirli automaticamente; se vedi questo warning, verifica che il CSV input abbia colonne sensore

### Pipeline KeyError: "'to_next_*_cycle' not in index"
- Assicurati di aver eseguito `--steps aggregate` prima di `--steps pipeline`
- Il target deve essere una colonna presente in dati aggregati (Cycles_to_HPC_SV, Cycles_to_HPT_SV, Cycles_to_WW)

### Smoothing produce NaN in serie breve
- Se `smoothing-window > len(ESN data)`, aumenta `smoothing-step` o riduci `smoothing-window`
- Savitzky-Golay richiede window >= polynomial_order + 1 (generalmente 4 in questo caso)
