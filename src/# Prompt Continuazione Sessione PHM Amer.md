# Prompt Continuazione Sessione PHM America 2025

## Stato Attuale Progetto (28 Gennaio 2026)

### Completato nella sessione odierna:
1. **Fixato KeyError 'snap_index'** in 04_residual_analysis.py (linea 154)
   - Aggiunto: creazione dinamica di snap_index se non esiste nel dataframe aggregato

2. **Implementato RUL Comparison Plot** nel modulo residual analysis:
   - Aggiunto parametro `--target-rul` per selezionare tra HPC, HPT, WW
   - Creato dual-axis plot che confronta residui calcolati vs RUL effettivo
   - Mapping automatico: HPC→Cycles_to_HPC_SV, HPT→Cycles_to_HPT_SV, WW→Cycles_to_WW
   - File output: `W{window_size}_residuals_vs_rul_{target_rul}.png`

3. **Aggiornamenti GUI**:
   - Aggiunto dropdown "Target RUL" nella sezione 04_residual_analysis.py
   - JavaScript updated per raccogliere e inviare il parametro target_rul
   - main.py updated per mappare target_rul in --target-rul CLI arg

### Architettura Attuale:
- **Backend**: FastAPI (web_gui/main.py) + subprocess task execution
- **Frontend**: Jinja2 templates + vanilla JS + localStorage persistence
- **Task Structure**:
  - 01_preprocess.py: Preprocessing con selezione step + metodi outlier/smoothing
  - 02_feature_engineering.py: Feature pipeline con statistical features selezionabili
  - 03_train_models.py: Training modelli usando SOLO engineered features
  - 04_residual_analysis.py: Residual analysis con RUL comparison plot (NUOVO)

### Dati Canonici:
- ESN in uppercase
- Colonne RUL: Cycles_to_HPC_SV, Cycles_to_HPT_SV, Cycles_to_WW
- Primary index: Cycles_Since_New
- Sensor names: unprefixed (Altitude non Sensed_Altitude)

### Prossimi Step Suggeriti:
1. **Testing**: Eseguire 04_residual_analysis.py con diversi target_rul via GUI
2. **Validazione**: Verificare che i plot mostrano dati corretti (non immagini bianche)
3. **Raffinamento**: Se necessario, aggiungere filtri per singoli motori (ESN)
4. **Documentation**: Aggiornare user guide se non già fatto

### Note Tecniche:
- Residual analysis: windows_to_test=[100], sliding_window_step=50 (350× faster than original)
- Preprocessing methods: isoforest/zscore/iqr (outlier), rolling_mean/exponential/savitzky_golay (smoothing)
- RUL columns all checked before use - graceful fallback if missing

### File Modificati in Questa Sessione:
- `/src/notebooks/tasks/04_residual_analysis.py`: Added argparse, snap_index fix, RUL comparison plot
- `/src/notebooks/web_gui/templates/index.html`: Added 04_residual_analysis section with target-rul dropdown
- `/src/notebooks/web_gui/main.py`: Added target_rul parameter mapping

### Known Issues (if any):
- Nessuno noto al momento - tutti i problemi precedenti risolti

### Comandi Utili per Continuare:
```bash
cd /home/began/srcs/begbaj/phm-america-2025
source .venv/bin/activate
python -m src.notebooks.web_gui.main  # Start FastAPI server
# Then navigate to http://localhost:8000