# Predictor Pipeline

## End-to-end RUL pipeline

```mermaid
flowchart TD
    A[Raw CSV data] --> B[Aggregate by engine-cycle]
    B --> C[Outlier removal\nZ-score on sensors]
    C --> D[Missing value fill\nInterpolation + fleet mean + ffill/bfill]
    D --> E[Drop non-common sensors\ntrain/validation/test intersection]

    E --> F[Prepare training subset\ninclude/exclude TESTING_ESN]
    F --> G[HITrainer linear models]
    G --> H[Residual computation]
    H --> I[HI coefficient stage\ndefault or optimization]

    I --> J[Cycle classifier\nLightGBM multiclass]
    I --> K[Gap correction\nLightGBM regressor]
    J --> K

    K --> L[Inference on validation/test]
    L --> M[Classify cycle]
    M --> N[Scale HI by cycle]
    N --> O[Gap correction]
    O --> Q[Cycles_to_HPT_SV / Cycles_to_HPC_SV]
```

## Water Wash pipeline

```mermaid
flowchart TD
    A[Engine data + residuals] --> B[Remove maintenance effects\non Sensed_T45]
    B --> C[Build one T45 signal]
    C --> D[Linear trend slope]
    C --> E[WW event detection\nrolling deviation from trend]
    D --> E
    E --> F[Cycles to next WW\nfrom end of sequence]
```

## Windowtraining LOEO

```mermaid
flowchart TD
    A[Preprocessed training set] --> B[Select one ESN as holdout]
    B --> C[Train full RUL stack on remaining ESNs]
    C --> D[Predict final HPT/HPC on holdout ESN]
    D --> E[Store fold prediction + truth]
    E --> F{More ESNs?}
    F -->|Yes| B
    F -->|No| G[Aggregate RMSE/MAE across folds]
    G --> H[Save metrics-N.csv for optimizer]
```

## Notes

- Aggregation is applied before outlier removal and missing-value filling.
- Sensor alignment updates active sensor lists to the shared intersection.
- RUL correction uses classifier output at inference and nearest known cycle fallback if a cycle is unseen.
- WW slope estimation and event detection now use the same T45 signal.
- WW prediction can reuse a global slope equal to the mean training slope (`WW_USE_TRAINING_MEAN_SLOPE=True`).
- Windowtraining mode uses LOEO cross-validation on the labeled training set.
- WW can be included in windowtraining objective with a proxy LOEO metric (`--windowtraining-include-ww` and optimizer `--include-ww-objective`).
