# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Running the pipeline

Files are numbered by execution order. Binary pipeline = steps 01–20. Multiclass pipeline = steps 13b–19b.

```
# Binary pipeline (steps 14–18)
python src/modeling/14_lgbm_baseline.py   # Tầng 1: LightGBM tabular baseline
python src/modeling/15_rf_baseline.py     # Tầng 1b: Random Forest baseline
python src/modeling/16_lstm_hybrid.py     # Tầng 2: LSTM Hybrid (requires tensors from step 13)
python src/modeling/17_tcn_hybrid.py      # Tầng 2 alt: TCN Hybrid (faster on CPU)
python src/modeling/18_stacking_ensemble.py  # Tầng 3: LR meta-learner over 4 base learners

# Multiclass pipeline (steps 14b–18b)
python src/modeling/14b_lgbm_multiclass.py
python src/modeling/15b_rf_multiclass.py
python src/modeling/16b_lstm_multiclass.py
python src/modeling/17b_tcn_multiclass.py
python src/modeling/18b_stacking_multiclass.py
```

**LightGBM** outputs to `models/14_lgbm_baseline/`: `lgbm_{tag}.joblib`, `importance_{tag}.csv`, `test_predictions_{tag}.csv` (columns: `Date`, `y_true`, `y_prob_lgb`, `y_pred_lgb`), `results_{tag}.json`, `feature_cols_{tag}.json`.

**LSTM** outputs to `models/16_lstm_hybrid/`: `lstm_{tag}_win{W}.pt` (best window's state_dict), `test_predictions_{tag}.csv` (columns: `y_true`, `y_prob_lstm`, `y_pred_lstm` — no Date), `results_{tag}.json` (includes `best_window`).

**TCN** outputs to `models/17_tcn_hybrid/`: `tcn_{tag}_win{W}.pt`, `test_predictions_{tag}.csv` (columns: `y_true`, `y_prob_tcn`, `y_pred_tcn` — no Date), `results_{tag}.json` (includes `best_window`).

**Stacking** outputs to `models/18_stacking_ensemble/`: `meta_lr_{tag}.joblib`, `test_predictions_{tag}.csv` (columns: `Date`, `y_prob_lgb/lstm/tcn/rf`, `y_true`, `y_prob_stack`, `y_pred_stack`), `results_{tag}.json`.

**Multiclass** outputs to `models/14b_lgbm_multiclass/`, `16b_lstm_multiclass/`, etc. Predictions have 3 columns: `y_prob_{model}_{down,flat,up}`.

Run 14_lgbm_baseline.py first to establish a floor AUC. LSTM/TCN must beat it on PR-AUC and F1 to be considered worthwhile. Stacking ensemble requires all four base models to have completed.

---

## Running the full pipeline

All scripts must be run **from the project root** (not from inside `src/`), because ingestion scripts resolve `config/coordinates.json` and `data/` relative to the working directory. Preprocessing scripts use `os.path.abspath(__file__)` so they are path-safe from anywhere.

```
# Activate the venv first
venv\Scripts\activate        # Windows
source venv/bin/activate     # Unix

# Stage 1 – Ingestion (run once to pull raw data)
python src/ingestion/01_market_ingestion.py
python src/ingestion/02_weather_ingestion.py
python src/ingestion/03_macro_ingestion.py
python src/ingestion/04_farming_ingestion.py

# Stage 2 – Preprocessing (order matters within weather)
python src/preprocessing/05_weather_anomaly_removal.py   # raw → weather_despiked/
python src/preprocessing/06_weather_preprocessing.py     # weather_despiked/ → weather_clean/
python src/preprocessing/07_weather_weekly_aggregation.py
python src/preprocessing/08_market_acu_filter.py
python src/preprocessing/09_macro_preprocessing.py
python src/preprocessing/10_farming_preprocessing.py

# Stage 3 – Integration
python src/11_data_integration.py

# Stage 4 – Lag analysis
python src/12_lag_analysis.py

# Stage 5 – Tensor packing
python src/13_tensor_packing.py           # binary tensors → data/tensors/
python src/13b_tensor_packing_mc.py       # multiclass tensors → data/tensors_multiclass_soft/

# Stage 6 – Modeling Binary
python src/modeling/14_lgbm_baseline.py
python src/modeling/15_rf_baseline.py
python src/modeling/16_lstm_hybrid.py
python src/modeling/17_tcn_hybrid.py

# Stage 6b – Modeling Multiclass
python src/modeling/14b_lgbm_multiclass.py
python src/modeling/15b_rf_multiclass.py
python src/modeling/16b_lstm_multiclass.py
python src/modeling/17b_tcn_multiclass.py

# Stage 7 – Stacking
python src/modeling/18_stacking_ensemble.py
python src/modeling/18b_stacking_multiclass.py

# Stage 8 – Backtesting
python src/modeling/19_backtesting_engine.py
python src/modeling/19b_backtesting_mc.py

# Stage 9 – Report
python src/20_pipeline_report.py
```

There are no tests or linters configured in this project.

## Data directory layout

```
data/
  raw/
    market/      coffee_market.csv, corn_market.csv
    weather/     {commodity}_{location}.csv  (one file per crop-region pair)
    macro/       usd_brl_exchange.csv, us_inflation.csv
    farming/     coffee_calendar.csv, corn_calendar.csv
  preprocessed/
    market/      daily_{coffee,corn}_clean.csv, weekly_{coffee,corn}_clean.csv
    weather/
      weather_despiked/    despiked_{original_name}.csv
      weather_clean/       clean_{original_name}.csv
      weather_weekly/      weekly_{original_name}.csv   (not used downstream)
    macro/       weekly_usd_brl_clean.csv, weekly_us_inflation_clean.csv
    farming/     weekly_{coffee,corn}_calendar.csv
  integrated/    integrated_{coffee,corn}_{daily,weekly}.csv
  tensors/
    win_{14,30,45}/  (daily) or win_{4,8,12}/  (weekly)
      {dataset_name}/
        X_{train,val,test}_dynamic.npy
        X_{train,val,test}_static.npy   (if static cols present)
        y_{train,val,test}.npy
        {train,val,test}_metadata.parquet
        scaler_dyn_*.joblib
        scaler_stat_*.joblib
```

## Architecture overview

The pipeline is a sequential ETL → feature engineering → tensor packing workflow for binary price-direction classification on coffee (KC=F) and corn (ZC=F) futures.

### Key design decisions

**Universal weekly anchor: `resample('W-MON')`**  
All weekly resampling in every file must use `resample('W-MON')`. Using the pandas default `resample('W')` (which is W-SUN) causes a 1-day index offset that produces all-NaN columns after merge. This affects: `market_acu_filter.py`, `macro_preprocessing.py`, `farming_preprocessing.py`, `data_integration.py`.

**Two market frequencies, separate files**  
`market_acu_filter.py` outputs both `daily_*_clean.csv` (post-ACU, pre-resample) and `weekly_*_clean.csv`. `data_integration.py` loads all four and routes them to the correct pipeline. Never pass a weekly-frequency DataFrame to the daily build path.

**Weather backbone per crop**  
Each crop has 5 monitoring regions (see `config/coordinates.json`). `data_integration.py` averages all same-crop region DataFrames into a single backbone before joining to market data. File names starting with `coffee_` belong to the coffee backbone; `corn_` to corn.

**Feature provenance — what is computed where**  
- Financial indicators (RSI_14, SMA_*, EMA_*, BB_*, MACD_*, momentum_*): computed in `market_acu_filter.py`, preserved through weekly resample via explicit `agg_dict`.
- Macro lag features (Close_lag_1w, log_return_lag_1w, etc.): computed in `macro_preprocessing.py`; `data_integration.py` must not recompute `usd_log_return` or it overwrites the pre-lagged version.
- Cyclical calendar features (sin/cos week/month, is_harvest, is_planting): computed in `farming_preprocessing.py`; after `add_prefix('cal_')` in integration they become `cal_sin_week`, `cal_cos_week`, `cal_sin_month`, `cal_cos_month`, `cal_is_harvest`, `cal_is_planting`. These exact names are used in `tensor_packing.py` as the static features list.
- `BB_upper`/`BB_lower`/`RSI_adj`/`EMA_adj`/`volatility`: recomputed in `build_integrated_dataset` on `currency_adjusted_close` (not raw Close). `BB_middle` is intentionally dropped from the weekly resample agg to avoid using an inconsistent base.

**Target variable**  
Binary: `(return_future > 0.05)` where `return_future` is the 7-day-ahead (daily) or 1-week-ahead (weekly) relative return of `currency_adjusted_close`. The last `horizon` rows of each split have NaN targets and are excluded via the embargo gaps.

**Train/Val/Test split in tensor_packing.py**  
70% train / 10% val / 20% test, with two embargo gaps of `horizon` rows at each boundary. Feature selection (LightGBM null importances + TimeSeriesSplit) runs on `train_df` only. Scalers are fit on train and applied to val and test.

### Modeling layer (`src/modeling/`)

**`lgbm_baseline.py`** — Tầng 1, the tabular benchmark. Reads directly from `data/integrated/` (not the `.npy` tensors), takes a snapshot at time T (one row = all features), and applies the same 70/10/20 + embargo split as `tensor_packing.py`. LightGBM works on the lag-encoded row rather than a 3D sequence because the lag features (momentum, rolling windows, close_lag_*) already encode history.

Key design points:
- `scale_pos_weight` = neg/pos ratio, capped at 10 to prevent instability when base rate is very low
- `find_optimal_threshold()` maximizes F1 on the **val set only**; the chosen threshold is then applied unchanged to the test set — do not retune on test or combined data
- `test_predictions_{tag}.csv` stores `y_prob_lgb` per test row — this is the stacking input for Tầng 3 (LightGBM + LSTM + TCN ensemble)
- Weekly params override (`PARAMS_WEEKLY_OVERRIDE`) reduces complexity because weekly datasets are much smaller
- Overfit detection: logs a warning when train AUC − val AUC > 0.10

Any subsequent deep-learning model (LSTM, TCN, TFT) must outperform this baseline on PR-AUC and F1 to be considered worthwhile.

**`lstm_hybrid.py`** — Tầng 2, sequence model. Reads `.npy` tensors from `data/tensors/win_{W}/{file_tag}/` (output of `tensor_packing.py`). Iterates over all available window sizes and picks the one with best val AUC.

Key design points:
- `EarlyStopping.best_state` stores the best epoch's `state_dict` in RAM as a dict of cloned tensors; `restore_best()` is called after the training loop ends, before final evaluation — evaluating on the last epoch instead of the best checkpoint is a common mistake
- `ReduceLROnPlateau` (patience=5, factor=0.5) and `EarlyStopping` (patience=15) are coordinated: expect 1–2 LR reductions before stop
- `BCEWithLogitsLoss(pos_weight=spw)` mirrors the `scale_pos_weight` approach in LightGBM; capped at 10 for the same stability reason
- `bidirectional=False` by default — safe because the full window is historical (no look-ahead); enabling BiLSTM on layer 1 is a valid experiment
- `test_predictions_{tag}.csv` stores `y_prob_lstm` per test row — same index convention as LightGBM's `y_prob_lgb` for Tầng 3 stacking

### `config/coordinates.json`

Drives both ingestion scripts: weather fetch locations (lat/lon per region) and crop calendar definitions (which months are harvest/flowering/planting). Farming ingestion generates binary calendar flags from these month lists.

### Timezone handling

All date columns are UTC-normalized throughout the pipeline. Market data from yfinance arrives with timezone info; weather from open-meteo arrives as UTC; BLS CPI is month-start dates shifted +1 month +12 days to approximate the actual release date before being forward-filled to daily frequency.
