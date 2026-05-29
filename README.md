# DS108 — Agricultural Commodity Price Forecasting

**Multi-source ML pipeline for Coffee (KC=F) and Corn (ZC=F) futures · 2010–2026**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Dataset](https://img.shields.io/badge/Dataset-Kaggle-blue)](https://www.kaggle.com/datasets/khimtagia/agricommodity-futures-multi-source-ml-dataset)
[![Python](https://img.shields.io/badge/Python-3.10%2B-green)](requirements.txt)

---

## Overview

End-to-end data pipeline and ML modeling system for predicting price movement direction (>2.5% in 7 days) of agricultural commodity futures. Integrates **4 heterogeneous data sources** — including LLM-extracted crop calendar features (Bonus 2) — with strict causal preprocessing and zero data leakage guarantee.

### Key Results

| Metric | Value |
|--------|-------|
| Best AUC-ROC (Coffee Weekly LSTM) | **0.557** |
| AUC-ROC LLM calendar vs synthetic (Coffee Weekly) | +8.7 pp |
| Sharpe ratio (Binary Coffee Daily Stack) | **2.154** |
| Sharpe ratio (MC Coffee Daily Stack) | **1.839** |
| Alpha vs B&H (Coffee Weekly RF) | **+16.4 pp** |
| Alpha vs B&H (Corn Daily MC RF L/S) | **+57.4 pp** |
| Hurdle Stage 2b r (Corn Daily negative leg) | **+0.371** (p<0.0001) |
| Walkforward viability (Coffee) | 3/3 years ✓ |
| Leakage check (center=True rolling) | +0.248 AUC inflation → confirmed clean |

---

## Architecture

```
Market (KC=F, ZC=F)    →  ACU filter + RSI/BB/MACD          ─┐
Weather (Open-Meteo)   →  MIQR + ffill + rolling (center=F)  ─┤
Macro  (CPI, VIX, FX)  →  CPI lag fix + VIX resample         ─┼─→ Integration → Null Importances → Tensor → Models
Farming (Synthetic)    →  Binary flags + sin/cos encode       ─┤
Farming (LLM USDA/PSD) →  Claude API → structured JSON (04b) ─┘
```

**Models:** LightGBM · Random Forest · LSTM Hybrid · TCN Hybrid · Stacking Ensemble · Two-Stage Hurdle

---

## Quick Start

```bash
# 1. Clone and install
git clone https://github.com/ZaKhimNe/DS108_Commodity-price-forecasting.git
cd DS108_Commodity-price-forecasting
pip install -r requirements.txt

# 2. Data ingestion (run in order)
python src/ingestion/01_market_ingestion.py
python src/ingestion/02_weather_ingestion.py          # ~10 min, rate-limited
python src/ingestion/03_macro_ingestion.py
python src/ingestion/04_farming_ingestion.py          # synthetic calendar
python src/ingestion/04b_llm_farming_ingestion.py     # LLM calendar (Claude API, optional)

# 3. Preprocessing
python src/preprocessing/05_weather_anomaly_removal.py
python src/preprocessing/06_weather_preprocessing.py
python src/preprocessing/07_weather_weekly_aggregation.py
python src/preprocessing/08_market_acu_filter.py
python src/preprocessing/09_macro_preprocessing.py
python src/preprocessing/10_farming_preprocessing.py

# 4. Integration & Feature Engineering
python src/11_data_integration.py
python src/12_lag_analysis.py

# 5. Tensor Packing
python src/13_tensor_packing.py           # binary
python src/13b_tensor_packing_mc.py       # multiclass

# 6. Modeling — Binary
python src/modeling/14_lgbm_baseline.py
python src/modeling/15_rf_baseline.py
python src/modeling/16_lstm_hybrid.py
python src/modeling/17_tcn_hybrid.py
python src/modeling/18_stacking_ensemble.py

# 6b. Modeling — Multiclass
python src/modeling/14b_lgbm_multiclass.py
python src/modeling/15b_rf_multiclass.py
python src/modeling/16b_lstm_multiclass.py
python src/modeling/17b_tcn_multiclass.py
python src/modeling/18b_stacking_multiclass.py

# 7. Evaluation
python src/modeling/19_backtesting_engine.py
python src/modeling/19b_backtesting_mc.py
python src/modeling/21_walkforward_eval.py
python src/modeling/22_hurdle_model.py
python src/20_pipeline_report.py
```

> **Skip ingestion:** Download processed dataset directly from [Kaggle](https://www.kaggle.com/datasets/khimtagia/agricommodity-futures-multi-source-ml-dataset) and place CSVs in `data/integrated/`.

---

## Project Structure

```
DS108_Commodity-price-forecasting/
├── src/
│   ├── ingestion/
│   │   ├── 01_market_ingestion.py
│   │   ├── 02_weather_ingestion.py
│   │   ├── 03_macro_ingestion.py
│   │   ├── 04_farming_ingestion.py
│   │   └── 04b_llm_farming_ingestion.py   # Claude API: USDA + PSD → JSON
│   ├── preprocessing/
│   │   ├── 05_weather_anomaly_removal.py  # MIQR + flatline detection
│   │   ├── 06_weather_preprocessing.py   # ffill + rolling (center=False)
│   │   ├── 07_weather_weekly_aggregation.py
│   │   ├── 08_market_acu_filter.py        # ACU + RSI/BB/MACD
│   │   ├── 09_macro_preprocessing.py      # CPI lag fix + VIX
│   │   └── 10_farming_preprocessing.py
│   ├── 11_data_integration.py
│   ├── 12_lag_analysis.py                 # CCF bootstrap validation
│   ├── 13_tensor_packing.py               # binary: 70/10/20 + embargo
│   ├── 13b_tensor_packing_mc.py           # multiclass tensors
│   └── modeling/
│       ├── 14_lgbm_baseline.py
│       ├── 14b_lgbm_multiclass.py
│       ├── 15_rf_baseline.py
│       ├── 15b_rf_multiclass.py
│       ├── 16_lstm_hybrid.py              # BiLSTM(128→64) + static concat
│       ├── 16b_lstm_multiclass.py
│       ├── 17_tcn_hybrid.py               # TCN 4 dilated blocks RF=31
│       ├── 17b_tcn_multiclass.py
│       ├── 18_stacking_ensemble.py        # LR meta OOF — binary
│       ├── 18b_stacking_multiclass.py     # LR meta OOF — multiclass
│       ├── 19_backtesting_engine.py
│       ├── 19b_backtesting_mc.py
│       ├── 21_walkforward_eval.py
│       └── 22_hurdle_model.py             # Two-stage hurdle (full)
├── src/20_pipeline_report.py
├── config/
│   └── coordinates.json                   # Weather API coordinates
├── figures/
│   ├── fig1_pipeline.pdf/png
│   ├── fig2_ccf.pdf/png
│   ├── fig3_split.pdf/png
│   ├── fig4_equity.pdf/png
│   ├── fig5_hurdle.pdf/png
│   └── fig6_ablation.pdf/png              # Ablation bar chart
├── models/ablation/
│   ├── 00_baseline/
│   ├── 01_global_scaler/
│   ├── 02_no_embargo/
│   └── 03_center_rolling/
├── data/
│   └── .gitkeep                           # empty — real data on Kaggle
├── models/
│   └── .gitkeep                           # empty — generated by scripts
├── PIPELINE_REPORT.md
├── requirements.txt
├── CLAUDE.md
└── README.md
```

---

## Data Leakage Prevention

All preprocessing modules are verified leakage-free (ablation experiment 03 confirms `center=True` rolling inflates AUC by +0.248):

| Module | Operation | Leak? |
|--------|-----------|-------|
| MIQR | Rolling Q1/Q3 (`center=False`) | No |
| Imputation | `ffill()` only | No |
| CPI | +DateOffset(1mo+12d) before ffill | No |
| Scaler | `MinMaxScaler.fit(train_only)` | No |
| Split | 70/10/20 + embargo gap (7 rows) | No |

---

## LLM Calendar Features (Bonus 2)

Module `04b_llm_farming_ingestion.py` uses **Claude Haiku 4.5** to extract structured crop calendar features from USDA crop progress reports (corn) and USDA PSD production database (coffee) via few-shot prompting with JSON schema validation.

| Dataset | Synthetic AUC | LLM AUC | Improvement |
|---------|-------------:|--------:|------------:|
| Coffee Weekly | 0.404 | **0.491** | **+8.7 pp** |
| Corn Daily | 0.475 | **0.491** | +1.6 pp |
| Coffee Daily | 0.405 | 0.377 | — (hybrid wins: 0.412) |
| Corn Weekly | **0.598** | 0.548 | synthetic wins |

---

## Two-Stage Hurdle Model

Addresses zero-inflation in return distribution (~34–52% flat zone):

```
E[return] = P(return > θ) × E[return | return > θ]
          - P(return < -θ) × E[|return| | return < -θ]
```

**Corn Daily results:** Stage 2b r=**+0.371** (p<0.0001, n=164) — driven by `cal_cos_week` (harvest seasonality), CPI stress, and USD realized volatility. Down-moves are more predictable than up-moves, consistent with supply-side shock asymmetry.

---

## Dataset

Processed dataset available on Kaggle:  
**[DS108 AgriCommodity Futures — Multi-Source ML Dataset](https://www.kaggle.com/datasets/khimtagia/agricommodity-futures-multi-source-ml-dataset)**

4 files · ~11.5 MB · CC BY 4.0 · Usability 8.82

---

## Paper

*Hệ Thống Tiền Xử Lý Dữ Liệu Đa Nguồn và Dự Báo Biến Động Giá Hàng Hóa Nông Nghiệp*  
DS108 Research Group · 2026  
Full results: [PIPELINE_REPORT.md](PIPELINE_REPORT.md)

---

## License

MIT License — see [LICENSE](LICENSE)
