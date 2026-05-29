# DS108 AgriCommodity Futures — Multi-Source ML Dataset

**Version:** 1.0 · **License:** CC BY 4.0 · **Ingestion:** 2010–2026 · **Integrated data:** 2011-02-14 → 2025-12-19  
**Paper:** *Hệ Thống Tiền Xử Lý Dữ Liệu Đa Nguồn và Dự Báo Biến Động Giá Hàng Hóa Nông Nghiệp*

---

## Overview

This dataset provides **production-ready, leakage-free feature tensors** for agricultural commodity futures price direction forecasting. It integrates four heterogeneous data sources — financial market data, micro-climate weather, macroeconomic indicators, and synthetic crop calendars — into four machine-learning-ready CSV files.

The dataset was built as part of the DS108 research project with a focus on **causal preprocessing**: every transformation (rolling statistics, imputation, scaling parameters) uses only past data at each time step, with no look-ahead bias.

---

## Files

| File | Rows | Features | Commodity | Frequency | Period |
|------|-----:|--------:|-----------|-----------|--------|
| `integrated_coffee_daily.csv` | 2,832 | 99 | Coffee (KC=F) | Daily | 2011-02-14 → 2025-12-19 |
| `integrated_coffee_weekly.csv` | 724 | 81 | Coffee (KC=F) | Weekly (W-MON) | 2011-02-14 → 2025-12-29 |
| `integrated_corn_daily.csv` | 3,654 | 101 | Corn (ZC=F) | Daily | 2011-02-14 → 2025-12-19 |
| `integrated_corn_weekly.csv` | 700 | 83 | Corn (ZC=F) | Weekly (W-MON) | 2011-02-14 → 2025-12-29 |

---

## Target Variables

Each file contains four parallel label formats derived from `return_future = Close[t+7]/Close[t] − 1`:

| Column | Type | Description |
|--------|------|-------------|
| `target_binary` | 0/1 | Return > 2.5% threshold (base rate ~30–35%) |
| `target_soft` | [0,1] | Sigmoid-smoothed binary label |
| `target_reg` | float | Clipped return ∈ [−0.30, +0.30] |
| `target_multiclass` | 0/1/2 | Down / Flat / Up |

---

## Feature Groups

### 1. Market Features (`Close`, `Volume`, `RSI_14`, `BB_upper`, `MACD`, ...)
Price and volume data with technical indicators computed on `currency_adjusted_close = Close × USD/BRL_rate`.

### 2. Weather Features (`temp_max_rolling_7d`, `precip_cumsum_30d`, `et0_cumsum_30d`, ...)
Five variables (T_max, T_min, Precip, ET0, VPD) from 5 geographic locations per crop, preprocessed with causal MIQR and forward-fill.

- **Coffee:** Cerrado Baiano, Cerrado Mineiro, Matas de Minas, Mogiana, Sul de Minas (Brazil)
- **Corn:** Illinois, Indiana, Iowa, Minnesota, Nebraska (USA)

Biological lag features: `temp_bio_lag`, `precip_bio_lag` — shifted by 34 weeks (coffee) and 9 weeks (corn) based on CCF bootstrap validation.

### 3. Macroeconomic Features (`inf_CPI_MoM_pct`, `usd_log_return`, `vix_close`, ...)
- USD/BRL exchange rate (currency risk)
- US CPI with +1 month +12 day publication lag correction
- CBOE VIX (global risk sentiment)
- `inflation_pressure = usd_log_return_lag_1w × 100 − inf_CPI_MoM_pct`

### 4. Crop Calendar Features (`cal_sin_week`, `cal_cos_week`, `cal_is_harvest`, ...)
Synthetic binary flags for planting, pollination/flowering, and harvest stages with cyclical sin/cos encoding.

**LLM-enhanced variant (module 04b):** Structured features extracted from USDA Weekly Crop Progress reports (corn) and USDA PSD annual production database (coffee) via Claude Haiku 4.5 few-shot prompting. Replaces binary flags with continuous values (`corn_planting_pct`, `corn_condition_ge_pct`, `coffee_production_change_pct`, etc.). Coffee Weekly AUC improves +8.7 pp over synthetic baseline.

---

## Data Pipeline Summary

```
Market (yfinance)      →  ACU filter + RSI/BB/MACD          ─┐
Weather (Open-Meteo)   →  MIQR + ffill + EWM                 ─┤
Macro (BLS + yfinance) →  CPI lag fix + VIX                  ─┼─→ Integration → Null Importances → CSV
Farming (synthetic)    →  Binary flags + sin/cos              ─┤
Farming (LLM USDA/PSD) →  Claude API → structured JSON (04b) ─┘
```

**Zero data leakage guarantee:**
- All rolling operations: `center=False` (causal)
- Imputation: `ffill` only (forward-fill, never backward)
- CPI: shifted +1 month +12 days to respect BLS publication lag
- Scaler: `MinMaxScaler` fit on train split only
- Embargo gap: 7 rows removed at each split boundary

---

## Recommended Train/Val/Test Split

Use **chronological split only** — never random K-fold for time series.

```python
n = len(df)
val_start  = int(n * 0.70)   # ~2010–2022 train
test_start = int(n * 0.80)   # ~2022–2024 val
# 2024–2026: test
horizon = 7  # daily; horizon = 1 for weekly

train = df.iloc[:val_start - horizon]
val   = df.iloc[val_start:test_start - horizon]
test  = df.iloc[test_start:]
```

---

## Known Biases and Limitations

### Geographic Bias
Only US corn and Brazilian coffee are covered. Major producers excluded:
- Coffee: Vietnam (world #2), Colombia, Indonesia
- Corn: China, EU, Argentina

### Temporal Bias
Test period (2022–2025) coincides with extreme events:
- Coffee bull run 2023–2024: KC=F +203.8% (Brazil drought)
- Post-COVID corn supply chain normalization

Model performance metrics on this test period may not generalize to other market regimes.

### Synthetic Crop Calendar
`is_planting`, `is_harvest`, `is_flowering` are rule-based approximations from average historical crop cycles. They do not reflect year-to-year variation due to climate change or farmer decisions. Module `04b_llm_farming_ingestion.py` partially addresses this by extracting USDA weekly progress data via LLM, improving Coffee Weekly AUC by +8.7pp.

### US-Centric Macro Indicators
- **CPI:** US consumer price index, not Brazilian producer costs
- **VIX:** CBOE measure of US market fear, not global

### Small Weekly Val Sets
Weekly datasets have only 24–33 positive-class validation samples, making reliable hyperparameter tuning difficult.

---

## Ethical Considerations

1. **Not financial advice.** Models trained on this dataset should not be deployed for live trading without additional risk management and longer paper-trading validation (minimum 12 months post-training).

2. **No personal data.** All source data comes from public APIs (yfinance, Open-Meteo, BLS). No personally identifiable information is included.

3. **Honest performance reporting.** The Binary Coffee Daily Stack Sharpe ratio (2.154) is partly inflated by the 2023–2024 bull run (KC=F +203.8%); alpha vs B&H is −8pp. Corn Daily MC RF Long/Short alpha (+57.4pp) against a negative Buy-and-Hold (−26.7%) is the more representative metric for strategy evaluation.

---

## Data Sources

| Source | Provider | License |
|--------|----------|---------|
| Coffee KC=F, Corn ZC=F futures | Yahoo Finance (via yfinance) | Public |
| Weather (T, Precip, ET0, VPD) | Open-Meteo Archive API | CC BY 4.0 |
| US CPI (CUUR0000SA0) | U.S. Bureau of Labor Statistics | Public Domain |
| VIX (^VIX) | Yahoo Finance (via yfinance) | Public |
| Crop calendar | Synthetic (rule-based) | N/A |
| USDA Crop Progress (corn) | USDA NASS QuickStats API | Public Domain |
| USDA PSD (coffee production) | USDA Foreign Agricultural Service | Public Domain |

---

## Citation

If you use this dataset, please cite:

```bibtex
@dataset{ds108_agri_commodity_2026,
  title     = {DS108 AgriCommodity Futures — Multi-Source ML Dataset},
  author    = {DS108 Research Group},
  year      = {2026},
  publisher = {Kaggle},
  url       = {https://www.kaggle.com/datasets/khimtagia/agricommodity-futures-multi-source-ml-dataset}
}
```
---

*DS108 AgriCommodity Dataset · v1.0 · 2026 · CC BY 4.0*
