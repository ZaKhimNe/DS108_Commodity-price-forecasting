# Báo Cáo Kết Quả Pipeline — DS108 Scraped Data
**Ngày chạy:** 2026-05-24  
**Dữ liệu:** Coffee (KC=F) & Corn (ZC=F) futures — phân loại nhị phân chiều hướng giá (>5% trong 7 ngày/1 tuần tới)

---

## 1. Tổng Quan Pipeline

Pipeline gồm 6 giai đoạn tuần tự:

| Giai đoạn | Mô tả | Trạng thái |
|---|---|---|
| 1 – Ingestion | Thu thập dữ liệu thị trường, thời tiết, vĩ mô, canh tác | ✅ Hoàn tất |
| 2 – Preprocessing | Lọc nhiễu, chuẩn hóa, resample W-MON | ✅ Hoàn tất |
| 3 – Integration | Gộp 4 nguồn → 4 file tích hợp | ✅ Hoàn tất |
| 4 – Tensor Packing | Sliding-window, split 70/10/20 + embargo, feature selection | ✅ Hoàn tất |
| 5a – LightGBM Baseline | Tầng 1: tabular baseline | ✅ Hoàn tất |
| 5b/c – LSTM & TCN | Tầng 2: sequence models | ✅ Hoàn tất |
| 6 – Stacking Ensemble | Tầng 3: LR meta-learner (LGB + LSTM + TCN) | ✅ Hoàn tất |

---

## 2. Dữ Liệu & Phân Chia

### Phân phối mẫu (sau embargo gaps)

| Dataset | Train | Val | Test | Base rate (Train) | Base rate (Test) |
|---|---|---|---|---|---|
| Coffee Daily | 672 | 90 | 194 | 21.1% | 23.7% |
| Coffee Weekly | 130 | 18 | 38 | 14.6% | 15.8% |
| Corn Daily | 670 | 90 | 194 | 9.0% | 7.7% |
| Corn Weekly | 129 | 17 | 38 | 9.3% | **0.0%** ⚠️ |

> **Ghi chú Corn Weekly:** Toàn bộ 12 nhãn dương tập trung trong 70% đầu (train). Val và test không có positive → không thể đánh giá mô hình có ý nghĩa.

### Feature Selection (LightGBM null-importance + TimeSeriesSplit trên train)

| Dataset | Tổng features | Được chọn | Dynamic | Static |
|---|---|---|---|---|
| Coffee Daily | 82 | 31 | 30 | 1 |
| Coffee Weekly | 64 | 16 | 15 | 1 |
| Corn Daily | 84 | 26 | 24 | 2 |
| Corn Weekly | 66 | 9 | 9 | 0 |

---

## 3. Kết Quả Mô Hình

### 3.1 LightGBM Baseline (Tầng 1)

| Dataset | Iter* | Train AUC | Val AUC | **Test AUC** | Test PR-AUC | Test F1 |
|---|---|---|---|---|---|---|
| Coffee Daily | 3 | 0.975 | 0.704 | **0.564** | 0.276 | 0.387 |
| Coffee Weekly | 1 | 0.965 | 0.708 | **0.357** | 0.144 | 0.095 |
| Corn Daily | 6 | 0.995 | 0.571 | **0.443** | 0.081 | 0.118 |
| Corn Weekly | 1 | 0.975 | n/a | **n/a** | n/a | 0.000 |

*Best iteration — LightGBM dừng rất sớm (3–6 iter), cho thấy overfitting cực nặng.  
*scale_pos_weight: Coffee Daily=3.73 | Coffee Weekly=5.84 | Corn Daily=10.0 (capped) | Corn Weekly=9.75*

**Top 5 features quan trọng nhất:**

| Coffee Daily | Corn Daily |
|---|---|
| inf_US_CPI (gain=276) | inflation_pressure (gain=847) |
| temp_max_cumsum_30d (256) | usd_RSI_14 (839) |
| currency_adjusted_close (190) | volatility (728) |
| MACD (139) | usd_volatility_20d (403) |
| RSI_14 (120) | precip_wavelet_trend (376) |

---

### 3.2 LSTM Hybrid (Tầng 2)

| Dataset | Window | n_dyn | n_stat | Val AUC | **Test AUC** | Test PR-AUC | Test F1 |
|---|---|---|---|---|---|---|---|
| Coffee Daily | 45 | 30 | 1 | 0.618 | **0.709** | 0.449 | 0.512 |
| Coffee Weekly | 4 | 15 | 1 | 0.846 | **0.667** | 0.398 | 0.313 |
| Corn Daily | 45 | 24 | 2 | 0.886 | **0.601** | 0.080 | 0.089 |
| Corn Weekly | 4 | 9 | 0 | 0.500 | **n/a** | n/a | 0.000 |

*Cấu hình: BiLSTM(128→64) + head(64), dropout=0.3, lr=1e-3, patience=15, max_epochs=120*

---

### 3.3 TCN Hybrid (Tầng 2 alt)

| Dataset | Window | RF | Blocks | Val AUC | **Test AUC** | Test PR-AUC | Test F1 | Precision |
|---|---|---|---|---|---|---|---|---|
| Coffee Daily | 14 | 31 | 4 | 0.795 | **0.699** | 0.495 | 0.494 | **0.613** |
| Coffee Weekly | 8 | 31 | 4 | 1.000† | **0.331** | 0.308 | 0.242 | 0.143 |
| Corn Daily | 30 | 31 | 4 | 0.800 | **0.662** | 0.231 | 0.088 | 0.046 |
| Corn Weekly | 4 | 31 | 4 | 0.500 | **n/a** | n/a | 0.000 | — |

*† Val AUC=1.000 do val set chỉ có 11 rows (1 positive) — không đáng tin cậy.*  
*Cấu hình: n_filters=64, kernel=3, 4 blocks, RF=31, dropout=0.2, lr=5e-4*

---

### 3.4 Stacking Ensemble (Tầng 3)

Meta-learner: Logistic Regression (`class_weight='balanced'`), split 50/50 trên test set của tầng 2.

| Dataset | n_meta | Stack AUC | **ΔvsBase** | Stack F1 | Hệ số tốt nhất |
|---|---|---|---|---|---|
| Coffee Daily | 75+75 | **0.723** | −0.041 | 0.000† | TCN=1.770, LSTM=0.101, LGB=−0.159 |
| Coffee Weekly | 15+16 | **0.933** | −0.067 | 0.222 | LSTM=0.034, LGB=−0.014, TCN=−0.073 |
| Corn Daily | 75+75 | **0.646** | −0.014 | 0.000† | TCN=3.401, LSTM=0.388, LGB=0.254 |
| Corn Weekly | — | SKIP | — | — | 0 positive trong meta-train |

*† F1=0 do ngưỡng được tối ưu trên meta-train (0.52/0.62) quá cao so với meta-test → cần OOF stacking.*

**Tương quan base learners (coffee daily):**  
LGB↔LSTM: r=0.47 | LGB↔TCN: r=0.27 | **LSTM↔TCN: r=0.85** ← quá cao, giảm lợi ích ensemble

---

## 4. So Sánh Tổng Hợp — Test AUC-ROC

| Dataset | LightGBM | LSTM | TCN | Stack | **Winner** |
|---|---|---|---|---|---|
| Coffee Daily | 0.564 | 0.709 | 0.699 | 0.723 | Stack / LSTM |
| Coffee Weekly | 0.357 | 0.667 | 0.331 | 0.933‡ | LSTM |
| Corn Daily | 0.443 | 0.601 | **0.662** | 0.646 | TCN |
| Corn Weekly | n/a | n/a | n/a | n/a | — |

‡ Coffee Weekly stack AUC=0.933 trên meta-test 16 rows (6% base rate, 1 positive) — quá nhỏ để kết luận.

**Nhận xét chính:**
- LSTM và TCN vượt LightGBM đáng kể trên Coffee Daily (+14.5% AUC)
- TCN là model mạnh nhất cho Corn Daily (AUC=0.662, PR-AUC=0.231)
- LightGBM bị overfitting nghiêm trọng (best_iter=3–6, gap train–val >0.27)
- Stacking không cải thiện AUC do (1) LSTM↔TCN correlation r=0.85, (2) meta-train quá nhỏ

---

## 5. Vấn Đề & Giới Hạn

### 5.1 Overfitting LightGBM
- Train AUC 0.975–0.995 vs Test AUC 0.357–0.564
- Best iteration 1–6 (hits early stopping ngay lập tức)
- **Nguyên nhân:** Quá ít dữ liệu (~672 train rows daily, ~130 weekly) so với số features (64–84)
- **Đề xuất:** Tăng `reg_alpha`/`reg_lambda`, giảm `num_leaves` xuống 7–15, thêm `min_child_weight`

### 5.2 Corn Weekly — Không thể đánh giá
- 186 rows tổng, chỉ 12 positives (6.4%), phân bố trong 4 năm đầu
- Val (17 rows) và Test (38 rows) đều có 0 positives
- AUC, PR-AUC = NaN; Stack bị skip
- **Nguyên nhân:** Target `return > 5%` trong 1 tuần quá nghiêm ngặt với corn (~1.5% weekly vol)
- **Đề xuất:** Giảm ngưỡng xuống 3% hoặc dùng regression thay classification

### 5.3 Stacking F1 = 0 trên Coffee/Corn Daily
- Ngưỡng được tối ưu trên meta-train (0.52 và 0.62) nhưng distribution khác meta-test
- **Đề xuất:** Dùng Out-of-Fold (OOF) stacking thay split-half để tránh threshold mismatch

### 5.4 TCN val AUC = 1.0 (Coffee Weekly win=8)
- Val chỉ có 11 rows, 1 positive → AUC=1.0 không có ý nghĩa thống kê
- Test AUC=0.331 phản ánh thực chất: model không generalize

### 5.5 LSTM↔TCN correlation r=0.851 (Coffee Daily)
- Cả hai đều là sequence models trên cùng features → predictions gần giống nhau
- Meta-learner LR không thể học được trọng số khác biệt → ΔAUC âm
- **Đề xuất:** Thêm base learner độc lập hơn (e.g., random forest tabular hoặc gradient boosting trên lag features)

---

## 6. Đề Xuất Cải Thiện

| Ưu tiên | Vấn đề | Giải pháp |
|---|---|---|
| 🔴 Cao | LightGBM overfit | `num_leaves=7`, `reg_lambda=5.0`, `min_child_samples=30` |
| 🔴 Cao | Stacking threshold mismatch | Dùng OOF cross-validation cho meta features |
| 🟠 Trung | Corn weekly không có positives | Giảm target threshold xuống 3%, hoặc dùng regression |
| 🟠 Trung | LSTM↔TCN correlation cao | Thêm RF hoặc XGBoost làm base learner thứ 3 độc lập |
| 🟡 Thấp | Weekly dataset quá nhỏ | Thu thập thêm dữ liệu (extend về 2010–) |
| 🟡 Thấp | TCN RF >> window size | Giảm `n_blocks` xuống 2 cho weekly datasets |

---

## 7. File Output

```
models/
  lgbm_baseline/
    lgbm_{tag}.joblib               ×4 tags
    test_predictions_{tag}.csv      ×4 (cột: Date, y_true, y_prob_lgb, y_pred_lgb)
    results_{tag}.json              ×4
    feature_cols_{tag}.json         ×4
    importance_{tag}.csv            ×4

  lstm_hybrid/
    lstm_{tag}_win{W}.pt            ×4 (win45/4/45/4)
    test_predictions_{tag}.csv      ×4 (cột: y_true, y_prob_lstm, y_pred_lstm)
    results_{tag}.json              ×4

  tcn_hybrid/
    tcn_{tag}_win{W}.pt             ×4 (win14/8/30/4)
    test_predictions_{tag}.csv      ×4 (cột: y_true, y_prob_tcn, y_pred_tcn)
    results_{tag}.json              ×4

  stacking_ensemble/
    meta_lr_{tag}.joblib            ×3 (corn_weekly skipped)
    test_predictions_{tag}.csv      ×3 (cột: Date, y_prob_lgb/lstm/tcn, y_true, y_prob_stack, y_pred_stack)
    results_{tag}.json              ×3
```

---

*Report được tạo tự động sau khi chạy xong toàn bộ 6 stages — 2026-05-24*
