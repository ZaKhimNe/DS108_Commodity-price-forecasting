# DS108 — Kế Hoạch Module 04b: LLM-Enhanced Farming Calendar
**Bonus 2: Generative AI / LLMs (+0.5đ → +1.0đ)**  
**Phiên bản:** 1.0 · 2026-05-29  
**Mục tiêu:** Thay thế synthetic binary flags trong `04_farming_ingestion.py` bằng structured JSON được extract tự động từ báo cáo nông nghiệp thực tế (USDA + CONAB) thông qua Claude API.

---

## 1. Bối Cảnh & Motivation

### 1.1 Vấn Đề Của Synthetic Calendar (04)

Module `04_farming_ingestion.py` hiện tại gán nhãn các giai đoạn sinh trưởng theo **quy tắc cứng dựa trên tháng**:

```python
# Hiện tại — quá đơn giản
is_planting  = month.isin([4, 5])          # Iowa corn: April–May
is_flowering = month.isin([6, 7])          # Coffee Brazil: June–July
is_harvest   = month.isin([9, 10, 11])     # Corn: Sep–Nov
```

**Nhược điểm:**
- Không phản ánh biến động năm theo năm (năm 2012 El Niño harvest trễ 3 tuần)
- Không có thông tin về **mức độ** (planting 30% vs 90% là khác nhau hoàn toàn)
- Không có **condition signal** (good/fair/poor crop conditions)
- Không phân biệt theo **vùng địa lý** (Iowa vs Nebraska có thể lệch 2 tuần)

### 1.2 Giải Pháp: LLM-Extracted Structured Calendar

USDA và CONAB publish báo cáo text hàng tuần/tháng chứa chính xác các thông tin này. LLM sẽ bóc tách chúng thành JSON có cấu trúc — đây là use case "Prompt Engineering chuẩn hóa dữ liệu phi cấu trúc" mà Bonus 2 yêu cầu.

**Ví dụ thực tế từ USDA (2024-05-13):**

> *"Corn: Nationally, 49 percent of the corn crop had been planted, compared to 36 percent last week, 62 percent last year, and a 5-year average of 56 percent. In Iowa, planting was 45 percent complete, behind the 5-year average of 67 percent due to wet conditions in western counties."*

→ LLM extract thành:
```json
{
  "report_date": "2024-05-13",
  "national_planting_pct": 49,
  "vs_last_year": -13,
  "vs_5yr_avg": -7,
  "iowa_planting_pct": 45,
  "iowa_vs_avg": -22,
  "delay_reason": "wet_conditions",
  "signal": "delayed_bearish"
}
```

---

## 2. Nguồn Dữ Liệu

### 2.1 USDA Weekly Crop Progress (Corn)

| Thông tin | Chi tiết |
|-----------|---------|
| **Nhà cung cấp** | USDA National Agricultural Statistics Service (NASS) |
| **URL chính** | https://usda.library.cornell.edu/concern/publications/8336h188j |
| **URL API** | https://quickstats.nass.usda.gov/api |
| **Tần suất** | Weekly, thứ Hai 3:00 PM ET |
| **Period có data** | 1981 → nay |
| **Format** | CSV qua API, PDF qua Cornell library |
| **License** | Public domain (US Government) |
| **Nội dung** | Planting %, Emerged %, Silking %, Dough %, Mature %, Harvested %, Condition (G/E/F/P/VP) |

**USDA QuickStats API call:**
```
GET https://quickstats.nass.usda.gov/api/api_GET/
    ?key={API_KEY}
    &commodity_desc=CORN
    &statisticcat_desc=PROGRESS
    &freq_desc=WEEKLY
    &year__GE=2010
    &state_name=IOWA
    &format=JSON
```

**5 bang cần query:** Illinois, Indiana, Iowa, Minnesota, Nebraska

### 2.2 CONAB Boletim Agropecuário (Coffee)

| Thông tin | Chi tiết |
|-----------|---------|
| **Nhà cung cấp** | Companhia Nacional de Abastecimento (CONAB), Brazil |
| **URL** | https://www.conab.gov.br/info-agro/safras/cafe |
| **Tần suất** | Monthly (tháng 1, 4, 7, 10) |
| **Period có data** | 2000 → nay |
| **Format** | PDF |
| **License** | Public (Brazilian government) |
| **Nội dung** | Flowering stage, harvest progress %, production forecast, regional condition |

**5 vùng cần extract:** Cerrado Baiano, Cerrado Mineiro, Matas de Minas, Mogiana, Sul de Minas

### 2.3 So Sánh Hai Nguồn

| Tiêu chí | USDA (Corn) | CONAB (Coffee) |
|----------|------------|----------------|
| Tần suất | Weekly ✅ | Monthly ⚠️ |
| API có sẵn | ✅ QuickStats | ❌ Chỉ PDF |
| Cần LLM parsing | Một phần | Hoàn toàn |
| Coverage 2010– | ✅ | ✅ |
| Ngôn ngữ | English | Portuguese |

---

## 3. Kiến Trúc Module 04b

```
┌─────────────────────────────────────────────────────────────────┐
│                    04b_llm_farming_ingestion.py                  │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│  DataFetcher                    LLMExtractor                      │
│  ┌─────────────┐                ┌───────────────────────────┐     │
│  │ USDA API    │─── CSV text ──▶│ CornPromptBuilder         │     │
│  │ QuickStats  │                │ (few-shot + schema)       │     │
│  └─────────────┘                └───────────────┬───────────┘     │
│                                                 │                 │
│  ┌─────────────┐                                │ Claude API      │
│  │ CONAB PDF   │─── text ──────▶ CoffeePrompt  │                 │
│  │ pdfplumber  │                 Builder        ▼                 │
│  └─────────────┘                ┌───────────────────────────┐     │
│                                 │ JSON Validator            │     │
│                                 │ (Pydantic schema)         │     │
│                                 └───────────────┬───────────┘     │
│                                                 │                 │
│                                 ┌───────────────▼───────────┐     │
│                                 │ TimeSeriesBuilder         │     │
│                                 │ - daily index             │     │
│                                 │ - ffill gaps              │     │
│                                 │ - merge with synthetic    │     │
│                                 └───────────────────────────┘     │
│                                                                   │
│  Output: data/raw/farming/llm_corn_calendar.csv                   │
│          data/raw/farming/llm_coffee_calendar.csv                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 4. Prompt Design — Chi Tiết

### 4.1 Nguyên Tắc Thiết Kế

1. **Schema-first:** Định nghĩa output JSON schema trước, viết prompt xung quanh schema
2. **Few-shot examples:** 3 ví dụ thật (easy + medium + edge case)
3. **Constraint injection:** Nói rõ giới hạn giá trị (0–100 cho %, enum cho signal)
4. **Fallback handling:** Nếu không tìm thấy giá trị → `null`, không hallucinate
5. **Language handling:** CONAB prompt cần xử lý Portuguese

---

### 4.2 CORN PROMPT — USDA Weekly Crop Progress

#### System Prompt
```
You are a precision agricultural data extractor specializing in USDA Crop Progress reports.

Your task is to extract structured planting and crop condition data from USDA Weekly 
Crop Progress report text and return ONLY a valid JSON object matching the exact schema 
provided. 

CRITICAL RULES:
1. Return ONLY the JSON object — no explanation, no markdown, no preamble
2. Use null for any value not explicitly mentioned in the text
3. Never hallucinate or infer values not present in the text
4. All percentages must be integers 0–100
5. signal must be one of: "ahead_bullish", "on_track", "delayed_bearish", 
   "very_delayed_very_bearish", "condition_deteriorating", "condition_improving"
6. If the report covers multiple crops, extract ONLY corn data
```

#### JSON Schema (Corn)
```json
{
  "report_date": "YYYY-MM-DD",
  "week_ending": "YYYY-MM-DD",
  "national": {
    "planting_pct": null,
    "emerged_pct": null,
    "silking_pct": null,
    "dough_pct": null,
    "dented_pct": null,
    "mature_pct": null,
    "harvested_pct": null,
    "vs_last_year_planting": null,
    "vs_5yr_avg_planting": null,
    "condition_good_excellent_pct": null,
    "condition_poor_very_poor_pct": null
  },
  "states": {
    "iowa": {
      "planting_pct": null,
      "vs_5yr_avg": null,
      "condition_good_excellent_pct": null
    },
    "illinois": { "planting_pct": null, "vs_5yr_avg": null },
    "indiana":  { "planting_pct": null, "vs_5yr_avg": null },
    "minnesota":{ "planting_pct": null, "vs_5yr_avg": null },
    "nebraska": { "planting_pct": null, "vs_5yr_avg": null }
  },
  "notable_events": [],
  "signal": "on_track",
  "signal_reasoning": "Brief explanation in English (max 20 words)"
}
```

#### Few-Shot Example 1 — Normal planting
```
INPUT:
"For the week ending May 13, 2024, national corn planting progress 
stood at 49 percent, ahead of 36 percent last week but behind 
62 percent a year ago and the 5-year average of 56 percent. 
In Iowa, 45 percent of the corn crop had been planted compared 
to the 5-year average of 67 percent. Crop condition rated 
72 percent good to excellent nationally."

OUTPUT:
{
  "report_date": "2024-05-13",
  "week_ending": "2024-05-13",
  "national": {
    "planting_pct": 49,
    "emerged_pct": null,
    "silking_pct": null,
    "dough_pct": null,
    "dented_pct": null,
    "mature_pct": null,
    "harvested_pct": null,
    "vs_last_year_planting": -13,
    "vs_5yr_avg_planting": -7,
    "condition_good_excellent_pct": 72,
    "condition_poor_very_poor_pct": null
  },
  "states": {
    "iowa": { "planting_pct": 45, "vs_5yr_avg": -22, "condition_good_excellent_pct": null },
    "illinois": { "planting_pct": null, "vs_5yr_avg": null },
    "indiana":  { "planting_pct": null, "vs_5yr_avg": null },
    "minnesota":{ "planting_pct": null, "vs_5yr_avg": null },
    "nebraska": { "planting_pct": null, "vs_5yr_avg": null }
  },
  "notable_events": ["wet conditions in western Iowa"],
  "signal": "delayed_bearish",
  "signal_reasoning": "National planting 7pp behind 5yr avg; Iowa 22pp behind average"
}
```

#### Few-Shot Example 2 — Harvest season
```
INPUT:
"For the week ending October 7, 2019, corn harvest progress 
reached 22 percent complete, ahead of 15 percent last year 
and equal to the 5-year average of 22 percent. Crop condition 
was 57 percent good to excellent, 11 percent poor to very poor."

OUTPUT:
{
  "report_date": "2019-10-07",
  "week_ending": "2019-10-07",
  "national": {
    "planting_pct": null,
    "emerged_pct": null,
    "silking_pct": null,
    "dough_pct": null,
    "dented_pct": null,
    "mature_pct": null,
    "harvested_pct": 22,
    "vs_last_year_planting": null,
    "vs_5yr_avg_planting": null,
    "condition_good_excellent_pct": 57,
    "condition_poor_very_poor_pct": 11
  },
  "states": {
    "iowa": { "planting_pct": null, "vs_5yr_avg": null, "condition_good_excellent_pct": null },
    "illinois": { "planting_pct": null, "vs_5yr_avg": null },
    "indiana":  { "planting_pct": null, "vs_5yr_avg": null },
    "minnesota":{ "planting_pct": null, "vs_5yr_avg": null },
    "nebraska": { "planting_pct": null, "vs_5yr_avg": null }
  },
  "notable_events": [],
  "signal": "on_track",
  "signal_reasoning": "Harvest pace equal to 5yr avg; condition slightly below average"
}
```

#### Few-Shot Example 3 — Edge case: drought stress
```
INPUT:
"For the week ending July 18, 2012, 40 percent of the corn crop 
was rated poor to very poor, the worst rating since 1988. 
Silking was 72 percent complete. The persistent drought conditions 
in the Corn Belt have severely stressed the crop."

OUTPUT:
{
  "report_date": "2012-07-18",
  "week_ending": "2012-07-18",
  "national": {
    "planting_pct": null,
    "emerged_pct": null,
    "silking_pct": 72,
    "dough_pct": null,
    "dented_pct": null,
    "mature_pct": null,
    "harvested_pct": null,
    "vs_last_year_planting": null,
    "vs_5yr_avg_planting": null,
    "condition_good_excellent_pct": null,
    "condition_poor_very_poor_pct": 40
  },
  "states": {
    "iowa": { "planting_pct": null, "vs_5yr_avg": null, "condition_good_excellent_pct": null },
    "illinois": { "planting_pct": null, "vs_5yr_avg": null },
    "indiana":  { "planting_pct": null, "vs_5yr_avg": null },
    "minnesota":{ "planting_pct": null, "vs_5yr_avg": null },
    "nebraska": { "planting_pct": null, "vs_5yr_avg": null }
  },
  "notable_events": ["drought_stress", "worst_condition_since_1988"],
  "signal": "very_delayed_very_bearish",
  "signal_reasoning": "40% poor/very poor — worst since 1988; drought across Corn Belt"
}
```

---

### 4.3 COFFEE PROMPT — CONAB Monthly Report

#### System Prompt (Portuguese-aware)
```
You are a precision agricultural data extractor specializing in 
Brazilian coffee production reports from CONAB (Companhia Nacional 
de Abastecimento).

The input text may be in Portuguese or English. Extract data and 
return ONLY a valid JSON object. 

CRITICAL RULES:
1. Return ONLY the JSON object — no explanation, no markdown
2. Use null for any value not found in the text
3. Never hallucinate values not present in the text
4. stage must be one of: "dormancy", "flowering", "fruit_development", 
   "maturation", "harvest", "post_harvest"
5. condition must be one of: "excellent", "good", "fair", "poor", "very_poor"
6. signal must be one of: "bumper_crop_bullish", "above_average", "on_track",
   "below_average", "crop_stress_bearish", "severe_stress_very_bearish"
7. All percentages are integers 0–100
8. production_change_pct: positive = higher than last year forecast
```

#### JSON Schema (Coffee)
```json
{
  "report_date": "YYYY-MM-DD",
  "report_type": "CONAB_monthly",
  "national": {
    "stage": null,
    "harvest_completion_pct": null,
    "production_forecast_million_bags": null,
    "production_change_vs_last_year_pct": null,
    "arabica_pct": null,
    "robusta_pct": null,
    "overall_condition": null
  },
  "regions": {
    "sul_de_minas": {
      "stage": null,
      "condition": null,
      "harvest_pct": null
    },
    "cerrado_mineiro": {
      "stage": null,
      "condition": null,
      "harvest_pct": null
    },
    "matas_de_minas": {
      "stage": null,
      "condition": null,
      "harvest_pct": null
    },
    "mogiana": {
      "stage": null,
      "condition": null,
      "harvest_pct": null
    },
    "cerrado_baiano": {
      "stage": null,
      "condition": null,
      "harvest_pct": null
    }
  },
  "weather_events": [],
  "signal": "on_track",
  "signal_reasoning": "Brief explanation (max 20 words)"
}
```

#### Few-Shot Example 1 — Flowering report (Portuguese)
```
INPUT:
"A florada principal do café arábica nas regiões do Sul de Minas e 
Cerrado Mineiro ocorreu em setembro com boas condições de umidade. 
A produção estimada para 2024/25 é de 54,94 milhões de sacas, 
queda de 2,5% em relação à safra anterior. No Cerrado Baiano, 
condições de seca prejudicaram o desenvolvimento vegetativo."

OUTPUT:
{
  "report_date": "2023-10-01",
  "report_type": "CONAB_monthly",
  "national": {
    "stage": "flowering",
    "harvest_completion_pct": null,
    "production_forecast_million_bags": 54.94,
    "production_change_vs_last_year_pct": -2.5,
    "arabica_pct": null,
    "robusta_pct": null,
    "overall_condition": "fair"
  },
  "regions": {
    "sul_de_minas": {
      "stage": "flowering",
      "condition": "good",
      "harvest_pct": null
    },
    "cerrado_mineiro": {
      "stage": "flowering",
      "condition": "good",
      "harvest_pct": null
    },
    "matas_de_minas": {
      "stage": null,
      "condition": null,
      "harvest_pct": null
    },
    "mogiana": {
      "stage": null,
      "condition": null,
      "harvest_pct": null
    },
    "cerrado_baiano": {
      "stage": "flowering",
      "condition": "poor",
      "harvest_pct": null
    }
  },
  "weather_events": ["drought_cerrado_baiano", "good_moisture_sul_minas"],
  "signal": "below_average",
  "signal_reasoning": "Production forecast -2.5% YoY; drought stress in Cerrado Baiano"
}
```

#### Few-Shot Example 2 — Harvest season (English translation)
```
INPUT:
"Coffee harvest in Brazil is 68% complete as of July 2024. 
Sul de Minas leads at 78% harvested with excellent quality. 
Total production revised upward to 58.1 million bags, 
up 4.2% from previous forecast. Robusta production in 
Espirito Santo at record levels."

OUTPUT:
{
  "report_date": "2024-07-01",
  "report_type": "CONAB_monthly",
  "national": {
    "stage": "harvest",
    "harvest_completion_pct": 68,
    "production_forecast_million_bags": 58.1,
    "production_change_vs_last_year_pct": 4.2,
    "arabica_pct": null,
    "robusta_pct": null,
    "overall_condition": "excellent"
  },
  "regions": {
    "sul_de_minas": {
      "stage": "harvest",
      "condition": "excellent",
      "harvest_pct": 78
    },
    "cerrado_mineiro": { "stage": "harvest", "condition": null, "harvest_pct": null },
    "matas_de_minas":  { "stage": "harvest", "condition": null, "harvest_pct": null },
    "mogiana":         { "stage": "harvest", "condition": null, "harvest_pct": null },
    "cerrado_baiano":  { "stage": "harvest", "condition": null, "harvest_pct": null }
  },
  "weather_events": ["record_robusta_production"],
  "signal": "bumper_crop_bullish",
  "signal_reasoning": "Production revised +4.2%; 68% complete; Sul de Minas excellent quality"
}
```

---

## 5. Feature Engineering Từ JSON

### 5.1 Corn — Features Được Tạo Ra

```python
# Từ JSON → numeric features
corn_planting_pct          # 0–100, national
corn_planting_vs_avg       # delta vs 5yr avg (positive = ahead)
corn_condition_ge_pct      # good+excellent %
corn_condition_pvp_pct     # poor+very_poor %
corn_stress_index          # condition_pvp_pct / 100
corn_iowa_planting_pct     # Iowa-specific (highest weight state)
corn_signal_encoded        # ordinal: very_bearish=-2, bearish=-1, on_track=0, bullish=1, very_bullish=2
corn_harvest_pct           # 0–100
```

### 5.2 Coffee — Features Được Tạo Ra

```python
coffee_stage_encoded          # ordinal: dormancy=0, flowering=1, fruit_dev=2, maturation=3, harvest=4
coffee_harvest_completion     # 0–100
coffee_production_change_pct  # YoY % change forecast
coffee_sul_minas_condition    # ordinal: very_poor=0...excellent=4
coffee_signal_encoded         # ordinal giống corn
coffee_flowering_flag         # binary: stage == "flowering"
coffee_drought_flag           # binary: "drought" in weather_events
```

### 5.3 So Sánh Với Synthetic Calendar (04)

| Feature synthetic (04) | Feature LLM (04b) | Cải thiện |
|-----------------------|-------------------|-----------|
| `is_planting` (0/1) | `corn_planting_pct` (0–100) | Granular |
| `is_harvest` (0/1) | `corn_harvest_pct` (0–100) | Granular |
| — | `corn_planting_vs_avg` | Mới hoàn toàn |
| — | `corn_condition_ge_pct` | Mới hoàn toàn |
| — | `corn_stress_index` | Mới hoàn toàn |
| `is_flowering` (0/1) | `coffee_flowering_flag` (0/1) | Tương đương |
| — | `coffee_production_change_pct` | Mới hoàn toàn |
| — | `coffee_drought_flag` | Mới hoàn toàn |

---

## 6. Ablation Study — Đánh Giá Định Lượng

### 6.1 Setup

Chạy lại pipeline từ Stage 3 (tensor packing) với 3 cấu hình calendar:

```
Experiment A: Synthetic only (04)           ← baseline hiện tại
Experiment B: LLM only (04b)               ← full replacement
Experiment C: Hybrid (04 + 04b merged)     ← kết hợp tốt nhất
```

### 6.2 Metrics Đánh Giá

| Metric | Mô tả |
|--------|-------|
| AUC-ROC | Primary metric — binary classification |
| PR-AUC | Quan trọng với imbalanced classes |
| Null Importance IV score của calendar features | LLM features có IV cao hơn synthetic? |
| Sharpe ratio (backtest) | Economic value |

### 6.3 Results (Actual)

| Calendar | AUC Coffee Daily | AUC Coffee Weekly | AUC Corn Daily | AUC Corn Weekly | Best |
|---------|---------------:|------------------:|---------------:|----------------:|------|
| Synthetic (A) | 0.405 | 0.404 | 0.475 | **0.598** | A wins Corn Weekly |
| LLM only (B) | 0.377 | **0.491** | **0.491** | 0.548 | B wins Coffee Wkly (+8.7pp), Corn Daily (+1.6pp) |
| Hybrid (C) | **0.412** | 0.407 | 0.487 | 0.526 | C wins Coffee Daily (+0.7pp) |

> Nếu LLM AUC < synthetic → vẫn là valid academic finding: "LLM-extracted calendar does not improve AUC over rule-based approximation for this task." Vẫn lấy điểm bonus vì đã integrate LLM đúng cách.

### 6.4 Zero-Shot vs Few-Shot Comparison

Chạy thêm variant để cross-validate:
- **Zero-shot:** Chỉ system prompt + schema, không có examples
- **Few-shot (3):** 3 examples như trên
- **Few-shot (5):** 5 examples (thêm 2 edge cases)

Đo JSON parse success rate và extraction accuracy trên 50 reports có ground truth.

---

## 7. Cấu Trúc Code

```
src/ingestion/
├── 04_farming_ingestion.py          ← giữ nguyên (synthetic baseline)
└── 04b_llm_farming_ingestion.py     ← module mới

config/
└── llm_farming_config.json          ← API keys, schema version, params

data/raw/farming/
├── synthetic/                       ← output 04 (giữ nguyên)
├── usda_reports/                    ← raw text từ USDA API
├── conab_reports/                   ← raw PDFs từ CONAB
└── llm_extracted/
    ├── corn_weekly_llm.csv          ← JSON → DataFrame
    └── coffee_monthly_llm.csv

models/ablation/
├── 04a_synthetic/                   ← pipeline results với synthetic
├── 04b_llm_only/                    ← pipeline results với LLM
└── 04c_hybrid/                      ← pipeline results với hybrid
```

---

## 8. Timeline

```
Ngày 1 — Data Acquisition
├── Đăng ký USDA QuickStats API key (free, instant)
├── Download 10 USDA reports sample (2020–2024)
├── Download 5 CONAB PDFs sample
└── Test pdfplumber extraction trên CONAB PDF

Ngày 2 — Prompt Development & Testing
├── Test zero-shot prompt trên 5 USDA reports
├── Refine → few-shot với 3 examples
├── Test few-shot trên 10 reports
├── Measure: JSON parse rate, null rate, accuracy vs known values
└── Finalize prompt (corn + coffee)

Ngày 3 — Full Pipeline Run
├── Fetch toàn bộ USDA 2010–2026 (~800 weekly reports)
│   (ước tính: ~4 giờ với rate limiting 10 req/min)
├── Fetch CONAB PDFs 2010–2026 (~64 monthly reports)
├── Run extraction pipeline
└── Build DataFrames + ffill

Ngày 4 — Integration & Ablation
├── Integrate vào 11_data_integration.py
├── Chạy ablation: A vs B vs C
├── Document kết quả
└── Viết subsection cho paper
```

---

## 9. Rủi Ro & Xử Lý

| Rủi ro | Xác suất | Xử lý |
|--------|----------|-------|
| USDA API trả về format khác nhau qua các năm | Cao | Dùng `try/except` + fallback về text parsing |
| CONAB PDF scan (không có text layer) | Trung | Claude Vision API — gửi PDF as image |
| LLM hallucinate giá trị không có trong text | Trung | Validation rule: cross-check với giá trị lân cận (±50% threshold) |
| API rate limit Claude | Thấp | Batch 50 reports, cache kết quả vào JSON files |
| CONAB thay đổi format PDF qua các năm | Cao | Few-shot per decade (2010s vs 2020s format khác) |
| JSON parse fail | Thấp | Retry với simplified prompt, log failed cases |

---

## 10. Đưa Vào Paper

### Section mới: IV.D — LLM-Enhanced Crop Calendar

```
Để khắc phục giới hạn của lịch mùa vụ tổng hợp (Phần II.B), 
chúng tôi phát triển module 04b sử dụng Claude API (Anthropic, 2024) 
để extract thông tin crop calendar có cấu trúc từ báo cáo nông nghiệp 
thực tế: USDA Weekly Crop Progress [REF] cho ngô và CONAB Boletim 
Agropecuário [REF] cho cà phê.

Pipeline gồm 3 bước: (1) fetch báo cáo từ API/PDF, (2) few-shot 
prompted extraction với Claude claude-sonnet-4-20250514 theo JSON schema 
định nghĩa sẵn, (3) validation tự động và time-series alignment.

Ablation study so sánh 3 cấu hình calendar (Bảng X) cho thấy...
```

### Bảng ablation (Section IX — kết quả thực tế):

| Calendar | Coffee Daily | Coffee Weekly | Corn Daily | Corn Weekly | Note |
|---------|-------------:|--------------:|-----------:|------------:|------|
| Synthetic (A) | 0.405 | 0.404 | 0.475 | **0.598** | baseline |
| LLM only (B) | 0.377 | **0.491** | **0.491** | 0.548 | +8.7pp coffee weekly |
| Hybrid (C) | **0.412** | 0.407 | 0.487 | 0.526 | +0.7pp coffee daily |

---

## 11. Checklist Hoàn Thành

- [x] USDA QuickStats API key đã đăng ký
- [x] 10 report samples đã download và test
- [x] Corn prompt đã finalized (zero-shot + 3 examples)
- [x] Coffee prompt đã finalized (Portuguese + 2 examples)
- [x] JSON validation với Pydantic schema
- [x] Full fetch 2010–2026 hoàn chỉnh
- [x] Ablation A vs B vs C đã chạy
- [x] Kết quả ablation đã document
- [x] Subsection IV.D đã viết (PIPELINE_REPORT.md §7b)
- [x] Bảng ablation đã thêm vào paper

---

*DS108 Module 04b Plan · v1.0 · 2026-05-29 · Bonus 2 LLM Integration*
