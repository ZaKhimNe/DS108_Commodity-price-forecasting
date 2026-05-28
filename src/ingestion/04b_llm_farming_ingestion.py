"""
Module 04b — LLM-Enhanced Farming Calendar
Bonus 2: Generative AI / LLMs integration

Replaces rule-based binary flags (04_farming_ingestion.py) with structured
data extracted from real agricultural reports:
  - USDA Weekly Crop Progress (corn) via QuickStats API
  - CONAB Monthly Coffee Reports (coffee) via PDF + Claude API

Usage:
    python src/ingestion/04b_llm_farming_ingestion.py

Requirements:
    pip install anthropic pdfplumber pydantic requests
    ANTHROPIC_API_KEY and USDA_NASS_API_KEY in .env
"""

import json
import os
import time
import logging
from datetime import datetime, date
from pathlib import Path
from typing import Optional

import pandas as pd
import requests
from dotenv import load_dotenv
from pydantic import BaseModel, Field, field_validator

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# ── Configuration ──────────────────────────────────────────────────────────────

with open("config/llm_farming_config.json") as f:
    CFG = json.load(f)

# Proxy 1gw.gwai.cloud dùng Authorization: Bearer (ANTHROPIC_AUTH_TOKEN)
# Official Anthropic dùng x-api-key (ANTHROPIC_API_KEY)
# Script tự detect cái nào được set
ANTHROPIC_AUTH_TOKEN = os.getenv("ANTHROPIC_AUTH_TOKEN", "")
ANTHROPIC_API_KEY    = os.getenv("ANTHROPIC_API_KEY", "")
_LLM_KEY             = ANTHROPIC_AUTH_TOKEN or ANTHROPIC_API_KEY
ANTHROPIC_BASE_URL   = os.getenv("ANTHROPIC_BASE_URL", "https://api.anthropic.com")
USDA_API_KEY         = os.getenv("USDA_NASS_API_KEY", "")

START_YEAR = CFG.get("start_year", 2010)
END_YEAR   = CFG.get("end_year",   2026)
MODEL      = CFG.get("model", "claude-haiku-4-5-20251001")
MAX_TOKENS = CFG.get("max_tokens", 1024)
RATE_SLEEP = CFG.get("rate_sleep_seconds", 1.2)   # ~50 req/min headroom

RAW_DIR  = Path("data/raw/farming")
USDA_DIR = RAW_DIR / "usda_reports"
CONAB_DIR = RAW_DIR / "conab_reports"
LLM_DIR  = RAW_DIR / "llm_extracted"

for d in [USDA_DIR, CONAB_DIR, LLM_DIR]:
    d.mkdir(parents=True, exist_ok=True)


# ══════════════════════════════════════════════════════════════════════════════
# Pydantic Schemas
# ══════════════════════════════════════════════════════════════════════════════

CORN_SIGNALS   = {"ahead_bullish", "on_track", "delayed_bearish",
                  "very_delayed_very_bearish", "condition_deteriorating",
                  "condition_improving"}
COFFEE_SIGNALS = {"bumper_crop_bullish", "above_average", "on_track",
                  "below_average", "crop_stress_bearish",
                  "severe_stress_very_bearish"}
COFFEE_STAGES  = {"dormancy", "flowering", "fruit_development",
                  "maturation", "harvest", "post_harvest"}
CONDITIONS     = {"excellent", "good", "fair", "poor", "very_poor"}


class CornStateData(BaseModel):
    planting_pct: Optional[int]  = Field(None, ge=0, le=100)
    vs_5yr_avg:   Optional[int]  = None
    condition_good_excellent_pct: Optional[int] = Field(None, ge=0, le=100)


class CornNationalData(BaseModel):
    planting_pct:          Optional[int] = Field(None, ge=0, le=100)
    emerged_pct:           Optional[int] = Field(None, ge=0, le=100)
    silking_pct:           Optional[int] = Field(None, ge=0, le=100)
    dough_pct:             Optional[int] = Field(None, ge=0, le=100)
    dented_pct:            Optional[int] = Field(None, ge=0, le=100)
    mature_pct:            Optional[int] = Field(None, ge=0, le=100)
    harvested_pct:         Optional[int] = Field(None, ge=0, le=100)
    vs_last_year_planting: Optional[int] = None
    vs_5yr_avg_planting:   Optional[int] = None
    condition_good_excellent_pct: Optional[int] = Field(None, ge=0, le=100)
    condition_poor_very_poor_pct: Optional[int] = Field(None, ge=0, le=100)


class CornReportData(BaseModel):
    report_date:     str
    week_ending:     str
    national:        CornNationalData
    states:          dict[str, CornStateData]
    notable_events:  list[str] = []
    signal:          str = "on_track"
    signal_reasoning: str = ""

    @field_validator("signal")
    @classmethod
    def validate_signal(cls, v: str) -> str:
        return v if v in CORN_SIGNALS else "on_track"


class CoffeeRegionData(BaseModel):
    stage:       Optional[str] = None
    condition:   Optional[str] = None
    harvest_pct: Optional[int] = Field(None, ge=0, le=100)

    @field_validator("stage")
    @classmethod
    def validate_stage(cls, v):
        return v if v in COFFEE_STAGES else None

    @field_validator("condition")
    @classmethod
    def validate_condition(cls, v):
        return v if v in CONDITIONS else None


class CoffeeNationalData(BaseModel):
    stage:                              Optional[str]   = None
    harvest_completion_pct:             Optional[int]   = Field(None, ge=0, le=100)
    production_forecast_million_bags:   Optional[float] = None
    production_change_vs_last_year_pct: Optional[float] = None
    arabica_pct:   Optional[int] = Field(None, ge=0, le=100)
    robusta_pct:   Optional[int] = Field(None, ge=0, le=100)
    overall_condition: Optional[str] = None

    @field_validator("stage")
    @classmethod
    def validate_stage(cls, v):
        return v if v in COFFEE_STAGES else None


class CoffeeReportData(BaseModel):
    report_date:  str
    report_type:  str = "CONAB_monthly"
    national:     CoffeeNationalData
    regions:      dict[str, CoffeeRegionData]
    weather_events: list[str] = []
    signal:          str = "on_track"
    signal_reasoning: str = ""

    @field_validator("signal")
    @classmethod
    def validate_signal(cls, v: str) -> str:
        return v if v in COFFEE_SIGNALS else "on_track"


# ══════════════════════════════════════════════════════════════════════════════
# Prompts
# ══════════════════════════════════════════════════════════════════════════════

CORN_SYSTEM = """You are a precision agricultural data extractor specializing in USDA Crop Progress reports.

Extract structured planting and crop condition data and return ONLY a valid JSON object.

CRITICAL RULES:
1. Return ONLY the JSON — no explanation, no markdown, no preamble
2. Use null for any value not explicitly mentioned in the text
3. Never hallucinate or infer values not present in the text
4. All percentages must be integers 0-100
5. signal must be one of: "ahead_bullish", "on_track", "delayed_bearish",
   "very_delayed_very_bearish", "condition_deteriorating", "condition_improving"
6. Extract ONLY corn data if multiple crops are mentioned"""

CORN_USER_TEMPLATE = """Extract corn crop progress data from this USDA report.
Report date: {report_date}

Text:
{text}

Return this exact JSON structure (use null for missing values):
{schema}"""

COFFEE_SYSTEM = """You are a precision agricultural data extractor specializing in Brazilian coffee
production reports from CONAB (Companhia Nacional de Abastecimento).

The input text may be in Portuguese or English. Extract data and return ONLY a valid JSON object.

CRITICAL RULES:
1. Return ONLY the JSON — no explanation, no markdown
2. Use null for any value not found in the text
3. Never hallucinate values not present in the text
4. stage must be one of: "dormancy", "flowering", "fruit_development",
   "maturation", "harvest", "post_harvest"
5. condition must be one of: "excellent", "good", "fair", "poor", "very_poor"
6. signal must be one of: "bumper_crop_bullish", "above_average", "on_track",
   "below_average", "crop_stress_bearish", "severe_stress_very_bearish"
7. production_change_pct: positive = higher than last year forecast"""

COFFEE_USER_TEMPLATE = """Extract coffee crop data from this CONAB report.
Report date: {report_date}

Text:
{text}

Return this exact JSON structure (use null for missing values):
{schema}"""

CORN_SCHEMA = json.dumps({
    "report_date": "YYYY-MM-DD", "week_ending": "YYYY-MM-DD",
    "national": {
        "planting_pct": None, "emerged_pct": None, "silking_pct": None,
        "dough_pct": None, "dented_pct": None, "mature_pct": None,
        "harvested_pct": None, "vs_last_year_planting": None,
        "vs_5yr_avg_planting": None, "condition_good_excellent_pct": None,
        "condition_poor_very_poor_pct": None
    },
    "states": {
        "iowa":     {"planting_pct": None, "vs_5yr_avg": None, "condition_good_excellent_pct": None},
        "illinois": {"planting_pct": None, "vs_5yr_avg": None},
        "indiana":  {"planting_pct": None, "vs_5yr_avg": None},
        "minnesota":{"planting_pct": None, "vs_5yr_avg": None},
        "nebraska": {"planting_pct": None, "vs_5yr_avg": None}
    },
    "notable_events": [], "signal": "on_track", "signal_reasoning": ""
}, indent=2)

COFFEE_SCHEMA = json.dumps({
    "report_date": "YYYY-MM-DD", "report_type": "CONAB_monthly",
    "national": {
        "stage": None, "harvest_completion_pct": None,
        "production_forecast_million_bags": None,
        "production_change_vs_last_year_pct": None,
        "arabica_pct": None, "robusta_pct": None, "overall_condition": None
    },
    "regions": {
        "sul_de_minas":    {"stage": None, "condition": None, "harvest_pct": None},
        "cerrado_mineiro": {"stage": None, "condition": None, "harvest_pct": None},
        "matas_de_minas":  {"stage": None, "condition": None, "harvest_pct": None},
        "mogiana":         {"stage": None, "condition": None, "harvest_pct": None},
        "cerrado_baiano":  {"stage": None, "condition": None, "harvest_pct": None}
    },
    "weather_events": [], "signal": "on_track", "signal_reasoning": ""
}, indent=2)


# ══════════════════════════════════════════════════════════════════════════════
# LLM Extractor
# ══════════════════════════════════════════════════════════════════════════════

class LLMExtractor:
    """Calls Claude API to extract structured JSON from agricultural report text."""

    def __init__(self) -> None:
        if not _LLM_KEY:
            raise EnvironmentError(
                "No API key found. Set ANTHROPIC_AUTH_TOKEN (proxy) "
                "or ANTHROPIC_API_KEY (official) in .env"
            )
        import anthropic as _anthropic
        if ANTHROPIC_AUTH_TOKEN:
            self._headers = {
                "Authorization": f"Bearer {ANTHROPIC_AUTH_TOKEN}",
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            }
            log.info("LLM auth: Bearer token | endpoint: %s", ANTHROPIC_BASE_URL)
        else:
            self._headers = {
                "x-api-key": ANTHROPIC_API_KEY,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            }
            log.info("LLM auth: x-api-key | endpoint: %s", ANTHROPIC_BASE_URL)
        # Keep SDK client as fallback reference (not used for actual calls)
        self.client = None

    def _call(self, system: str, user: str) -> str:
        resp = requests.post(
            f"{ANTHROPIC_BASE_URL}/v1/messages",
            headers=self._headers,
            json={
                "model": MODEL,
                "max_tokens": MAX_TOKENS,
                "system": system,
                "messages": [{"role": "user", "content": user}],
            },
            timeout=60,
        )

        # Rate limit: wait even on success to avoid hammering proxy
        time.sleep(RATE_SLEEP)

        if resp.status_code == 429:
            retry_after = int(resp.headers.get("retry-after", 30))
            log.info("429 rate limit — sleeping %ds", retry_after)
            time.sleep(retry_after)
            raise RuntimeError(f"Rate limited (429)")

        resp.raise_for_status()

        body = resp.text.strip()
        if not body:
            raise RuntimeError("Empty response from API")

        data = resp.json()
        # Handle both Anthropic format {"content":[{"text":"..."}]} and
        # proxy-specific format {"choices":[{"message":{"content":"..."}}]}
        if "content" in data and data["content"]:
            return data["content"][0].get("text", "").strip()
        if "choices" in data and data["choices"]:
            return data["choices"][0].get("message", {}).get("content", "").strip()
        raise RuntimeError(f"Unexpected response format: {str(data)[:100]}")

    def extract_corn(self, text: str, report_date: str) -> Optional[CornReportData]:
        user = CORN_USER_TEMPLATE.format(
            report_date=report_date, text=text[:3000], schema=CORN_SCHEMA
        )
        cache_path = USDA_DIR / f"llm_{report_date}.json"
        if cache_path.exists():
            raw = cache_path.read_text()
        else:
            try:
                raw = self._call(CORN_SYSTEM, user)
                cache_path.write_text(raw)
            except Exception as e:
                log.warning("LLM call failed for corn %s: %s", report_date, e)
                return None
        try:
            data = json.loads(raw)
            # Ensure nested state objects use CornStateData
            states = {k: CornStateData(**v) for k, v in data.get("states", {}).items()}
            data["states"] = states
            return CornReportData(**data)
        except Exception as e:
            log.warning("Parse failed for corn %s: %s | raw: %.100s", report_date, e, raw)
            return None

    def extract_coffee(self, text: str, report_date: str) -> Optional[CoffeeReportData]:
        user = COFFEE_USER_TEMPLATE.format(
            report_date=report_date, text=text[:3000], schema=COFFEE_SCHEMA
        )
        cache_path = CONAB_DIR / f"llm_{report_date}.json"
        if cache_path.exists():
            raw = cache_path.read_text()
        else:
            try:
                raw = self._call(COFFEE_SYSTEM, user)
                cache_path.write_text(raw)
            except Exception as e:
                log.warning("LLM call failed for coffee %s: %s", report_date, e)
                return None
        try:
            data = json.loads(raw)
            regions = {k: CoffeeRegionData(**v) for k, v in data.get("regions", {}).items()}
            data["regions"] = regions
            return CoffeeReportData(**data)
        except Exception as e:
            log.warning("Parse failed for coffee %s: %s | raw: %.100s", report_date, e, raw)
            return None


# ══════════════════════════════════════════════════════════════════════════════
# USDA Fetcher — QuickStats API (structured CSV, no LLM for numbers)
# ══════════════════════════════════════════════════════════════════════════════

# Map USDA short_desc → our column names
_USDA_MEASURES = {
    "CORN - PROGRESS, MEASURED IN PCT PLANTED":             "planting_pct",
    "CORN - PROGRESS, MEASURED IN PCT EMERGED":             "emerged_pct",
    "CORN - PROGRESS, MEASURED IN PCT SILKING":             "silking_pct",
    "CORN - PROGRESS, MEASURED IN PCT DOUGH":               "dough_pct",
    "CORN - PROGRESS, MEASURED IN PCT DENTED":              "dented_pct",
    "CORN - PROGRESS, MEASURED IN PCT MATURE":              "mature_pct",
    "CORN - PROGRESS, MEASURED IN PCT HARVESTED":           "harvested_pct",
    "CORN - CONDITION, MEASURED IN PCT GOOD":               "condition_good_pct",
    "CORN - CONDITION, MEASURED IN PCT EXCELLENT":          "condition_excellent_pct",
    "CORN - CONDITION, MEASURED IN PCT POOR":               "condition_poor_pct",
    "CORN - CONDITION, MEASURED IN PCT VERY POOR":          "condition_very_poor_pct",
}
_USDA_STATES = ["IOWA", "ILLINOIS", "INDIANA", "MINNESOTA", "NEBRASKA",
                "US TOTAL"]


def _signal_from_numbers(planting_vs_avg: Optional[float],
                         condition_pvp: Optional[float]) -> str:
    """Derive signal label from numeric indicators without LLM."""
    if planting_vs_avg is not None:
        if planting_vs_avg >= 5:
            return "ahead_bullish"
        if planting_vs_avg <= -15:
            return "very_delayed_very_bearish"
        if planting_vs_avg <= -5:
            return "delayed_bearish"
    if condition_pvp is not None:
        if condition_pvp >= 30:
            return "condition_deteriorating"
        if condition_pvp <= 10:
            return "condition_improving"
    return "on_track"


def parse_usda_downloaded_csv(csv_path: str | Path) -> pd.DataFrame:
    """
    Parse a USDA QuickStats CSV export (downloaded manually from
    https://quickstats.nass.usda.gov) into the same pivot format
    that fetch_usda_corn() produces via API.

    The CSV must have columns:
        Week Ending, Geo Level, State, Data Item, Value

    Only columns present in _USDA_MEASURES are kept; missing measure types
    (e.g. if you only downloaded PCT PLANTED) result in NaN columns — this is
    expected. Download separate CSV files for each Data Item and merge them,
    or re-run with all Data Items selected on the QuickStats query builder.

    Args:
        csv_path: path to the downloaded .csv file (UUID filename is fine)

    Returns:
        DataFrame with columns [week_ending, state_name, planting_pct, ...]
        ready to be passed into build_corn_features().
    """
    csv_path = Path(csv_path)
    log.info("Parsing USDA downloaded CSV: %s (%d KB)",
             csv_path.name, csv_path.stat().st_size // 1024)

    df = pd.read_csv(csv_path, dtype=str)

    # Normalise column names (QuickStats sometimes differs by export version)
    df.columns = [c.strip().title().replace(" ", "_") for c in df.columns]
    # Expected after normalisation: Week_Ending, Geo_Level, State, Data_Item, Value
    col_map = {
        "Week_Ending": "week_ending",
        "Geo_Level":   "geo_level",
        "State":       "state_name",
        "Data_Item":   "data_item",
        "Value":       "value_raw",
    }
    df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})

    required = {"week_ending", "geo_level", "state_name", "data_item", "value_raw"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing expected columns: {missing}. "
                         f"Got: {list(df.columns)}")

    # Parse date
    df["week_ending"] = pd.to_datetime(df["week_ending"], errors="coerce")
    df = df.dropna(subset=["week_ending"])

    # Filter to target states + national
    target = {s.upper() for s in _USDA_STATES}
    df["state_name"] = df["state_name"].str.upper().str.strip()
    df = df[df["state_name"].isin(target)].copy()

    # Numeric value — "(D)" means withheld → NaN
    df["value_int"] = pd.to_numeric(
        df["value_raw"].str.replace(",", "").str.replace(r"[()D ]", "", regex=True),
        errors="coerce",
    )

    # Map Data Item → column name
    df["data_item"] = df["data_item"].str.strip().str.upper()
    _measures_upper = {k.upper(): v for k, v in _USDA_MEASURES.items()}
    df["col"] = df["data_item"].map(_measures_upper)
    df = df.dropna(subset=["col"])

    if df.empty:
        log.warning("No matching Data Items found in CSV. "
                    "Check that 'Data Item' column contains CORN PROGRESS/CONDITION rows.")
        return pd.DataFrame(columns=["week_ending", "state_name"])

    # Pivot: one row per (week_ending, state_name), one column per measure
    pivot = df.pivot_table(
        index=["week_ending", "state_name"],
        columns="col",
        values="value_int",
        aggfunc="first",
    ).reset_index()
    pivot.columns.name = None

    # Combine Good + Excellent → condition_ge_pct
    if "condition_good_pct" in pivot.columns and "condition_excellent_pct" in pivot.columns:
        pivot["condition_ge_pct"] = (
            pivot["condition_good_pct"].fillna(0)
            + pivot["condition_excellent_pct"].fillna(0)
        )
    if "condition_poor_pct" in pivot.columns and "condition_very_poor_pct" in pivot.columns:
        pivot["condition_pvp_pct"] = (
            pivot["condition_poor_pct"].fillna(0)
            + pivot["condition_very_poor_pct"].fillna(0)
        )

    measures_found = [c for c in _USDA_MEASURES.values() if c in pivot.columns]
    log.info("Parsed %d rows, %d states, measures present: %s",
             len(pivot), pivot["state_name"].nunique(), measures_found)
    return pivot


# Path where auto-downloaded CSV will be looked up as fallback
_USDA_CSV_FALLBACK = USDA_DIR / "usda_corn_planting.csv"


def fetch_usda_corn(use_cache: bool = True) -> pd.DataFrame:
    """
    Fetch USDA QuickStats corn crop progress data for Illinois, Indiana,
    Iowa, Minnesota, Nebraska + US Total (2010–2026).

    Priority:
      1. Cache file (usda_corn_raw.csv) — fastest
      2. Downloaded CSV in data/raw/farming/usda_reports/ — no API key needed
      3. QuickStats API — requires USDA_NASS_API_KEY in .env

    Returns weekly DataFrame with one row per (week_ending, state).
    Columns: planting_pct, emerged_pct, ..., condition_ge_pct, condition_pvp_pct, signal
    """
    cache_file = USDA_DIR / "usda_corn_raw.csv"

    if use_cache and cache_file.exists():
        log.info("Loading USDA corn from cache: %s", cache_file)
        return pd.read_csv(cache_file, parse_dates=["week_ending"])

    # Fallback: parse any manually-downloaded USDA CSV
    downloaded_csvs = sorted(USDA_DIR.glob("usda_corn*.csv")) + \
                      sorted(USDA_DIR.glob("*.csv"))  # includes UUID-named files
    downloaded_csvs = [p for p in downloaded_csvs if p != cache_file]
    if downloaded_csvs:
        log.info("No API key — using downloaded CSV: %s", downloaded_csvs[0].name)
        pivot = parse_usda_downloaded_csv(downloaded_csvs[0])
        pivot.to_csv(cache_file, index=False)
        log.info("Cached parsed result → %s", cache_file)
        return pivot

    if not USDA_API_KEY:
        raise EnvironmentError(
            "No USDA data available. Either:\n"
            "  1. Place a downloaded USDA QuickStats CSV in: data/raw/farming/usda_reports/\n"
            "     (Download at: https://quickstats.nass.usda.gov — select Corn, Progress)\n"
            "  2. Or set USDA_NASS_API_KEY in .env"
        )

    log.info("Fetching USDA QuickStats corn progress 2010–2026 ...")
    base_url = "https://quickstats.nass.usda.gov/api/api_GET/"
    rows = []

    for state in _USDA_STATES:
        params = {
            "key":              USDA_API_KEY,
            "commodity_desc":   "CORN",
            "statisticcat_desc": "PROGRESS,CONDITION",
            "freq_desc":        "WEEKLY",
            "year__GE":         str(START_YEAR),
            "year__LE":         str(END_YEAR),
            "state_name":       state,
            "format":           "JSON",
        }
        try:
            resp = requests.get(base_url, params=params, timeout=60)
            resp.raise_for_status()
            data = resp.json().get("data", [])
            rows.extend(data)
            log.info("  USDA %s: %d records", state, len(data))
            time.sleep(0.5)
        except Exception as e:
            log.warning("USDA fetch failed for %s: %s", state, e)

    if not rows:
        raise RuntimeError("No USDA data fetched — check API key and connectivity")

    raw = pd.DataFrame(rows)
    raw.to_csv(cache_file, index=False)

    # Pivot: one row per (week_ending, state)
    raw["week_ending"] = pd.to_datetime(raw["week_ending_date"], errors="coerce")
    raw["value_int"] = pd.to_numeric(raw["Value"].str.replace(",", ""), errors="coerce")
    raw["col"] = raw["short_desc"].map(_USDA_MEASURES)
    raw = raw.dropna(subset=["col", "week_ending"])

    pivot = raw.pivot_table(
        index=["week_ending", "state_name"],
        columns="col",
        values="value_int",
        aggfunc="first",
    ).reset_index()

    # Combine good + excellent into one column
    if "condition_good_pct" in pivot and "condition_excellent_pct" in pivot:
        pivot["condition_ge_pct"] = (
            pivot["condition_good_pct"].fillna(0)
            + pivot["condition_excellent_pct"].fillna(0)
        )
    if "condition_poor_pct" in pivot and "condition_very_poor_pct" in pivot:
        pivot["condition_pvp_pct"] = (
            pivot["condition_poor_pct"].fillna(0)
            + pivot["condition_very_poor_pct"].fillna(0)
        )

    pivot.columns.name = None
    return pivot


# ══════════════════════════════════════════════════════════════════════════════
# CONAB Fetcher — PDF download + pdfplumber text extraction
# ══════════════════════════════════════════════════════════════════════════════

def _scrape_conab_report_urls() -> list[dict]:
    """
    Scrape CONAB boletim page for PDF report URLs.
    Returns list of {date: 'YYYY-MM-DD', url: 'https://...'}.
    Falls back to a hardcoded sample if scraping fails.
    """
    # CONAB reports are quarterly: Jan, Apr, Jul, Oct
    # This generator builds URLs for known report patterns
    sample_reports = []
    for year in range(START_YEAR, END_YEAR + 1):
        for month in [1, 4, 7, 10]:
            sample_reports.append({
                "date": f"{year}-{month:02d}-01",
                "url": None,   # populated by scraper below
            })

    try:
        import urllib.request
        from html.parser import HTMLParser

        class PDFLinkParser(HTMLParser):
            def __init__(self):
                super().__init__()
                self.links = []
            def handle_starttag(self, tag, attrs):
                if tag == "a":
                    href = dict(attrs).get("href", "")
                    if "cafe" in href.lower() and href.endswith(".pdf"):
                        self.links.append(href)

        page_url = "https://www.conab.gov.br/info-agro/safras/cafe/boletim-da-safra-de-cafe"
        req = urllib.request.Request(page_url, headers={"User-Agent": "DS108-research"})
        with urllib.request.urlopen(req, timeout=15) as r:
            html = r.read().decode("utf-8", errors="replace")
        parser = PDFLinkParser()
        parser.feed(html)
        if parser.links:
            log.info("Found %d CONAB PDF links on page", len(parser.links))
            # Match links to dates (newest first on CONAB page)
            for i, link in enumerate(parser.links[:len(sample_reports)]):
                full = link if link.startswith("http") else "https://www.conab.gov.br" + link
                sample_reports[-(i+1)]["url"] = full
    except Exception as e:
        log.warning("CONAB scrape failed: %s — URLs will be None, LLM step skipped", e)

    return [r for r in sample_reports if r["url"]]


def _download_pdf(url: str, dest: Path) -> bool:
    if dest.exists():
        return True
    try:
        import urllib.request
        req = urllib.request.Request(url, headers={"User-Agent": "DS108-research"})
        with urllib.request.urlopen(req, timeout=30) as r:
            dest.write_bytes(r.read())
        return True
    except Exception as e:
        log.warning("PDF download failed %s: %s", url, e)
        return False


def _extract_pdf_text(pdf_path: Path) -> str:
    try:
        import pdfplumber
        with pdfplumber.open(pdf_path) as pdf:
            pages = [p.extract_text() or "" for p in pdf.pages[:6]]
        return "\n".join(pages)
    except Exception as e:
        log.warning("pdfplumber failed on %s: %s", pdf_path.name, e)
        return ""


def fetch_conab_coffee(llm: Optional["LLMExtractor"] = None,
                       use_cache: bool = True) -> list[CoffeeReportData]:
    """
    Fetch CONAB PDF reports, extract text, and parse with LLM.
    Returns list of CoffeeReportData objects (quarterly, 2010–2026).
    """
    reports: list[CoffeeReportData] = []
    urls = _scrape_conab_report_urls()

    if not urls:
        log.warning("No CONAB URLs found — returning empty list")
        return reports

    log.info("Processing %d CONAB reports ...", len(urls))
    for entry in urls:
        report_date = entry["date"]
        pdf_path = CONAB_DIR / f"conab_{report_date}.pdf"

        # Download PDF
        if not _download_pdf(entry["url"], pdf_path):
            continue

        # Extract text
        text = _extract_pdf_text(pdf_path)
        if len(text) < 100:
            log.warning("Very short text from %s (%d chars) — skipping", pdf_path.name, len(text))
            continue

        # LLM extraction
        if llm is None:
            log.info("  LLM not available, skipping extraction for %s", report_date)
            continue

        result = llm.extract_coffee(text, report_date)
        if result:
            reports.append(result)
            log.info("  ✓ coffee %s: signal=%s", report_date, result.signal)
        else:
            log.warning("  ✗ extraction failed for %s", report_date)

    return reports


# ══════════════════════════════════════════════════════════════════════════════
# PSD Fallback — USDA PSD coffee Brazil (annual) + LLM signal classification
# ══════════════════════════════════════════════════════════════════════════════

_PSD_SIGNAL_SYSTEM = """You are an agricultural commodity analyst specializing in Brazilian coffee markets.
Given annual Brazil coffee production statistics from the USDA PSD database, classify the market signal.

Return ONLY a JSON object with exactly two fields:
- "signal": one of "bumper_crop_bullish", "above_average", "on_track", "below_average", "crop_stress_bearish", "severe_stress_very_bearish"
- "signal_reasoning": max 20 words explaining the signal

No explanation, no markdown — ONLY the JSON object."""

_PSD_SIGNAL_USER = """Brazil Coffee Production Statistics — {year}:
- Total Production: {production:.2f} million 60-kg bags
- Arabica: {arabica:.2f} million bags ({arabica_pct:.0f}% of total)
- Robusta: {robusta:.2f} million bags ({robusta_pct:.0f}% of total)
- Production change vs prior year: {change:+.1f}%
- Bean Exports: {exports:.2f} million bags
- Ending Stocks: {stocks:.2f} million bags
- Stock-to-use ratio: {stu:.1f}%

Classify the market signal for coffee price direction."""


def fetch_coffee_from_psd(
    psd_path: str | Path = "data/psd_coffee_brazil.csv",
    llm: Optional["LLMExtractor"] = None,
    use_cache: bool = True,
) -> list[CoffeeReportData]:
    """
    Parse USDA PSD coffee Brazil data and use LLM to classify annual market signal.

    This is the fallback when CONAB PDFs are unavailable. Demonstrates LLM use on
    structured production statistics — Claude classifies signal + reasoning per year.

    Args:
        psd_path: path to filtered coffee+Brazil PSD CSV (run psd filter first)
        llm:      LLMExtractor instance (Claude API)
        use_cache: skip years already cached in conab_reports/llm_*.json

    Returns:
        list of CoffeeReportData (one per year, report_type='PSD_annual')
    """
    psd_path = Path(psd_path)
    if not psd_path.exists():
        log.warning("PSD coffee Brazil file not found: %s", psd_path)
        return []

    df = pd.read_csv(psd_path)
    df.columns = [c.strip() for c in df.columns]

    # Unit is "1000 60 KG BAGS" → divide by 1000 → million 60-kg bags
    def _attr(attr_name: str) -> pd.Series:
        mask = df["Attribute_Description"].str.strip() == attr_name
        s = df[mask][["Market_Year", "Value"]].drop_duplicates("Market_Year")
        s = s.set_index("Market_Year")["Value"] / 1000.0
        return s

    prod     = _attr("Production")
    arabica  = _attr("Arabica Production")
    robusta  = _attr("Robusta Production")
    exports  = _attr("Bean Exports")
    e_stocks = _attr("Ending Stocks")
    supply   = _attr("Total Supply")

    years = sorted(prod.index)
    reports: list[CoffeeReportData] = []

    log.info("Processing %d years of PSD coffee Brazil data ...", len(years))
    for year in years:
        if year < START_YEAR or year > END_YEAR:
            continue

        report_date = f"{year}-07-01"  # mid-year estimate

        p      = float(prod.get(year, 0) or 0)
        a      = float(arabica.get(year, 0) or 0)
        r_     = float(robusta.get(year, 0) or 0)
        ex     = float(exports.get(year, 0) or 0)
        es     = float(e_stocks.get(year, 0) or 0)
        sup    = float(supply.get(year, 0) or 0)
        p_prev = float(prod.get(year - 1, 0) or 0)

        change    = ((p - p_prev) / p_prev * 100) if p_prev > 0 else 0.0
        arab_pct  = (a / p * 100) if p > 0 else 0.0
        rob_pct   = (r_ / p * 100) if p > 0 else 0.0
        stu       = (es / sup * 100) if sup > 0 else 0.0

        # LLM signal classification
        signal = "on_track"
        reasoning = ""
        if llm is not None and p > 0:
            cache_path = CONAB_DIR / f"llm_{report_date}_psd.json"
            if use_cache and cache_path.exists():
                try:
                    cached = json.loads(cache_path.read_text())
                    signal    = cached.get("signal", "on_track")
                    reasoning = cached.get("signal_reasoning", "")
                except Exception:
                    pass
            else:
                prompt = _PSD_SIGNAL_USER.format(
                    year=year, production=p, arabica=a, arabica_pct=arab_pct,
                    robusta=r_, robusta_pct=rob_pct, change=change,
                    exports=ex, stocks=es, stu=stu,
                )
                try:
                    raw = llm._call(_PSD_SIGNAL_SYSTEM, prompt)
                    cache_path.write_text(raw)
                    parsed = json.loads(raw)
                    signal    = parsed.get("signal", "on_track")
                    reasoning = parsed.get("signal_reasoning", "")
                    log.info("  ✓ PSD coffee %d: signal=%s", year, signal)
                except Exception as e:
                    log.warning("  LLM signal failed for %d: %s", year, e)
        else:
            # Rule-based fallback when no LLM
            if change >= 10:   signal = "bumper_crop_bullish"
            elif change >= 3:  signal = "above_average"
            elif change <= -10: signal = "crop_stress_bearish"
            elif change <= -3:  signal = "below_average"
            reasoning = f"Production {change:+.1f}% YoY; stock-to-use {stu:.1f}%"

        nat = CoffeeNationalData(
            stage=None,
            harvest_completion_pct=None,
            production_forecast_million_bags=round(p, 3),
            production_change_vs_last_year_pct=round(change, 2),
            arabica_pct=int(arab_pct) if arab_pct else None,
            robusta_pct=int(rob_pct) if rob_pct else None,
            overall_condition=None,
        )
        regions = {k: CoffeeRegionData() for k in
                   ["sul_de_minas", "cerrado_mineiro", "matas_de_minas",
                    "mogiana", "cerrado_baiano"]}

        report = CoffeeReportData(
            report_date=report_date,
            report_type="PSD_annual",
            national=nat,
            regions=regions,
            weather_events=[],
            signal=signal,
            signal_reasoning=reasoning,
        )
        reports.append(report)

    log.info("PSD: built %d annual coffee reports (LLM=%s)", len(reports), llm is not None)
    return reports


# ══════════════════════════════════════════════════════════════════════════════
# Feature Builders
# ══════════════════════════════════════════════════════════════════════════════

# Signal → ordinal encoding
_CORN_SIGNAL_MAP = {
    "very_delayed_very_bearish": -2,
    "delayed_bearish": -1,
    "on_track": 0,
    "condition_deteriorating": -1,
    "condition_improving": 1,
    "ahead_bullish": 2,
}
_COFFEE_SIGNAL_MAP = {
    "severe_stress_very_bearish": -2,
    "crop_stress_bearish": -1,
    "below_average": -1,
    "on_track": 0,
    "above_average": 1,
    "bumper_crop_bullish": 2,
}


def build_corn_features(usda_pivot: pd.DataFrame) -> pd.DataFrame:
    """
    Build daily corn feature DataFrame from USDA pivot.

    Columns produced (national US-level):
        corn_planting_pct, corn_planting_vs_5yr_avg, corn_harvest_pct,
        corn_condition_ge_pct, corn_condition_pvp_pct, corn_stress_index,
        corn_iowa_planting_pct, corn_signal_encoded
    """
    us = usda_pivot[usda_pivot["state_name"] == "US TOTAL"].copy()
    iowa = usda_pivot[usda_pivot["state_name"] == "IOWA"].copy()

    def safe_col(df: pd.DataFrame, c: str) -> pd.Series:
        return df[c] if c in df.columns else pd.Series(dtype=float, index=df.index)

    us["corn_planting_vs_5yr_avg"] = None  # QuickStats doesn't return this directly
    us["corn_signal_encoded"] = us.apply(
        lambda r: _CORN_SIGNAL_MAP.get(
            _signal_from_numbers(None, safe_col(us, "condition_pvp_pct").get(r.name)),
            0
        ),
        axis=1,
    )

    us_feat = us[["week_ending"]].copy()
    for src, dst in [
        ("planting_pct", "corn_planting_pct"),
        ("emerged_pct",  "corn_emerged_pct"),
        ("silking_pct",  "corn_silking_pct"),
        ("harvested_pct","corn_harvest_pct"),
        ("condition_ge_pct",  "corn_condition_ge_pct"),
        ("condition_pvp_pct", "corn_condition_pvp_pct"),
    ]:
        us_feat[dst] = safe_col(us, src).values

    us_feat["corn_stress_index"] = (us_feat.get("corn_condition_pvp_pct", 0) / 100)
    us_feat["corn_signal_encoded"] = us["corn_signal_encoded"].values

    iowa_feat = iowa[["week_ending"]].copy()
    iowa_feat["corn_iowa_planting_pct"] = safe_col(iowa, "planting_pct").values

    merged = pd.merge(us_feat, iowa_feat, on="week_ending", how="left")
    merged = merged.sort_values("week_ending").set_index("week_ending")
    return _to_daily(merged, "corn_llm")


def build_coffee_features(reports: list[CoffeeReportData]) -> pd.DataFrame:
    """
    Build daily coffee feature DataFrame from CONAB LLM-extracted reports.

    Columns produced:
        coffee_stage_encoded, coffee_harvest_completion, coffee_production_change_pct,
        coffee_sul_minas_condition, coffee_signal_encoded,
        coffee_flowering_flag, coffee_drought_flag
    """
    _stage_enc = {s: i for i, s in enumerate(
        ["dormancy", "flowering", "fruit_development", "maturation", "harvest", "post_harvest"]
    )}
    _cond_enc  = {"very_poor": 0, "poor": 1, "fair": 2, "good": 3, "excellent": 4}

    rows = []
    for r in reports:
        nat = r.national
        sul = r.regions.get("sul_de_minas", CoffeeRegionData())
        rows.append({
            "week_ending": pd.to_datetime(r.report_date),
            "coffee_stage_encoded":         _stage_enc.get(nat.stage or "", 0),
            "coffee_harvest_completion":     nat.harvest_completion_pct,
            "coffee_production_change_pct":  nat.production_change_vs_last_year_pct,
            "coffee_sul_minas_condition":    _cond_enc.get(sul.condition or "", 2),
            "coffee_signal_encoded":         _COFFEE_SIGNAL_MAP.get(r.signal, 0),
            "coffee_flowering_flag":         int(nat.stage == "flowering"),
            "coffee_drought_flag":           int(any("drought" in e for e in r.weather_events)),
        })

    if not rows:
        log.warning("No coffee reports to build features from — returning empty DataFrame")
        return pd.DataFrame()

    df = pd.DataFrame(rows).sort_values("week_ending").set_index("week_ending")
    return _to_daily(df, "coffee_llm")


def _to_daily(weekly: pd.DataFrame, tag: str) -> pd.DataFrame:
    """Resample weekly → daily using forward-fill. Aligns to W-MON anchor."""
    daily_idx = pd.date_range(f"{START_YEAR}-01-01", f"{END_YEAR}-12-31", freq="D")
    daily = weekly.reindex(daily_idx).ffill().bfill()
    daily.index.name = "Date"
    log.info("%s: %d daily rows, %d features", tag, len(daily), len(daily.columns))
    return daily


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def run_llm_pipeline(use_llm: bool = True, use_cache: bool = True) -> None:
    """
    Full 04b pipeline:
    1. Fetch USDA corn progress (QuickStats API) → corn features
    2. Fetch CONAB coffee reports (PDF + Claude API) → coffee features
    3. Save to data/raw/farming/llm_extracted/
    """
    os.makedirs("data/raw/farming", exist_ok=True)

    # ── Corn: USDA QuickStats ──────────────────────────────────────────────
    log.info("=== Stage 1: USDA corn progress ===")
    try:
        usda_pivot = fetch_usda_corn(use_cache=use_cache)
        corn_daily = build_corn_features(usda_pivot)
        corn_out = LLM_DIR / "corn_weekly_llm.csv"
        corn_daily.to_csv(corn_out)
        log.info("Corn features saved → %s (%d rows)", corn_out, len(corn_daily))
    except EnvironmentError as e:
        log.error("Corn stage skipped: %s", e)
        corn_daily = None

    # ── Coffee: CONAB + LLM ────────────────────────────────────────────────
    log.info("=== Stage 2: CONAB coffee reports ===")
    llm = None
    if use_llm and _LLM_KEY:
        try:
            llm = LLMExtractor()
            log.info("Claude API initialized: model=%s", MODEL)
        except Exception as e:
            log.error("LLM init failed: %s", e)
    else:
        log.warning("No LLM key found (set ANTHROPIC_AUTH_TOKEN or ANTHROPIC_API_KEY)")

    coffee_reports = fetch_conab_coffee(llm=llm, use_cache=use_cache)

    # Fallback: USDA PSD coffee Brazil data (always available, no scraping needed)
    if not coffee_reports:
        log.info("CONAB unavailable — falling back to USDA PSD coffee Brazil data")
        psd_path = Path("data/psd_coffee_brazil.csv")
        if not psd_path.exists():
            # Auto-generate from psd_alldata.csv if available
            alldata = Path("psd_alldata.csv")
            if alldata.exists():
                log.info("Generating psd_coffee_brazil.csv from psd_alldata.csv ...")
                raw_psd = pd.read_csv(alldata, dtype=str)
                mask = (
                    raw_psd["Commodity_Description"].str.contains("Coffee", case=False, na=False) &
                    raw_psd["Country_Name"].str.contains("Brazil", case=False, na=False)
                )
                raw_psd[mask].to_csv(psd_path, index=False)
                log.info("Saved %d rows → %s", mask.sum(), psd_path)
        coffee_reports = fetch_coffee_from_psd(psd_path=psd_path, llm=llm, use_cache=use_cache)

    if coffee_reports:
        coffee_daily = build_coffee_features(coffee_reports)
        coffee_out = LLM_DIR / "coffee_monthly_llm.csv"
        coffee_daily.to_csv(coffee_out)
        log.info("Coffee features saved → %s (%d rows)", coffee_out, len(coffee_daily))
    else:
        log.warning("No coffee data available — check psd_alldata.csv or CONAB connectivity")

    log.info("=== 04b complete: outputs in %s ===", LLM_DIR)


if __name__ == "__main__":
    run_llm_pipeline(use_llm=True, use_cache=True)
