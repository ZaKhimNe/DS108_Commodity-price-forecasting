import pandas as pd
import numpy as np
import os
import json
from datetime import timedelta

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

RAW_DIR = os.path.join(PROJECT_ROOT, "data", "raw")
PREPROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "preprocessed")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "data", "integrated")
os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_data():
    coffee_market_daily  = pd.read_csv(os.path.join(PREPROCESSED_DIR, "market", "daily_coffee_clean.csv"))
    coffee_market_weekly = pd.read_csv(os.path.join(PREPROCESSED_DIR, "market", "weekly_coffee_clean.csv"))
    corn_market_daily    = pd.read_csv(os.path.join(PREPROCESSED_DIR, "market", "daily_corn_clean.csv"))
    corn_market_weekly   = pd.read_csv(os.path.join(PREPROCESSED_DIR, "market", "weekly_corn_clean.csv"))
    
    # SỬA LỖI: Đọc dữ liệu thời tiết đã qua xử lý (Clean) thay vì Raw
    weather_clean_dir = os.path.join(PREPROCESSED_DIR, "weather", "weather_clean")
    weather_files = [f for f in os.listdir(weather_clean_dir) if f.endswith(".csv")]
    
    weather_data = {}
    for file in weather_files:
        name = file.replace("despiked_", "").replace("clean_", "").replace(".csv", "")
        df = pd.read_csv(os.path.join(weather_clean_dir, file))
        df['date'] = pd.to_datetime(df['date'], utc=True)
        weather_data[name] = df.set_index('date')
    
    usd_brl = pd.read_csv(os.path.join(PREPROCESSED_DIR, "macro", "weekly_usd_brl_clean.csv"))
    inflation = pd.read_csv(os.path.join(PREPROCESSED_DIR, "macro", "weekly_us_inflation_clean.csv"))
    coffee_cal = pd.read_csv(os.path.join(PREPROCESSED_DIR, "farming", "weekly_coffee_calendar.csv"))
    corn_cal = pd.read_csv(os.path.join(PREPROCESSED_DIR, "farming", "weekly_corn_calendar.csv"))

    vix_path = os.path.join(PREPROCESSED_DIR, "macro", "weekly_vix_clean.csv")
    vix_weekly = None
    if os.path.exists(vix_path):
        vix_weekly = pd.read_csv(vix_path)
        print(f"   VIX loaded: {len(vix_weekly)} rows")
    else:
        print("   [INFO] weekly_vix_clean.csv not found — VIX features will be skipped")

    return coffee_market_daily, coffee_market_weekly, corn_market_daily, corn_market_weekly, weather_data, usd_brl, inflation, coffee_cal, corn_cal, vix_weekly

def clean_market_data(df):
    df_clean = df.copy()
    if pd.isna(pd.to_numeric(df_clean.iloc[0]['Close'], errors='coerce')):
        df_clean = df_clean.iloc[1:].reset_index(drop=True)
    df_clean['Date'] = pd.to_datetime(df_clean['Date'], errors='coerce', utc=True)
    numeric_cols = ['Close', 'High', 'Low', 'Open', 'Volume']
    for col in numeric_cols:
        if col in df_clean.columns:
            df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
    return df_clean.dropna(subset=['Date']).sort_values('Date').reset_index(drop=True)


def apply_sma_smoothing(weather_df):
    """Khử nhiễu vặt (SMA 7 ngày) cho dữ liệu thời tiết (ngoại trừ temperature_2m_max đã được smooth ở preprocessing)"""
    df = weather_df.copy()
    # FIX: Loại bỏ 'temperature_2m_max' vì weather_preprocessing.py đã tính temp_max_rolling_7d
    # Tránh multicollinearity: temperature_2m_max smooth 7d ≈ temp_max_rolling_7d
    cols_to_smooth = ['precipitation_sum', 'et0_fao_evapotranspiration']
    for col in cols_to_smooth:
        if col in df.columns:
            df[col] = df[col].rolling(window=7, min_periods=1).mean()
    return df

def extract_wavelet_trend(series, span=30):
    return series.ewm(span=span, adjust=False).mean()

def weekend_aggregation(weather_df):
    df = weather_df.copy()
    df['is_weekend'] = df.index.dayofweek.isin([5, 6])
    
    weekend_agg = df[df['is_weekend']].resample('W-MON').agg({
        'temperature_2m_max': 'mean',
        'precipitation_sum': 'sum',
        'et0_fao_evapotranspiration': 'mean' if 'et0_fao_evapotranspiration' in df.columns else 'mean'
    }).rename(columns={
        'temperature_2m_max': 'weekend_temp_max',
        'precipitation_sum': 'weekend_precip_sum',
        'et0_fao_evapotranspiration': 'weekend_et0'
    })
    # FIX: Giữ index ở W-MON (Thứ 2) để weekend features sẵn có từ đầu tuần, không phải chờ ffill() đến Thứ 3
    # weekend_agg.index = weekend_agg.index - timedelta(days=1)  # [REMOVED - Gây Thứ 2 không có data]
    
    df = df.join(weekend_agg, how='left')
    df[['weekend_temp_max', 'weekend_precip_sum', 'weekend_et0']] = df[['weekend_temp_max', 'weekend_precip_sum', 'weekend_et0']].ffill()
    
    if 'temperature_2m_max' in df.columns:
        df['hot_day'] = (df['temperature_2m_max'] > 30).astype(int)
        df['cold_day'] = (df['temperature_2m_max'] < 10).astype(int)
    else:
        df['hot_day'] = 0
        df['cold_day'] = 0
        
    if 'precipitation_sum' in df.columns:
        df['dry_day'] = (df['precipitation_sum'] == 0).astype(int)
        df['dry_spell'] = df['dry_day'].groupby((df['dry_day'] != df['dry_day'].shift()).cumsum()).cumsum()
    else:
        df['dry_spell'] = 0
        
    return df

def build_integrated_dataset(market_df, weather_backbone, usd_brl, inflation, cal_df, timeframe='weekly', crop_type='coffee', target_threshold=None, vix_weekly=None):
    df = market_df.copy()
    
    cal_encoded = cal_df.copy()
    
    if timeframe == 'weekly':
        market_agg = {
            'Close': 'last', 'High': 'max', 'Low': 'min', 'Open': 'first', 'Volume': 'sum',
            'RSI_14': 'last', 'volatility_20d': 'mean',
            'SMA_20': 'last', 'SMA_50': 'last', 'EMA_20': 'last',
            'BB_upper': 'last', 'BB_lower': 'last',
            'MACD': 'last', 'MACD_signal': 'last', 'MACD_hist': 'last',
            'momentum_1w': 'last', 'momentum_1m': 'last',
            'Close_lag_1w': 'last', 'Close_lag_1m': 'last',
            'log_return_lag_1w': 'last', 'volatility_lag_1w': 'last',
        }
        agg_dict = {k: v for k, v in market_agg.items() if k in df.columns}
        df = df.resample('W-MON').agg(agg_dict)
        if not weather_backbone.empty:
            weather_agg = {
                'temperature_2m_max': 'mean', 'precipitation_sum': 'sum',
                'et0_fao_evapotranspiration': 'mean',
                'weekend_temp_max': 'mean', 'weekend_precip_sum': 'sum', 'weekend_et0': 'mean',
                'hot_day': 'sum', 'cold_day': 'sum', 'dry_day': 'sum', 'dry_spell': 'last',
            }
            w_agg = {k: v for k, v in weather_agg.items() if k in weather_backbone.columns}
            weather_backbone = weather_backbone.resample('W-MON').agg(w_agg)
            
    if not weather_backbone.empty:
        df = df.join(weather_backbone, how='left')
        
    df = df.merge(usd_brl.add_prefix('usd_'), left_index=True, right_index=True, how='left')
    df = df.merge(inflation.add_prefix('inf_'), left_index=True, right_index=True, how='left')
    df = df.merge(cal_encoded.add_prefix('cal_'), left_index=True, right_index=True, how='left')

    if vix_weekly is not None:
        _vix = vix_weekly.copy()
        _vix["Date"] = pd.to_datetime(_vix["Date"], utc=True).dt.normalize()
        _vix = _vix.set_index("Date")
        df = df.merge(_vix.add_prefix("vix_"), left_index=True, right_index=True, how="left")

    df = df.ffill()

    df['currency_adjusted_close'] = df['Close'] * df['usd_Close']
    df['inflation_pressure'] = df['usd_log_return_lag_1w'] * 100 - df['inf_CPI_MoM_pct']
    

    
    df['close_lag_1'] = df['Close'].shift(1)
    df['close_lag_2'] = df['Close'].shift(2)

    if not weather_backbone.empty and 'weekend_temp_max' in df.columns:
        lag_weeks = 34 if crop_type == 'coffee' else 9
        lag_steps = lag_weeks * 5 if timeframe == 'daily' else lag_weeks

        df['temp_bio_lag'] = df['weekend_temp_max'].shift(lag_steps)
        df['precip_bio_lag'] = df['weekend_precip_sum'].shift(lag_steps)
        df['temp_wavelet_trend'] = extract_wavelet_trend(df['weekend_temp_max'])
        df['precip_wavelet_trend'] = extract_wavelet_trend(df['weekend_precip_sum'])

    # CCF-based lags từ lag_analysis.py (Sprint 2) — shift_feature only
    _lag_cfg_path = os.path.join(PROJECT_ROOT, "models", "12_lag_analysis", f"lag_config_{crop_type}.json")
    if os.path.exists(_lag_cfg_path):
        try:
            with open(_lag_cfg_path) as _f:
                _lag_cfg = json.load(_f)
            for _var, _cfg in _lag_cfg.items():
                if _var not in df.columns:
                    continue
                for _lag_w_str, _action in _cfg.get("action", {}).items():
                    if _action != "shift_feature":
                        continue
                    _lag_w = int(_lag_w_str)
                    _lag_s = _lag_w * 5 if timeframe == "daily" else _lag_w
                    _feat  = f"{_var}_lag_{_lag_w}w"
                    if _feat not in df.columns:
                        df[_feat] = df[_var].shift(_lag_s)
        except Exception as _e:
            print(f"[!] Không đọc được lag_config_{crop_type}.json: {_e}")
    
    window = 14 if timeframe == 'daily' else 4 # Cửa sổ linh hoạt theo khung thời gian
    
    sma = df['currency_adjusted_close'].rolling(window=window).mean()
    std = df['currency_adjusted_close'].rolling(window=window).std()
    df['BB_upper_adj'] = sma + (std * 2)
    df['BB_lower_adj'] = sma - (std * 2)

    df['volatility'] = std
    df['EMA_adj'] = df['currency_adjusted_close'].ewm(span=window).mean()
    delta = df['currency_adjusted_close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss.replace(0, np.nan)
    df['RSI_adj'] = 100 - (100 / (1 + rs))

    # ── Phase 2: Realized Volatility ───────────────────────────────────────
    _log_ret   = np.log(df['currency_adjusted_close'] / df['currency_adjusted_close'].shift(1))
    _ann       = np.sqrt(252) if timeframe == 'daily' else np.sqrt(52)
    df['rv_5d']    = _log_ret.rolling(5).std()  * _ann
    df['rv_10d']   = _log_ret.rolling(10).std() * _ann
    df['rv_20d']   = _log_ret.rolling(20).std() * _ann
    df['rv_ratio'] = df['rv_5d'] / (df['rv_20d'] + 1e-8)

    # ── High-Low Range ──────────────────────────────────────────────────────
    if 'High' in df.columns and 'Low' in df.columns:
        df['hl_range_pct']  = (df['High'] - df['Low']) / (df['Close'].replace(0, np.nan))
        df['hl_range_ma5']  = df['hl_range_pct'].rolling(5).mean()
        df['hl_range_ma20'] = df['hl_range_pct'].rolling(20).mean()
        df['hl_expansion']  = df['hl_range_pct'] / (df['hl_range_ma20'] + 1e-8)

    # ── Volume features ─────────────────────────────────────────────────────
    if 'Volume' in df.columns:
        _vol_ma20 = df['Volume'].rolling(20).mean()
        df['volume_ratio']   = df['Volume'] / (_vol_ma20 + 1e-8)
        df['price_vol_corr'] = _log_ret.rolling(10).corr(df['Volume'].pct_change())

    # ── USD Volatility extensions ───────────────────────────────────────────
    if 'usd_Close' in df.columns:
        _usd_ret = np.log(df['usd_Close'] / df['usd_Close'].shift(1))
        df['usd_rv_5d']    = _usd_ret.rolling(5).std()  * _ann
        df['usd_rv_20d']   = _usd_ret.rolling(20).std() * _ann
        df['usd_rv_ratio'] = df['usd_rv_5d'] / (df['usd_rv_20d'] + 1e-8)

    shift_steps = -7 if timeframe == 'daily' else -1
    df['return_future'] = (df['currency_adjusted_close'].shift(shift_steps) / df['currency_adjusted_close']) - 1

    # target_threshold: 0.03 cho corn weekly (vol ~1.5%), 0.05 cho các dataset khác
    if target_threshold is None:
        target_threshold = 0.015 if (crop_type == 'corn' and timeframe == 'weekly') else 0.025

    df['target_binary'] = np.where(df['return_future'].notna(),
                                   (df['return_future'] > target_threshold).astype(float),
                                   np.nan)
    _TAU = 0.02
    df['target_soft'] = np.where(df['return_future'].notna(),
                                 1 / (1 + np.exp(-(df['return_future'] - target_threshold) / _TAU)),
                                 np.nan)
    df['target_reg'] = np.where(df['return_future'].notna(),
                                df['return_future'].clip(-0.30, 0.30),
                                np.nan)
    df['target'] = df['target_binary']   # backward compat

    # Soft multiclass: down (0) / flat (1) / up (2)
    _TAU_MC = 0.02
    _lo = -target_threshold
    _hi =  target_threshold
    _rf = df['return_future'].fillna(0)
    _p_down = 1 / (1 + np.exp( (_rf - _lo) / _TAU_MC))
    _p_up   = 1 / (1 + np.exp(-(_rf - _hi) / _TAU_MC))
    _p_flat = np.maximum(0.0, 1 - _p_down - _p_up)
    _tot    = _p_down + _p_flat + _p_up
    df['target_mc_down'] = np.where(df['return_future'].notna(), _p_down / _tot, np.nan)
    df['target_mc_flat'] = np.where(df['return_future'].notna(), _p_flat / _tot, np.nan)
    df['target_mc_up']   = np.where(df['return_future'].notna(), _p_up   / _tot, np.nan)
    _stacked = np.column_stack([_p_down.values, _p_flat.values, _p_up.values])
    df['target_multiclass'] = np.where(
        df['return_future'].notna(),
        np.argmax(_stacked, axis=1).astype(float),
        np.nan
    )
    empty_cols = df.columns[df.isna().all()].tolist()
    if len(empty_cols) > 0:
        print(f"\n[*] Cảnh báo {timeframe}: Phát hiện các cột lỗi 100% NaN, tiến hành loại bỏ để cứu dữ liệu: {empty_cols}")
        df = df.drop(columns=empty_cols)
        
    print(f"[*] Kích thước dữ liệu {timeframe} trước khi dropna: {df.shape}")
    df_final = df.dropna()
    print(f"[*] Kích thước dữ liệu {timeframe} sau khi dropna: {df_final.shape}")
    
    return df_final

if __name__ == "__main__":
    print("Loading data...")
    coffee_market_daily, coffee_market_weekly, corn_market_daily, corn_market_weekly, weather_data, usd_brl, inflation, coffee_cal, corn_cal, vix_weekly = load_data()

    print("Preprocessing individual datasets...")
    def prep_market(df):
        df = clean_market_data(df)
        df['Date'] = pd.to_datetime(df['Date'], utc=True).dt.normalize()
        df.set_index('Date', inplace=True)
        return df

    coffee_market_d = prep_market(coffee_market_daily)
    coffee_market_w = prep_market(coffee_market_weekly)
    corn_market_d   = prep_market(corn_market_daily)
    corn_market_w   = prep_market(corn_market_weekly)
    
    for df in [usd_brl, inflation, coffee_cal, corn_cal]:
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'], errors='coerce', utc=True).dt.normalize()
            df.set_index('Date', inplace=True)

    processed_weather = {}
    for name, df in weather_data.items():
        df.index = pd.to_datetime(df.index, utc=True).normalize()
        df_smoothed = apply_sma_smoothing(df)
        processed_weather[name] = weekend_aggregation(df_smoothed)
    
    coffee_frames = [df for k, df in processed_weather.items() if k.startswith('coffee_')]
    corn_frames   = [df for k, df in processed_weather.items() if k.startswith('corn_')]
    coffee_backbone = pd.concat(coffee_frames).groupby(level=0).mean() if coffee_frames else pd.DataFrame()
    corn_backbone   = pd.concat(corn_frames).groupby(level=0).mean()   if corn_frames   else pd.DataFrame()
    print(f"Coffee backbone: {len(coffee_frames)} vùng, Corn backbone: {len(corn_frames)} vùng.")

    print("\n--- Xây dựng luồng Daily ---")
    coffee_daily = build_integrated_dataset(coffee_market_d, coffee_backbone, usd_brl, inflation, coffee_cal, timeframe='daily', crop_type='coffee', vix_weekly=vix_weekly)
    corn_daily = build_integrated_dataset(corn_market_d, corn_backbone, usd_brl, inflation, corn_cal, timeframe='daily', crop_type='corn', vix_weekly=vix_weekly)

    coffee_daily.to_csv(os.path.join(OUTPUT_DIR, "integrated_coffee_daily.csv"))
    corn_daily.to_csv(os.path.join(OUTPUT_DIR, "integrated_corn_daily.csv"))

    print("\n--- Xây dựng luồng Weekly ---")
    coffee_weekly = build_integrated_dataset(coffee_market_w, coffee_backbone, usd_brl, inflation, coffee_cal, timeframe='weekly', crop_type='coffee', vix_weekly=vix_weekly)
    corn_weekly = build_integrated_dataset(corn_market_w, corn_backbone, usd_brl, inflation, corn_cal, timeframe='weekly', crop_type='corn', vix_weekly=vix_weekly)

    coffee_weekly.to_csv(os.path.join(OUTPUT_DIR, "integrated_coffee_weekly.csv"))
    corn_weekly.to_csv(os.path.join(OUTPUT_DIR, "integrated_corn_weekly.csv"))

    print("\n[V] Đã hoàn tất xử lý, tạo Cờ mùa vụ, cắt tỉa cột đa cộng tuyến và lưu toàn bộ dữ liệu!")