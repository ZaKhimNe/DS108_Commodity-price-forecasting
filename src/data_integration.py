import pandas as pd
import numpy as np
import os
import warnings
from datetime import timedelta
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant
import pywt
import lightgbm as lgb
from sklearn.model_selection import TimeSeriesSplit

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

RAW_DIR = os.path.join(PROJECT_ROOT, "data", "raw")
PREPROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "preprocessed")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "data", "integrated")
os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_data():
    coffee_market = pd.read_csv(os.path.join(RAW_DIR, "market", "coffee_market.csv"))
    corn_market = pd.read_csv(os.path.join(RAW_DIR, "market", "corn_market.csv"))
    
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
    
    return coffee_market, corn_market, weather_data, usd_brl, inflation, coffee_cal, corn_cal

def clean_market_data(df):
    df_clean = df.drop(0).copy()
    # SỬA LỖI: Đồng bộ utc=True
    df_clean['Date'] = pd.to_datetime(df_clean['Date'], errors='coerce', utc=True)
    numeric_cols = ['Close', 'High', 'Low', 'Open', 'Volume']
    for col in numeric_cols:
        if col in df_clean.columns:
            df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
    return df_clean.dropna(subset=['Date']).sort_values('Date').reset_index(drop=True)

def encode_seasonal_calendar(cal_df):
    """Biến đổi các cột chứa text trong lịch mùa vụ thành cờ nhị phân (One-hot encoding)"""
    df = cal_df.copy()
    cat_cols = df.select_dtypes(include=['object', 'category']).columns
    if len(cat_cols) > 0:
        df = pd.get_dummies(df, columns=cat_cols, dummy_na=False, dtype=int)
    return df


def apply_sma_smoothing(weather_df):
    """Khử nhiễu vặt (SMA 7 ngày) cho dữ liệu thời tiết"""
    df = weather_df.copy()
    cols_to_smooth = ['temperature_2m_max', 'precipitation_sum', 'et0_fao_evapotranspiration']
    for col in cols_to_smooth:
        if col in df.columns:
            df[col] = df[col].rolling(window=7, min_periods=1).mean()
    return df

def extract_wavelet_trend(series, wavelet='db4', level=3):
    """Trích xuất xu hướng khí hậu dài hạn bằng biến đổi Wavelet (DWT)"""
    s = series.ffill().bfill().fillna(0).values.copy()
    if len(s) < 2**level:
        return s 
    
    coeffs = pywt.wavedec(s, wavelet, level=level)
    coeffs[1:] = [np.zeros_like(v) for v in coeffs[1:]]
    trend = pywt.waverec(coeffs, wavelet)
    
    return trend[:len(s)]

def lgbm_null_importance_selection(df, name, target_col='target', iv_threshold=0.3):
    """Sàng lọc Đặc trưng bằng LightGBM Null Importances & TimeSeriesSplit"""
    print(f"\n[+] Đang chạy LightGBM Null Importances cho {name}...")
    df_model = df.copy()
    
    protected_cols = [target_col, 'return_future', 'Close', 'currency_adjusted_close']
    features = [c for c in df_model.select_dtypes(include=[np.number]).columns if c not in protected_cols]
    
    X = df_model[features].replace([np.inf, -np.inf], np.nan).fillna(0)
    y = df_model[target_col]
    
    tscv = TimeSeriesSplit(n_splits=3)
    
    actual_imp = np.zeros(len(features))
    for train_idx, val_idx in tscv.split(X):
        model = lgb.LGBMClassifier(n_estimators=50, random_state=42, verbose=-1)
        model.fit(X.iloc[train_idx], y.iloc[train_idx])
        actual_imp += model.feature_importances_ / tscv.n_splits
        
    null_runs = 5 
    null_imp = np.zeros(len(features))
    for _ in range(null_runs):
        y_shuffled = np.random.permutation(y)
        fold_null = np.zeros(len(features))
        for train_idx, val_idx in tscv.split(X):
            model = lgb.LGBMClassifier(n_estimators=50, random_state=42, verbose=-1)
            model.fit(X.iloc[train_idx], y_shuffled[train_idx])
            fold_null += model.feature_importances_ / tscv.n_splits
        null_imp += fold_null / null_runs
        
    max_imp = actual_imp.max() + 1e-5
    iv_scores = np.maximum(0, actual_imp - null_imp) / max_imp
    
    imp_df = pd.DataFrame({'feature': features, 'IV_score': iv_scores})
    kept_features = imp_df[imp_df['IV_score'] >= iv_threshold]['feature'].tolist()
    
    final_cols = [c for c in df_model.columns if c in protected_cols + kept_features]
    
    print(f"    -> Đã giữ lại {len(kept_features)}/{len(features)} đặc trưng (IV >= {iv_threshold}).")
    return df_model[final_cols]

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
    weekend_agg.index = weekend_agg.index + timedelta(days=1)
    
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

def build_integrated_dataset(market_df, weather_backbone, usd_brl, inflation, cal_df, timeframe='weekly', crop_type='coffee'):
    df = market_df.copy()
    
    cal_encoded = encode_seasonal_calendar(cal_df)
    
    if timeframe == 'weekly':
        agg_dict = {'Close': 'last', 'High': 'max', 'Low': 'min', 'Open': 'first', 'Volume': 'sum'}
        df = df.resample('W-MON').agg(agg_dict)
        if not weather_backbone.empty:
            weather_backbone = weather_backbone.resample('W-MON').mean() # Đơn giản hóa, bạn có thể custom thêm
            
    if not weather_backbone.empty:
        df = df.join(weather_backbone, how='left')
        
    df = df.merge(usd_brl.add_prefix('usd_'), left_index=True, right_index=True, how='left')
    df = df.merge(inflation.add_prefix('inf_'), left_index=True, right_index=True, how='left')
    df = df.merge(cal_encoded.add_prefix('cal_'), left_index=True, right_index=True, how='left')
    
    df = df.ffill()

    df['currency_adjusted_close'] = df['Close'] / df['usd_Close']
    df['usd_log_return'] = np.log(df['usd_Close'] / df['usd_Close'].shift(1))
    df['inflation_pressure'] = df['usd_log_return'] - df['inf_CPI_MoM_pct']
    
    df['week_of_year'] = df.index.isocalendar().week
    df['week_sin'] = np.sin(2 * np.pi * df['week_of_year'] / 52)
    df['week_cos'] = np.cos(2 * np.pi * df['week_of_year'] / 52)
    
    df['month'] = df.index.month
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    
    df['close_lag_1'] = df['Close'].shift(1)
    df['close_lag_2'] = df['Close'].shift(2)

    if not weather_backbone.empty and 'weekend_temp_max' in df.columns:
        if crop_type == 'coffee':
            lag_steps = 240 if timeframe == 'daily' else 34
        elif crop_type == 'corn':
            lag_steps = 63 if timeframe == 'daily' else 9
            
        df['temp_bio_lag'] = df['weekend_temp_max'].shift(lag_steps)
        df['precip_bio_lag'] = df['weekend_precip_sum'].shift(lag_steps)
        df['temp_wavelet_trend'] = extract_wavelet_trend(df['weekend_temp_max'])
        df['precip_wavelet_trend'] = extract_wavelet_trend(df['weekend_precip_sum'])
    
    window = 14 if timeframe == 'daily' else 4 # Cửa sổ linh hoạt theo khung thời gian
    
    sma = df['currency_adjusted_close'].rolling(window=window).mean()
    std = df['currency_adjusted_close'].rolling(window=window).std()
    df['BB_upper'] = sma + (std * 2)
    df['BB_lower'] = sma - (std * 2)
    
    df['volatility'] = std
    df['EMA_adj'] = df['currency_adjusted_close'].ewm(span=window).mean()
    delta = df['currency_adjusted_close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    df['RSI_adj'] = 100 - (100 / (1 + rs))
    

    shift_steps = -7 if timeframe == 'daily' else -1
    df['return_future'] = (df['currency_adjusted_close'].shift(shift_steps) / df['currency_adjusted_close']) - 1
    df['target'] = (df['return_future'] > 0.05).astype(int)
    df = df.bfill()
    empty_cols = df.columns[df.isna().all()].tolist()
    if len(empty_cols) > 0:
        print(f"\n[*] Cảnh báo {timeframe}: Phát hiện các cột lỗi 100% NaN, tiến hành loại bỏ để cứu dữ liệu: {empty_cols}")
        df = df.drop(columns=empty_cols)
        
    print(f"[*] Kích thước dữ liệu {timeframe} trước khi dropna: {df.shape}")
    df_final = df.dropna()
    print(f"[*] Kích thước dữ liệu {timeframe} sau khi dropna: {df_final.shape}")
    
    return df_final

def feature_selection(df, name):
    df_cleaned = df.copy()
    numeric_cols = df_cleaned.select_dtypes(include=[np.number]).columns
    # Cấm drop các cột nhãn và thời gian cốt lõi
    protected_cols = ['target', 'return_future', 'Close', 'currency_adjusted_close']
    features_to_check = [col for col in numeric_cols if col not in protected_cols]
    
    # 1. Xử lý Tương quan cao (>0.85)
    corr_matrix = df_cleaned[features_to_check].corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    
    # Tìm các cột nằm ở tam giác trên có tương quan > 0.85 để drop
    to_drop_corr = [column for column in upper.columns if any(upper[column] > 0.85)]
    print(f"\n[+] {name}: Tự động drop {len(to_drop_corr)} cột do tương quan chéo > 0.85.")
    df_cleaned.drop(columns=to_drop_corr, inplace=True, errors='ignore')
    
    # 2. Xử lý VIF (>5)
    features_to_check = [col for col in df_cleaned.columns if col in features_to_check and col not in to_drop_corr]
    df_numeric = df_cleaned[features_to_check].replace([np.inf, -np.inf], np.nan).fillna(0).astype(float)
    df_numeric = df_numeric.loc[:, df_numeric.var() > 0] # Bỏ các cột có phương sai = 0
    
    df_vif = add_constant(df_numeric)
    
    if len(df_numeric.columns) > 1:
        vif_data = pd.DataFrame()
        vif_data["feature"] = df_vif.columns
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            vif_data["VIF"] = [variance_inflation_factor(df_vif.values, i) for i in range(df_vif.shape[1])]
        
        # Bọc lọc VIF > 5
        to_drop_vif = vif_data[(vif_data['VIF'] > 5) & (vif_data['feature'] != 'const')]['feature'].tolist()
        print(f"[+] {name}: Tự động drop {len(to_drop_vif)} cột do VIF > 5.")
        df_cleaned.drop(columns=to_drop_vif, inplace=True, errors='ignore')

    return df_cleaned

if __name__ == "__main__":
    print("Loading data...")
    coffee_market, corn_market, weather_data, usd_brl, inflation, coffee_cal, corn_cal = load_data()
    
    print("Preprocessing individual datasets...")
    coffee_market = clean_market_data(coffee_market)
    coffee_market['Date'] = pd.to_datetime(coffee_market['Date']).dt.normalize()
    coffee_market.set_index('Date', inplace=True)

    corn_market = clean_market_data(corn_market)
    corn_market['Date'] = pd.to_datetime(corn_market['Date']).dt.normalize()
    corn_market.set_index('Date', inplace=True)
    
    for df in [usd_brl, inflation, coffee_cal, corn_cal]:
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'], errors='coerce', utc=True).dt.normalize()
            df.set_index('Date', inplace=True)

    processed_weather = {}
    for name, df in weather_data.items():
        df.index = pd.to_datetime(df.index, utc=True).normalize()
        df_smoothed = apply_sma_smoothing(df)
        processed_weather[name] = weekend_aggregation(df_smoothed)
    
    backbone_key = 'coffee_Cerrado_Baiano' 
    if backbone_key not in processed_weather and len(processed_weather) > 0:
        old_key = backbone_key
        backbone_key = list(processed_weather.keys())[0] 
        print(f"Cảnh báo: Không tìm thấy '{old_key}', tự động sử dụng '{backbone_key}' làm backbone.")
        
    weather_backbone = processed_weather.get(backbone_key, pd.DataFrame())

    print("\n--- Xây dựng luồng Daily ---")
    coffee_daily = build_integrated_dataset(coffee_market, weather_backbone, usd_brl, inflation, coffee_cal, timeframe='daily', crop_type='coffee')
    corn_daily = build_integrated_dataset(corn_market, weather_backbone, usd_brl, inflation, corn_cal, timeframe='daily', crop_type='corn')
    
    print("Sàng lọc đặc trưng Daily...")
    coffee_daily_clean = feature_selection(coffee_daily, "Coffee_Daily")
    corn_daily_clean = feature_selection(corn_daily, "Corn_Daily")
    coffee_daily_final = lgbm_null_importance_selection(coffee_daily_clean, "Coffee_Daily", iv_threshold=0)
    corn_daily_final = lgbm_null_importance_selection(corn_daily_clean, "Corn_Daily", iv_threshold=0)

    coffee_daily_final.to_csv(os.path.join(OUTPUT_DIR, "integrated_coffee_daily.csv"))
    corn_daily_final.to_csv(os.path.join(OUTPUT_DIR, "integrated_corn_daily.csv"))

    print("\n--- Xây dựng luồng Weekly ---")
    coffee_weekly = build_integrated_dataset(coffee_market, weather_backbone, usd_brl, inflation, coffee_cal, timeframe='weekly', crop_type='coffee')
    corn_weekly = build_integrated_dataset(corn_market, weather_backbone, usd_brl, inflation, corn_cal, timeframe='weekly', crop_type='corn')

    print("Sàng lọc đặc trưng Weekly...")
    coffee_weekly_clean = feature_selection(coffee_weekly, "Coffee_Weekly")
    corn_weekly_clean = feature_selection(corn_weekly, "Corn_Weekly")
    coffee_weekly_final = lgbm_null_importance_selection(coffee_weekly_clean, "Coffee_Weekly", iv_threshold=0)
    corn_weekly_final = lgbm_null_importance_selection(corn_weekly_clean, "Corn_Weekly", iv_threshold=0)
    
    coffee_weekly_final.to_csv(os.path.join(OUTPUT_DIR, "integrated_coffee_weekly.csv"))
    corn_weekly_final.to_csv(os.path.join(OUTPUT_DIR, "integrated_corn_weekly.csv"))

    print("\n[V] Đã hoàn tất xử lý, tạo Cờ mùa vụ, cắt tỉa cột đa cộng tuyến và lưu toàn bộ dữ liệu!")