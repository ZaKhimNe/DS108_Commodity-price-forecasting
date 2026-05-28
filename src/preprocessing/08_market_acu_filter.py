import pandas as pd
import numpy as np
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))

INPUT_DIR = os.path.join(PROJECT_ROOT, "data", "raw", "market") 
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "data", "preprocessed", "market")

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

coffee_input        = os.path.join(INPUT_DIR,  'coffee_market.csv')
coffee_daily_output = os.path.join(OUTPUT_DIR, 'daily_coffee_clean.csv')
coffee_output       = os.path.join(OUTPUT_DIR, 'weekly_coffee_clean.csv')

corn_input        = os.path.join(INPUT_DIR,  'corn_market.csv')
corn_daily_output = os.path.join(OUTPUT_DIR, 'daily_corn_clean.csv')
corn_output       = os.path.join(OUTPUT_DIR, 'weekly_corn_clean.csv')

def clean_raw_market_data(df):
    df_clean = df.copy()
    # yfinance mới xuất cột tên 'index' thay vì 'Date'
    if 'index' in df_clean.columns and 'Date' not in df_clean.columns:
        df_clean = df_clean.rename(columns={'index': 'Date'})
    if pd.isna(pd.to_numeric(df_clean.iloc[0]['Close'], errors='coerce')):
        df_clean = df_clean.iloc[1:].reset_index(drop=True)
    # Áp dụng utc=True để đồng bộ timezone
    df_clean['Date'] = pd.to_datetime(df_clean['Date'], utc=True)
    numeric_cols = ['Close', 'High', 'Low', 'Open', 'Volume']
    for col in numeric_cols:
        df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
    return df_clean.sort_values('Date').reset_index(drop=True)

def apply_acu_filter(df, column='Close', threshold=0.05):
    df_clean = df.copy()
    
    diff_prev = df_clean[column].diff()
    diff_next = df_clean[column].shift(-1) - df_clean[column]
    
    pct_prev = diff_prev / df_clean[column].shift(1)
    pct_next = diff_next / df_clean[column]
    
    anomaly_mask = (
        (pct_prev.abs() > threshold) & 
        (pct_next.abs() > threshold) & 
        (diff_prev * diff_next < 0) 
    )
    
    df_clean.loc[anomaly_mask, column] = np.nan
    df_clean[column] = df_clean[column].interpolate(method='linear')
    
    anomaly_count = anomaly_mask.sum()
    print(f"[{column}] Đã xử lý {anomaly_count} điểm lỗi API/Flash Crash.")
    return df_clean, anomaly_count 

def calculate_financial_features(df):
    df_feat = df.copy()
    
    df_feat['log_return'] = np.log(df_feat['Close'] / df_feat['Close'].shift(1))
    
    delta = df_feat['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df_feat['RSI_14'] = 100 - (100 / (1 + rs))
    df_feat['volatility_20d'] = df_feat['log_return'].rolling(window=20).std() * np.sqrt(252)

    # Thêm Moving Averages
    df_feat['SMA_20'] = df_feat['Close'].rolling(window=20).mean()
    df_feat['SMA_50'] = df_feat['Close'].rolling(window=50).mean()
    df_feat['EMA_20'] = df_feat['Close'].ewm(span=20).mean()

    # Bollinger Bands
    df_feat['BB_middle'] = df_feat['SMA_20']
    df_feat['BB_upper'] = df_feat['BB_middle'] + 2 * df_feat['Close'].rolling(window=20).std()
    df_feat['BB_lower'] = df_feat['BB_middle'] - 2 * df_feat['Close'].rolling(window=20).std()
    
    # MACD
    ema_12 = df_feat['Close'].ewm(span=12).mean()
    ema_26 = df_feat['Close'].ewm(span=26).mean()
    df_feat['MACD'] = ema_12 - ema_26
    df_feat['MACD_signal'] = df_feat['MACD'].ewm(span=9).mean()
    df_feat['MACD_hist'] = df_feat['MACD'] - df_feat['MACD_signal']
    
    # Momentum
    df_feat['momentum_1w'] = df_feat['Close'] / df_feat['Close'].shift(5) - 1
    df_feat['momentum_1m'] = df_feat['Close'] / df_feat['Close'].shift(21) - 1
    
    return df_feat.dropna()

def resample_market_to_weekly(df, date_col='Date'):
    if date_col in df.columns:
        # Đồng bộ timezone tại bước resample
        df[date_col] = pd.to_datetime(df[date_col], utc=True) 
        df = df.set_index(date_col)
        
    ohlcv_dict = {
        'Open': 'first',
        'High': 'max',
        'Low': 'min',
        'Close': 'last',
        'Volume': 'sum',
        'RSI_14': 'last',         
        'volatility_20d': 'mean',
        'SMA_20': 'last',
        'SMA_50': 'last',
        'EMA_20': 'last',
        'BB_middle': 'last',
        'BB_upper': 'last',
        'BB_lower': 'last',
        'MACD': 'last',
        'MACD_signal': 'last',
        'MACD_hist': 'last',
        'momentum_1w': 'last',
        'momentum_1m': 'last'
    }
    
    agg_dict = {k: v for k, v in ohlcv_dict.items() if k in df.columns}
    
    weekly_df = df.resample('W-MON').agg(agg_dict).dropna()
    
    return weekly_df.reset_index()

if __name__ == "__main__":
    # === CHẠY THỬ PIPELINE ===
    print("Đang xử lý dữ liệu Cà Phê...")
    if os.path.exists(coffee_input):
        coffee_df = pd.read_csv(coffee_input)
        coffee_df = clean_raw_market_data(coffee_df)
        coffee_df, kc_c = apply_acu_filter(coffee_df, 'Close', 0.06)
        coffee_df, kc_h = apply_acu_filter(coffee_df, 'High', 0.06)
        coffee_df, kc_l = apply_acu_filter(coffee_df, 'Low', 0.06)
        coffee_df['High'] = np.maximum(coffee_df['High'], coffee_df['Close'])
        coffee_df['Low']  = np.minimum(coffee_df['Low'],  coffee_df['Close'])
        coffee_df_feat = calculate_financial_features(coffee_df)
        coffee_df_feat.to_csv(coffee_daily_output, index=False)
        coffee_weekly = resample_market_to_weekly(coffee_df_feat)
        coffee_weekly.to_csv(coffee_output, index=False)
        print(f"-> Hoàn tất! Đã lưu daily: {coffee_daily_output}, weekly: {coffee_output}")
    else:
        print(f"Không tìm thấy file: {coffee_input}")

    print("\nĐang xử lý dữ liệu Bắp (Ngô)...")
    if os.path.exists(corn_input):
        corn_df = pd.read_csv(corn_input)
        corn_df = clean_raw_market_data(corn_df)
        corn_df, zc_c = apply_acu_filter(corn_df, 'Close', 0.05)
        corn_df, zc_h = apply_acu_filter(corn_df, 'High', 0.05)
        corn_df, zc_l = apply_acu_filter(corn_df, 'Low', 0.05)
        corn_df['High'] = np.maximum(corn_df['High'], corn_df['Close'])
        corn_df['Low']  = np.minimum(corn_df['Low'],  corn_df['Close'])
        corn_df_feat = calculate_financial_features(corn_df)
        corn_df_feat.to_csv(corn_daily_output, index=False)
        corn_weekly = resample_market_to_weekly(corn_df_feat)
        corn_weekly.to_csv(corn_output, index=False)
        print(f"-> Hoàn tất! Đã lưu daily: {corn_daily_output}, weekly: {corn_output}")
    else:
        print(f"Không tìm thấy file: {corn_input}")