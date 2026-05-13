import pandas as pd
import numpy as np
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))

INPUT_DIR = os.path.join(PROJECT_ROOT, "data", "raw", "macro")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "data", "preprocessed", "macro")

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

def clean_raw_market_data(df):
    df_clean = df.drop(0).copy()
    # Áp dụng utc=True để đồng bộ với các module khác
    df_clean['Date'] = pd.to_datetime(df_clean['Date'], utc=True)
    numeric_cols = ['Close', 'High', 'Low', 'Open', 'Volume']
    for col in numeric_cols:
        if col in df_clean.columns:
            df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
    return df_clean.sort_values('Date').reset_index(drop=True)

def apply_acu_filter(df, column='Close', threshold=0.03):
    df_clean = df.copy()
    diff_prev = df_clean[column].diff()
    diff_next = df_clean[column].shift(-1) - df_clean[column]
    pct_prev = diff_prev / df_clean[column].shift(1)
    pct_next = diff_next / df_clean[column]
    
    anomaly_mask = ((pct_prev.abs() > threshold) & 
                    (pct_next.abs() > threshold) & 
                    (diff_prev * diff_next < 0))
    df_clean.loc[anomaly_mask, column] = np.nan
    df_clean[column] = df_clean[column].interpolate(method='linear')
    return df_clean

def calculate_financial_features(df):
    df_feat = df.copy()
    df_feat['log_return'] = np.log(df_feat['Close'] / df_feat['Close'].shift(1))
    delta = df_feat['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss.replace(0, np.nan)
    df_feat['RSI_14'] = 100 - (100 / (1 + rs))
    df_feat['RSI_14'] = df_feat['RSI_14'].fillna(100) 
    df_feat['volatility_20d'] = df_feat['log_return'].rolling(window=20).std() * np.sqrt(252)
    return df_feat.dropna()

def resample_market_to_weekly(df, date_col='Date'):
    if date_col in df.columns:
        df = df.set_index(date_col)
    
    # Tích hợp danh sách agg_dict bao gồm cả các Lag Features
    ohlcv_dict = {
        'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 
        'Volume': 'sum', 'RSI_14': 'last', 'volatility_20d': 'mean',
        'Close_lag_1w': 'last', 'Close_lag_1m': 'last', 
        'log_return_lag_1w': 'last', 'volatility_lag_1w': 'last'
    }
    agg_dict = {k: v for k, v in ohlcv_dict.items() if k in df.columns}
    return df.resample('W').agg(agg_dict).dropna().reset_index()

if __name__ == "__main__":
    usd_brl_path = os.path.join(INPUT_DIR, 'usd_brl_exchange.csv')
    if os.path.exists(usd_brl_path):
        print("Đang tiền xử lý Tỷ giá USD/BRL...")
        usd_brl_df = pd.read_csv(usd_brl_path)
        usd_brl_df = clean_raw_market_data(usd_brl_df)
        
        usd_brl_df = apply_acu_filter(usd_brl_df, 'Close', 0.03)
        usd_brl_df = apply_acu_filter(usd_brl_df, 'High', 0.03)
        usd_brl_df = apply_acu_filter(usd_brl_df, 'Low', 0.03)

        usd_brl_df = calculate_financial_features(usd_brl_df)
        
        # Thêm lag features
        usd_brl_df['Close_lag_1w'] = usd_brl_df['Close'].shift(7)
        usd_brl_df['Close_lag_1m'] = usd_brl_df['Close'].shift(30)
        usd_brl_df['log_return_lag_1w'] = usd_brl_df['log_return'].shift(7)
        usd_brl_df['volatility_lag_1w'] = usd_brl_df['volatility_20d'].shift(7)
        
        usd_brl_weekly = resample_market_to_weekly(usd_brl_df)
        usd_brl_weekly.to_csv(os.path.join(OUTPUT_DIR, 'weekly_usd_brl_clean.csv'), index=False)
        print("-> Đã lưu: weekly_usd_brl_clean.csv")

    cpi_path = os.path.join(INPUT_DIR, 'us_inflation.csv')
    if os.path.exists(cpi_path):
        print("Đang tiền xử lý Lạm phát Mỹ (US CPI)...")
        cpi_df = pd.read_csv(cpi_path)
        
        # Đồng bộ timezone
        cpi_df['Date'] = pd.to_datetime(cpi_df['Date'], utc=True)
        cpi_df = cpi_df.sort_values('Date').reset_index(drop=True)

        # Trải phẳng ra daily index và Forward-Fill
        start_date = cpi_df['Date'].min()
        end_date = cpi_df['Date'].max()
        daily_index = pd.date_range(start=start_date, end=end_date, freq='D')

        cpi_df = cpi_df.set_index('Date')
        cpi_df = cpi_df.reindex(daily_index)
        cpi_df['US_CPI'] = cpi_df['US_CPI'].ffill()
        cpi_df = cpi_df.reset_index().rename(columns={'index': 'Date'})

        # Tính MoM và YoY
        cpi_df['CPI_MoM_pct'] = cpi_df['US_CPI'].pct_change(periods=30) * 100 
        cpi_df['CPI_YoY_pct'] = cpi_df['US_CPI'].pct_change(periods=365) * 100 
        
        cpi_df.to_csv(os.path.join(OUTPUT_DIR, 'daily_us_inflation_clean.csv'), index=False)
        print("-> Đã lưu: daily_us_inflation_clean.csv")

        # Nén xuống Weekly
        cpi_weekly = cpi_df.set_index('Date').resample('W').last().dropna().reset_index()
        cpi_weekly.to_csv(os.path.join(OUTPUT_DIR, 'weekly_us_inflation_clean.csv'), index=False)
        print("-> Đã lưu: weekly_us_inflation_clean.csv")

    print("Hoàn tất tiền xử lý nhóm Macro!")