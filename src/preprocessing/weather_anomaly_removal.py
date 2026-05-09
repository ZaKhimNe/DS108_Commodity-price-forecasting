import pandas as pd
import numpy as np
import os
import pywt  

raw_weather_path = "data/raw/weather"
despiked_path = "data/preprocessed/weather/weather_despiked"
os.makedirs(despiked_path, exist_ok=True)

def apply_miqr(df, column_name, window=15, k=3.0):
    if column_name not in df.columns:
        return df

    Q1 = df[column_name].rolling(window=window, center=True, min_periods=1).quantile(0.25)
    Q3 = df[column_name].rolling(window=window, center=True, min_periods=1).quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - k * IQR
    upper_bound = Q3 + k * IQR
    
    is_anomaly = (df[column_name] < lower_bound) | (df[column_name] > upper_bound)
    
    df.loc[is_anomaly, column_name] = np.nan
    
    num_anomalies = is_anomaly.sum()
    if num_anomalies > 0:
        print(f"   -> Cột {column_name}: Đã chuyển {num_anomalies} điểm nhiễu thành NaN.")
        
    return df

def apply_wavelet_flatline_detection(df, column_name, width=5, dead_days_threshold=3):
    if column_name not in df.columns or df[column_name].isnull().all():
        return df

    temp_series = pd.to_numeric(df[column_name], errors='coerce').ffill().bfill()
    
    cwt_matrix, freqs = pywt.cwt(temp_series.to_numpy(), scales=[width], wavelet='mexh')
    wavelet_response = np.abs(cwt_matrix[0])

    is_dead = wavelet_response < 1e-4 
    dead_series = pd.Series(is_dead, index=df.index).astype(int)
    is_prolonged_dead = dead_series.rolling(window=dead_days_threshold, min_periods=1).sum() >= dead_days_threshold
    
    num_dead_points = is_prolonged_dead.sum()
    if num_dead_points > 0:
        print(f"   -> Cột {column_name}: Tìm thấy {num_dead_points} điểm kẹt cảm biến (Flatline) bằng Wavelet.")
        df[column_name] = np.where(is_prolonged_dead, np.nan, df[column_name])
    return df

print("BẮT ĐẦU QUÉT VÀ LOẠI BỎ NHIỄU (MIQR + WAVELET)...")

for file_name in os.listdir(raw_weather_path):
    if file_name.endswith(".csv"):
        print(f"\nĐang xử lý: {file_name}")
        df = pd.read_csv(os.path.join(raw_weather_path, file_name))
        continuous_cols = ['temperature_2m_max', 'temperature_2m_min', 'et0_fao_evapotranspiration', 'vpd_max']
        
        
        
        for col in continuous_cols:
            df = apply_miqr(df, col, window=15, k=3.0)
            df = apply_wavelet_flatline_detection(df, col, width=5, dead_days_threshold=4)
        
        save_path = os.path.join(despiked_path, f"despiked_{file_name}")
        df.to_csv(save_path, index=False)

print("\nHOÀN THÀNH BƯỚC LOẠI BỎ NHIỄU!")

