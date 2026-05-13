import pandas as pd
import numpy as np
import os
import pywt  

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))

raw_weather_path = os.path.join(PROJECT_ROOT, "data", "raw", "weather")
despiked_path = os.path.join(PROJECT_ROOT, "data", "preprocessed", "weather", "weather_despiked")
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
        print(f"   -> Cột {column_name}: Đã chuyển {num_anomalies} điểm nhiễu đột biến (Spike) thành NaN.")
        
    return df

def apply_wavelet_flatline_detection(df, column_name, width=5, dead_days_threshold=3):
    if column_name not in df.columns or df[column_name].isnull().all():
        return df

    # Lấp đầy tạm thời để thuật toán Wavelet chạy liên tục (không bị đứt gãy do NaN)
    temp_series = pd.to_numeric(df[column_name], errors='coerce').ffill().bfill()
    
    # Biến đổi CWT bằng Mexican Hat Wavelet
    cwt_matrix, freqs = pywt.cwt(temp_series.to_numpy(), scales=[width], wavelet='mexh')
    wavelet_response = np.abs(cwt_matrix[0])

    # Xác định các điểm mà tín hiệu gần như đi ngang (phản hồi cực thấp)
    is_dead = wavelet_response < 1e-4 
    dead_series = pd.Series(is_dead, index=df.index).astype(int)
    
    # Kiểm tra xem cảm biến có chết trong nhiều ngày liên tiếp hay không
    is_prolonged_dead = dead_series.rolling(window=dead_days_threshold, min_periods=1).sum() >= dead_days_threshold
    
    num_dead_points = is_prolonged_dead.sum()
    if num_dead_points > 0:
        print(f"   -> Cột {column_name}: Tìm thấy {num_dead_points} điểm kẹt cảm biến (Flatline) bằng Wavelet.")
        df.loc[is_prolonged_dead, column_name] = np.nan
        
    return df

if __name__ == "__main__":
    print("BẮT ĐẦU QUÉT VÀ LOẠI BỎ NHIỄU (MIQR + WAVELET)...")

    if not os.path.exists(raw_weather_path):
        print(f"Không tìm thấy thư mục dữ liệu đầu vào: {raw_weather_path}")
    else:
        for file_name in os.listdir(raw_weather_path):
            if file_name.endswith(".csv"):
                print(f"\nĐang xử lý: {file_name}")
                file_path = os.path.join(raw_weather_path, file_name)
                df = pd.read_csv(file_path)
                
                # Tích hợp logic xử lý chuẩn hóa ngày tháng
                if 'date' in df.columns:
                    df['date'] = pd.to_datetime(df['date'], errors='coerce').dt.tz_localize(None).dt.normalize()
                
                continuous_cols = ['temperature_2m_max', 'temperature_2m_min', 'et0_fao_evapotranspiration', 'vpd_max']
                
                for col in continuous_cols:
                    # 1. Cắt nhiễu đột biến
                    df = apply_miqr(df, col, window=15, k=3.0)
                    
                    # 2. Xử lý lỗi kẹt cảm biến
                    df = apply_wavelet_flatline_detection(df, col, width=5, dead_days_threshold=4)
                
                save_path = os.path.join(despiked_path, f"despiked_{file_name}")
                df.to_csv(save_path, index=False)

        print("\nHOÀN THÀNH BƯỚC LOẠI BỎ NHIỄU!")